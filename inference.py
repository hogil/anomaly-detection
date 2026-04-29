"""
Production inference script

Input:  data/timeseries.csv + data/scenarios.csv (또는 fab 실전 데이터)
Output: inference_output/
  ├── normal/       → 정상 판정 display 이미지 (불량 표시 없음)
  ├── abnormal/     → 불량 판정 display 이미지
  └── predictions.csv  → 판정 리스트

Usage:
  python inference.py --model logs/v8seed_n2800_s42/best_model.pth --limit 20
  python inference.py --model logs/v8seed_n1400_s42/best_model.pth --output_dir infer_v8seed

파일명: {device}_{step}_{item}_{yymmddhh24}.png
  - yymmddhh24는 데이터의 x_col에서 추출 (datetime이면 포맷, numeric이면 chart_id 기반)
"""

import argparse
import shutil
import tempfile
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
from PIL import Image
from torchvision import transforms
from datetime import datetime

from src.data.image_renderer import ImageRenderer
from src.data.schema import highlighted_member as read_highlighted_member
from src.data.schema import legend_axis as read_legend_axis
from src.data.schema import members as read_members
from src.data.schema import target as read_target


def _build_model(model_name: str, num_classes: int, dropout: float, device):
    """train.py의 create_model과 동일한 구조 (pretrained=False, head만 교체)"""
    model = timm.create_model(model_name, pretrained=False)

    if hasattr(model, 'head') and hasattr(model.head, 'fc'):
        in_features = model.head.fc.in_features
        model.head.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 512),
            nn.ReLU(inplace=True), nn.Linear(512, num_classes),
        )
    elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 512),
            nn.ReLU(inplace=True), nn.Linear(512, num_classes),
        )
    elif hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 512),
            nn.ReLU(inplace=True), nn.Linear(512, num_classes),
        )
    else:
        in_features = model.get_classifier().in_features
        model.reset_classifier(0)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 512),
            nn.ReLU(inplace=True), nn.Linear(512, num_classes),
        )

    return model.to(device)


def _load_model_from_best_info(log_dir: Path, device):
    """best_info.json 읽어 hparams 추출 → 모델 생성 → best_model.pth 로드"""
    info_path = log_dir / "best_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"best_info.json not found: {info_path}")
    import json
    with open(info_path, encoding="utf-8") as f:
        bi = json.load(f)
    hp = bi["hparams"]
    model_name = hp.get("pretrained", "convnextv2_tiny.fcmae_ft_in22k_in1k")
    num_classes = hp.get("num_classes", 2)
    # dropout은 best_info에 없을 수 있음 (v8_init는 0.0)
    dropout = 0.0

    model = _build_model(model_name, num_classes, dropout, device)
    weights = torch.load(log_dir / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    classes = hp.get("classes", ["normal", "abnormal"])
    print(f"  Loaded: {log_dir}/best_model.pth")
    print(f"    model: {model_name}")
    print(f"    classes: {classes}")
    return model, classes


def _format_timestamp(x_val) -> str:
    """x 값 → yymmddhh24 문자열. datetime이면 포맷, 아니면 숫자 그대로."""
    try:
        if hasattr(x_val, 'strftime'):
            return x_val.strftime("%y%m%d%H")
        if hasattr(x_val, 'dtype') and np.issubdtype(x_val.dtype, np.datetime64):
            return pd.Timestamp(x_val).strftime("%y%m%d%H")
        # numeric fallback — just format as integer
        return f"t{int(float(x_val)):08d}"
    except Exception:
        return "unknown"


def _detect_x_col(ts_df: pd.DataFrame) -> str:
    for cand in ["time_index", "timestamp", "datetime", "date", "time"]:
        if cand in ts_df.columns:
            return cand
    return ts_df.columns[1]  # fallback: 2번째 컬럼


def run_inference(
    log_dir: str,
    data_dir: str = "data",
    output_dir: str = "inference_output",
    limit: int = None,
    split_filter: str = None,
    scenarios_file: str = None,
):
    # config
    with open("dataset.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    img_cfg = cfg.get("image", {})
    title_columns = img_cfg.get("title_columns", ["device", "step", "item"])

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # model
    log_path = Path(log_dir)
    model, classes = _load_model_from_best_info(log_path, device)

    # data
    data_path = Path(data_dir)
    ts_df = pd.read_csv(data_path / "timeseries.csv")
    sc_file = Path(scenarios_file) if scenarios_file else (data_path / "scenarios.csv")
    sc_df = pd.read_csv(sc_file)
    print(f"Scenarios: {sc_file}")
    x_col = _detect_x_col(ts_df)
    print(f"X col: {x_col} (dtype={ts_df[x_col].dtype})")

    if split_filter:
        sc_df = sc_df[sc_df["split"] == split_filter].reset_index(drop=True)
    if limit:
        sc_df = sc_df.head(limit)

    print(f"Charts to process: {len(sc_df)}")

    ts_grouped = {sid: grp for sid, grp in ts_df.groupby("chart_id")}

    # output dirs — train.py와 동일하게 시각 prefix 자동 부여
    # 사용자가 --output_dir my_run 으로 주면 my_run_YYMMDD_HHMMSS/ 로 만들어 덮어쓰기 방지
    raw_out = Path(output_dir)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    if raw_out.parent == Path("."):
        out_path = raw_out.with_name(f"{stamp}_{raw_out.name}")
    else:
        out_path = raw_out.parent / f"{stamp}_{raw_out.name}"
    out_path.mkdir(parents=True, exist_ok=False)
    print(f"Output: {out_path}")
    (out_path / "normal").mkdir()
    (out_path / "abnormal").mkdir()
    temp_dir = Path(tempfile.mkdtemp(prefix="infer_"))

    renderer = ImageRenderer(cfg)

    # transform (train eval과 동일)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    results = []
    for idx, row in sc_df.iterrows():
        sid = row["chart_id"]
        legend_axis = read_legend_axis(row)
        members = read_members(row)
        highlighted_member = read_highlighted_member(row)
        if not legend_axis or not members or not highlighted_member:
            continue

        sc_ts = ts_grouped.get(sid)
        if sc_ts is None:
            continue

        # fleet data 구성
        fleet_data = {}
        for mid in members:
            m = sc_ts[sc_ts[legend_axis].astype(str) == str(mid)].sort_values(x_col)
            if m.empty:
                continue
            fleet_data[mid] = (m[x_col].to_numpy(), m["value"].to_numpy())

        if not fleet_data or highlighted_member not in fleet_data:
            continue

        # 1. training image (224x224) 생성 → 모델 입력
        train_img_path = temp_dir / f"{sid}.png"
        target = read_target(row)
        renderer.render_overlay(fleet_data, highlighted_member, str(train_img_path), target=target)

        # 2. 예측
        img = Image.open(train_img_path).convert("RGB")
        x = val_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(probs).item())

        pred_class = classes[pred] if pred < len(classes) else str(pred)
        # binary: abnormal 확률만 저장 (p_normal = 1 - p_abnormal)
        try:
            abn_idx = classes.index("abnormal")
        except ValueError:
            abn_idx = 1  # fallback
        p_abnormal = float(probs[abn_idx].item())

        # 3. filename: p{pct}_{device}_{step}_{item}_{target}.png
        # p_abnormal 3자리 퍼센트 → 폴더 정렬 시 심각도 순. timestamp 대신 target 값으로 변경.
        p_pct = f"p{int(round(p_abnormal * 100)):03d}"
        name_parts = [str(row.get(c, "unk")) for c in title_columns]
        if isinstance(target, (int, float)):
            tgt_str = f"t{target:+.3f}".replace(".", "p").replace("+", "p").replace("-", "n")
        else:
            tgt_str = "tna"
        base_fname = f"{p_pct}_{'_'.join(name_parts)}_{tgt_str}.png"
        dest_dir = out_path / pred_class
        disp_path = dest_dir / base_fname
        # 충돌 시 suffix
        suffix = 1
        while disp_path.exists():
            base = base_fname.replace(".png", "")
            disp_path = dest_dir / f"{base}_{suffix}.png"
            suffix += 1

        # 5. display image 저장 (불량 marker 없음 — production은 label 모름)
        title_parts = [str(row.get(c, "")) for c in title_columns if row.get(c) is not None]
        title = " / ".join(title_parts) if title_parts else sid
        renderer.render_overlay_display(
            fleet_data, highlighted_member, str(disp_path),
            anomalous_ids=[],             # production: label 없음
            defect_start_idx=None,         # 불량 marker 없음
            title=title,
            x_label=x_col,
            target=target,
        )

        # 6. results
        result_row = {
            "chart_id": sid,
            "highlighted_member": highlighted_member,
            "predicted": pred_class,
            "p_abnormal": round(p_abnormal, 4),
            "target": target if isinstance(target, (int, float)) else "",
            "image_file": str(disp_path.relative_to(out_path)).replace("\\", "/"),
        }
        for c in title_columns:
            result_row[c] = row.get(c, "")
        if "split" in row:
            result_row["split"] = row["split"]
        if "class" in row:  # 합성 데이터의 ground truth (참고용)
            result_row["true_class"] = row["class"]
        results.append(result_row)

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(sc_df)} processed")

    # 정리
    shutil.rmtree(temp_dir)

    # CSV 저장 (p_abnormal 내림차순 정렬 — 심각한 불량 상단)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("p_abnormal", ascending=False).reset_index(drop=True)
    csv_path = out_path / "predictions.csv"
    results_df.to_csv(csv_path, index=False)

    # 두 리스트 모두 p_abnormal 내림차순. CSV 형식 (콤마 구분자 + 헤더).
    abn_df = results_df[results_df["predicted"] == "abnormal"].sort_values("p_abnormal", ascending=False)
    nor_df = results_df[results_df["predicted"] == "normal"].sort_values("p_abnormal", ascending=False)

    list_columns = [c for c in ("device", "step", "item", "target", "p_abnormal", "chart_id", "image_file") if c in results_df.columns]

    abn_df[list_columns].to_csv(out_path / "abnormal_list.txt", index=False)
    nor_df[list_columns].to_csv(out_path / "normal_list.txt", index=False)

    # 통합 텍스트도 CSV. 위쪽이 ABNORMAL, 아래쪽이 NORMAL (둘 다 p_abn 내림차순).
    txt_path = out_path / "predictions.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# ABNORMAL ({len(abn_df)}) — p_abnormal desc\n")
        f.write(",".join(list_columns) + "\n")
        for _, r in abn_df.iterrows():
            f.write(",".join(str(r[c]) for c in list_columns) + "\n")
        f.write(f"\n# NORMAL ({len(nor_df)}) — p_abnormal desc\n")
        f.write(",".join(list_columns) + "\n")
        for _, r in nor_df.iterrows():
            f.write(",".join(str(r[c]) for c in list_columns) + "\n")

    # 요약
    n_total = len(results_df)
    n_normal = int((results_df["predicted"] == "normal").sum())
    n_abnormal = int((results_df["predicted"] == "abnormal").sum())

    print(f"\n{'='*60}")
    print(f"  Inference complete")
    print(f"{'='*60}")
    print(f"  Total:    {n_total}")
    print(f"  Normal:   {n_normal} ({n_normal/n_total*100:.1f}%)")
    print(f"  Abnormal: {n_abnormal} ({n_abnormal/n_total*100:.1f}%)")
    print(f"\n  Output: {out_path}/")
    print(f"    normal/      : {n_normal} images")
    print(f"    abnormal/    : {n_abnormal} images")
    print(f"    predictions.csv")
    print(f"    predictions.txt   (combined)")
    print(f"    abnormal_list.txt ({n_abnormal} items)")
    print(f"    normal_list.txt   ({n_normal} items)")

    # ground truth 있으면 accuracy 표시 (합성 데이터)
    if "true_class" in results_df.columns:
        # binary 변환
        true_binary = results_df["true_class"].apply(lambda c: "normal" if c == "normal" else "abnormal")
        acc = (results_df["predicted"] == true_binary).mean()
        print(f"\n  (참고) GT 비교:")
        print(f"    Accuracy: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="학습 log 디렉토리 (e.g., logs/v8seed_n2800_s42)")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="inference_output")
    parser.add_argument("--limit", type=int, default=None,
                        help="처리할 chart 수 제한 (prototype)")
    parser.add_argument("--split", type=str, default=None,
                        choices=[None, "train", "val", "test"],
                        help="특정 split만 처리")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="scenarios CSV 파일 (default: {data_dir}/scenarios.csv)")
    args = parser.parse_args()

    run_inference(
        log_dir=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        split_filter=args.split,
        scenarios_file=args.scenarios,
    )


if __name__ == "__main__":
    main()
