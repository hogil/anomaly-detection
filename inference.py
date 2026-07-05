"""
Production inference script

Input:  data/timeseries.csv + data/scenarios.csv (또는 fab 실전 데이터)
Output: inference_output/
  ├── normal/       → 정상 판정 display 이미지 (불량 표시 없음)
  ├── abnormal/     → 불량 판정 display 이미지
  └── predictions.csv  → 판정 리스트

3-phase 실행 (속도):
  1) 모델 입력 이미지 렌더링 — multiprocessing 병렬 (--workers)
  2) 배치 예측 — --batch-size 단위 + AMP (--precision fp16/bf16/fp32)
  3) display 이미지 렌더링 — 병렬 + 선택적 (--display-filter all/abnormal/none)

Usage:
  python inference.py --model logs/v8seed_n2800_s42/best_model.pth --limit 20
  python inference.py --model logs/v8seed_n1400_s42 --output_dir infer_v8seed
  python inference.py --model logs/<run> --workers 8 --display-filter abnormal

파일명: {device}_{step}_{item}_{yymmddhh24}.png
  - yymmddhh24는 데이터의 x_col에서 추출 (datetime이면 포맷, numeric이면 chart_id 기반)
"""

import argparse
import contextlib
import os
import shutil
import sys
import tempfile
import time
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from multiprocessing import Pool, cpu_count
from pathlib import Path
from PIL import Image
from torchvision import transforms
from datetime import datetime
from tqdm import tqdm

from src.data.image_renderer import ImageRenderer
from src.data.schema import highlighted_member as read_highlighted_member
from src.data.schema import legend_axis as read_legend_axis
from src.data.schema import members as read_members
from src.data.schema import target as read_target

TQDM_DISABLE = not sys.stderr.isatty()


def _build_model(
    model_name: str,
    num_classes: int,
    dropout: float,
    device,
    stochastic_depth_rate: float = 0.0,
):
    """train.py의 create_model과 동일한 구조 (pretrained=False, head만 교체)"""
    model = timm.create_model(
        model_name,
        pretrained=False,
        drop_path_rate=stochastic_depth_rate,
    )

    if hasattr(model, 'head') and hasattr(model.head, 'fc'):
        in_features = getattr(model.head.fc, "in_features", None) or getattr(model, "num_features", None)
        if in_features is None:
            raise RuntimeError(f"{model_name}의 head 입력 차원을 찾지 못했습니다.")
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
        in_features = getattr(model.classifier, "in_features", None) or getattr(model, "num_features", None)
        if in_features is None:
            raise RuntimeError(f"{model_name}의 classifier 입력 차원을 찾지 못했습니다.")
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 512),
            nn.ReLU(inplace=True), nn.Linear(512, num_classes),
        )
    else:
        in_features = getattr(model.get_classifier(), "in_features", None) or getattr(model, "num_features", None)
        if in_features is None:
            raise RuntimeError(f"{model_name}의 classifier 입력 차원을 찾지 못했습니다.")
        model.reset_classifier(0)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 512),
            nn.ReLU(inplace=True), nn.Linear(512, num_classes),
        )

    return model.to(device)


def _load_model_from_best_info(log_dir: Path, device):
    """best_info.json 읽어 hparams 추출 → 모델 생성 → best_model.pth 로드"""
    log_dir = Path(log_dir)
    if log_dir.is_file():
        log_dir = log_dir.parent
    info_path = log_dir / "best_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"best_info.json not found: {info_path}")
    import json
    with open(info_path, encoding="utf-8") as f:
        bi = json.load(f)
    hp = bi["hparams"]

    train_cfg = {}
    train_cfg_path = log_dir / "train_config_used.yaml"
    if train_cfg_path.exists():
        with open(train_cfg_path, encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f) or {}

    model_name = (
        train_cfg.get("model_name")
        or hp.get("model_name")
        or hp.get("pretrained")
        or "convnextv2_tiny.fcmae_ft_in22k_in1k"
    )
    num_classes = hp.get("num_classes", 2)
    dropout = float(train_cfg.get("dropout", hp.get("dropout", 0.0)))
    stochastic_depth_rate = float(
        train_cfg.get("stochastic_depth_rate", hp.get("stochastic_depth_rate", 0.0))
    )

    model = _build_model(model_name, num_classes, dropout, device, stochastic_depth_rate)
    weights = torch.load(log_dir / "best_model.pth", map_location=device)
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


def _norm_key(value) -> str:
    """chart 종류 매칭용 정규화 — 공백 제거 + 숫자형 '9100.0' → '9100'."""
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def load_prob_limits(csv_path, title_columns):
    """차트 종류별 p_abnormal 판정 한계 CSV 로드.

    CSV 컬럼: title_columns(device/step/item 등) 중 일부/전부 + prob_limit.
    빈 값 또는 '*' 는 wildcard. 채워진 필드가 많은(구체적인) 행이 우선, 동률이면 파일 순서.
    판정: p_abnormal >= prob_limit → abnormal (limit 낮음 = 민감/FN↓, 높음 = 둔감/FP↓).
    """
    df = pd.read_csv(csv_path, dtype=str)
    if "prob_limit" not in df.columns:
        raise ValueError(f"prob_limit 컬럼이 없습니다: {csv_path}")
    key_cols = [c for c in title_columns if c in df.columns]
    if not key_cols:
        raise ValueError(f"차트 종류 컬럼({', '.join(title_columns)})이 하나도 없습니다: {csv_path}")
    rules = []
    for _, r in df.iterrows():
        cond = {}
        for c in key_cols:
            v = _norm_key(r[c]) if pd.notna(r[c]) else ""
            if v not in ("", "*"):
                cond[c] = v
        rules.append((cond, float(r["prob_limit"])))
    rules.sort(key=lambda rule: len(rule[0]), reverse=True)
    print(f"  Prob limits: {len(rules)} rules from {csv_path} (keys: {', '.join(key_cols)})")
    return rules


def resolve_prob_limit(row, rules):
    """row(dict)에 매칭되는 가장 구체적인 prob_limit. 없으면 None."""
    for cond, limit in rules:
        if all(_norm_key(row.get(k, "")) == v for k, v in cond.items()):
            return limit
    return None


# ---------------------------------------------------------------------------
# 병렬 렌더링 worker — generate_images.py와 같은 패턴 (initializer + 캐시)
# ---------------------------------------------------------------------------
_worker_cache = {}


def _build_fleet_data(sc_ts, legend_axis, members, x_col):
    fleet_data = {}
    for mid in members:
        m = sc_ts[sc_ts[legend_axis].astype(str) == str(mid)].sort_values(x_col)
        if m.empty:
            continue
        fleet_data[mid] = (m[x_col].to_numpy(), m["value"].to_numpy())
    return fleet_data


def _init_render_worker(ts_pickle_path: str, cfg: dict, x_col: str):
    import pickle
    with open(ts_pickle_path, "rb") as f:
        _worker_cache["ts_grouped"] = pickle.load(f)
    _worker_cache["renderer"] = ImageRenderer(cfg)
    _worker_cache["x_col"] = x_col


def _render_input_task(task: dict):
    """모델 입력 이미지 1장 렌더링. 반환 (idx, ok)."""
    row = task["row"]
    x_col = _worker_cache["x_col"]
    legend_axis = read_legend_axis(row)
    members = read_members(row)
    highlighted_member = read_highlighted_member(row)
    if not legend_axis or not members or not highlighted_member:
        return task["idx"], False
    sc_ts = _worker_cache["ts_grouped"].get(row["chart_id"])
    if sc_ts is None:
        return task["idx"], False
    fleet_data = _build_fleet_data(sc_ts, legend_axis, members, x_col)
    if not fleet_data or highlighted_member not in fleet_data:
        return task["idx"], False
    _worker_cache["renderer"].render_overlay(
        fleet_data, highlighted_member, task["img_path"], target=read_target(row))
    return task["idx"], True


def _render_display_task(task: dict):
    """display 이미지 1장 렌더링. fleet_data는 worker의 ts_grouped에서 재구성."""
    row = task["row"]
    x_col = _worker_cache["x_col"]
    legend_axis = read_legend_axis(row)
    members = read_members(row)
    highlighted_member = read_highlighted_member(row)
    sc_ts = _worker_cache["ts_grouped"].get(row["chart_id"])
    fleet_data = _build_fleet_data(sc_ts, legend_axis, members, x_col)
    _worker_cache["renderer"].render_overlay_display(
        fleet_data, highlighted_member, task["disp_path"],
        anomalous_ids=[],             # production: label 없음
        defect_start_idx=None,         # 불량 marker 없음
        title=task["title"],
        x_label=x_col,
        target=read_target(row),
    )
    return task["idx"]


def run_inference(
    log_dir: str,
    data_dir: str = "data",
    output_dir: str = "inference_output",
    limit: int = None,
    split_filter: str = None,
    scenarios_file: str = None,
    image_dir_name: str = "model_inputs",
    display_dir_name: str = "display",
    save_model_inputs: bool = False,
    workers: int = 0,
    batch_size: int = 64,
    display_filter: str = "all",
    precision: str = "fp16",
    use_compile: bool = False,
    prob_limit_csv: str = None,
):
    # config
    with open("dataset.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    img_cfg = cfg.get("image", {})
    title_columns = img_cfg.get("title_columns", ["device", "step", "item"])

    # 차트 종류별 판정 한계 (없으면 기존 argmax 판정)
    prob_limit_rules = None
    if prob_limit_csv:
        prob_limit_rules = load_prob_limits(prob_limit_csv, title_columns)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # model — channels_last (ConvNeXt 계열 NHWC conv 커널 활용)
    log_path = Path(log_dir)
    model, classes = _load_model_from_best_info(log_path, device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if use_compile:
        if hasattr(torch, "compile") and device.type == "cuda":
            try:
                model = torch.compile(model)
                print("  torch.compile: enabled")
            except Exception as e:
                print(f"  torch.compile failed, falling back: {e}")
        else:
            print("  torch.compile requested but unsupported (cpu or old torch)")

    # precision — train.py eval 기본이 fp16 autocast이므로 fp16이 학습 시 평가와 동일 경로
    if device.type == "cuda" and precision == "fp16":
        amp_dtype = torch.float16
    elif device.type == "cuda" and precision == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None
    print(f"  Precision: {'fp32' if amp_dtype is None else precision}")

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
    display_root = out_path / display_dir_name
    (display_root / "normal").mkdir(parents=True)
    (display_root / "abnormal").mkdir()
    if save_model_inputs:
        inputs_root = out_path / image_dir_name
        (inputs_root / "normal").mkdir(parents=True)
        (inputs_root / "abnormal").mkdir()
    else:
        inputs_root = None
    temp_dir = Path(tempfile.mkdtemp(prefix="infer_"))

    # transform (train eval과 동일)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rows = sc_df.to_dict(orient="records")

    # auto worker: 작은 작업에서는 worker 스폰 비용(Windows spawn은 worker당 torch 재import)이
    # 지배하므로 작업량에 비례해 제한. 명시 지정 시 그대로 사용.
    if workers <= 0:
        workers = min(max(1, cpu_count() - 1), max(1, len(rows) // 25))
    print(f"Render workers: {workers}")
    tasks = [
        {"idx": i, "row": row, "img_path": str(temp_dir / f"{i:06d}_{row['chart_id']}.png")}
        for i, row in enumerate(rows)
    ]

    pool = None
    ts_pickle_path = None
    if workers == 1:
        _worker_cache["ts_grouped"] = ts_grouped
        _worker_cache["renderer"] = ImageRenderer(cfg)
        _worker_cache["x_col"] = x_col
    else:
        import pickle
        # 처리 대상 chart만 pickle (limit 사용 시 worker 초기화 비용 대폭 감소)
        needed = {row["chart_id"] for row in rows}
        ts_subset = {sid: grp for sid, grp in ts_grouped.items() if sid in needed}
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(ts_subset, f, protocol=pickle.HIGHEST_PROTOCOL)
            ts_pickle_path = f.name
        pool = Pool(processes=workers, initializer=_init_render_worker,
                    initargs=(ts_pickle_path, cfg, x_col))

    try:
        # -------- Phase 1: 모델 입력 이미지 렌더링 (병렬) --------
        t0 = time.perf_counter()
        ok = {}
        if pool is None:
            for task in tqdm(tasks, desc="render inputs", disable=TQDM_DISABLE):
                idx, good = _render_input_task(task)
                ok[idx] = good
        else:
            for idx, good in tqdm(pool.imap_unordered(_render_input_task, tasks, chunksize=4),
                                  total=len(tasks), desc="render inputs", disable=TQDM_DISABLE):
                ok[idx] = good
        render_input_sec = time.perf_counter() - t0
        valid_indices = [i for i in range(len(rows)) if ok.get(i)]

        # -------- Phase 2: 배치 예측 (AMP) --------
        t0 = time.perf_counter()
        probs_map = {}
        for start in range(0, len(valid_indices), batch_size):
            chunk = valid_indices[start:start + batch_size]
            tensors = [val_transform(Image.open(tasks[i]["img_path"]).convert("RGB")) for i in chunk]
            x = torch.stack(tensors).to(device)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            amp_ctx = (torch.amp.autocast("cuda", dtype=amp_dtype)
                       if amp_dtype is not None else contextlib.nullcontext())
            with torch.no_grad(), amp_ctx:
                probs_batch = F.softmax(model(x), dim=1).float().cpu()
            for i, p in zip(chunk, probs_batch):
                probs_map[i] = p
        predict_sec = time.perf_counter() - t0

        # -------- Phase 3: 판정 정리 + display 렌더링 (선택적, 병렬) --------
        try:
            abn_idx = classes.index("abnormal")
        except ValueError:
            abn_idx = 1  # fallback

        results = []
        display_tasks = []
        assigned_paths = set()
        binary_classes = "normal" in classes and "abnormal" in classes
        if prob_limit_rules and not binary_classes:
            print("  [warn] prob_limit CSV는 binary(normal/abnormal) 모델에서만 적용됩니다 — 무시함")
            prob_limit_rules = None

        for i in valid_indices:
            row = rows[i]
            probs = probs_map[i]
            # binary: abnormal 확률만 저장 (p_normal = 1 - p_abnormal)
            p_abnormal = float(probs[abn_idx].item())
            prob_limit = resolve_prob_limit(row, prob_limit_rules) if prob_limit_rules else None
            if prob_limit is not None:
                # 차트 종류별 한계: p_abnormal >= limit → abnormal
                pred_class = "abnormal" if p_abnormal >= prob_limit else "normal"
            else:
                pred = int(torch.argmax(probs).item())
                pred_class = classes[pred] if pred < len(classes) else str(pred)
            target = read_target(row)

            # filename: p{pct}_{device}_{step}_{item}_{target}.png
            # p_abnormal 3자리 퍼센트 → 폴더 정렬 시 심각도 순. timestamp 대신 target 값으로 변경.
            p_pct = f"p{int(round(p_abnormal * 100)):03d}"
            name_parts = [str(row.get(c, "unk")) for c in title_columns]
            if isinstance(target, (int, float)):
                tgt_str = f"t{target:+.3f}".replace(".", "p").replace("+", "p").replace("-", "n")
            else:
                tgt_str = "tna"
            base_fname = f"{p_pct}_{'_'.join(name_parts)}_{tgt_str}.png"
            dest_dir = display_root / pred_class
            disp_path = dest_dir / base_fname
            # 충돌 시 suffix (렌더링 전이므로 파일 대신 예약 set으로 확인)
            suffix = 1
            while str(disp_path) in assigned_paths or disp_path.exists():
                base = base_fname.replace(".png", "")
                disp_path = dest_dir / f"{base}_{suffix}.png"
                suffix += 1
            assigned_paths.add(str(disp_path))

            want_display = (display_filter == "all"
                            or (display_filter == "abnormal" and pred_class == "abnormal"))
            if want_display:
                title_parts = [str(row.get(c, "")) for c in title_columns if row.get(c) is not None]
                title = " / ".join(title_parts) if title_parts else row["chart_id"]
                display_tasks.append({
                    "idx": i, "row": row, "disp_path": str(disp_path), "title": title,
                })

            # 모델 input 이미지 (224x224) 도 보고 싶으면 같이 저장
            if inputs_root is not None:
                shutil.copyfile(tasks[i]["img_path"], str(inputs_root / pred_class / disp_path.name))

            result_row = {
                "chart_id": row["chart_id"],
                "highlighted_member": read_highlighted_member(row),
                "predicted": pred_class,
                "p_abnormal": round(p_abnormal, 4),
                "prob_limit": "" if prob_limit is None else prob_limit,
                "target": target if isinstance(target, (int, float)) else "",
                "image_file": (str(disp_path.relative_to(out_path)).replace("\\", "/")
                               if want_display else ""),
                "model_input_file": (
                    str((inputs_root / pred_class / disp_path.name).relative_to(out_path)).replace("\\", "/")
                    if inputs_root is not None else ""
                ),
            }
            for c in title_columns:
                result_row[c] = row.get(c, "")
            if "split" in row:
                result_row["split"] = row["split"]
            if "class" in row:  # 합성 데이터의 ground truth (참고용)
                result_row["true_class"] = row["class"]
            results.append(result_row)

        t0 = time.perf_counter()
        if display_tasks:
            if pool is None:
                for task in tqdm(display_tasks, desc="render display", disable=TQDM_DISABLE):
                    _render_display_task(task)
            else:
                for _ in tqdm(pool.imap_unordered(_render_display_task, display_tasks, chunksize=4),
                              total=len(display_tasks), desc="render display", disable=TQDM_DISABLE):
                    pass
        display_sec = time.perf_counter() - t0
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if ts_pickle_path is not None:
            os.unlink(ts_pickle_path)
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

    list_columns = [c for c in ("device", "step", "item", "target", "p_abnormal", "prob_limit", "chart_id", "image_file") if c in results_df.columns]

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
    print(f"  Timing:   render_inputs={render_input_sec:.1f}s ({len(valid_indices)} charts, {workers} workers)")
    print(f"            predict={predict_sec:.1f}s (batch={batch_size}, {'fp32' if amp_dtype is None else precision})")
    print(f"            render_display={display_sec:.1f}s ({len(display_tasks)} images, filter={display_filter})")
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
                        help="학습 log 디렉토리 또는 direct best_model.pth 경로")
    parser.add_argument("--data_dir", default="data",
                        help="timeseries.csv + scenarios.csv 가 있는 폴더")
    parser.add_argument("--output_dir", default="inference_output",
                        help="결과 저장 위치. 시각 prefix 자동 부여")
    parser.add_argument("--limit", type=int, default=None,
                        help="처리할 chart 수 제한 (prototype)")
    parser.add_argument("--split", type=str, default=None,
                        choices=[None, "train", "val", "test"],
                        help="scenarios.csv 의 split 컬럼으로 train/val/test 중 하나만 추론")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="scenarios CSV 파일 (default: {data_dir}/scenarios.csv)")
    parser.add_argument("--display-dir", dest="display_dir", default="display",
                        help="display 이미지 sub-folder 이름 (default: display). 결과 폴더 안에서 <name>/{normal,abnormal}/ 로 저장")
    parser.add_argument("--image-dir", dest="image_dir", default="model_inputs",
                        help="모델 input(224x224) 이미지 sub-folder 이름. --save-model-inputs 와 같이 써야 저장됨")
    parser.add_argument("--save-model-inputs", action="store_true",
                        help="display 외에 모델이 본 224x224 input 이미지도 같이 저장")
    parser.add_argument("--workers", type=int, default=0,
                        help="렌더링 병렬 worker 수 (0=auto, CPU 코어 수-1; 1=직렬)")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64,
                        help="예측 배치 크기")
    parser.add_argument("--display-filter", dest="display_filter", default="all",
                        choices=["all", "abnormal", "none"],
                        help="display 이미지 렌더링 대상. abnormal=불량 판정만 (대부분 normal인 현업 데이터에서 큰 속도 이득)")
    parser.add_argument("--precision", default="fp16",
                        choices=["fp16", "bf16", "fp32"],
                        help="예측 정밀도 (CUDA만 적용, CPU는 fp32). train.py eval 기본과 같은 fp16 권장")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile 활성화 (대량 추론 + Linux GPU 서버에서만 이득, warmup 수십 초)")
    parser.add_argument("--prob-limit-csv", dest="prob_limit_csv", default=None,
                        help="차트 종류별 판정 한계 CSV (컬럼: device/step/item 일부 + prob_limit, 빈칸/*=wildcard). "
                             "p_abnormal >= prob_limit 이면 abnormal. 없으면 기존 argmax 판정")
    args = parser.parse_args()

    run_inference(
        log_dir=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        split_filter=args.split,
        scenarios_file=args.scenarios,
        image_dir_name=args.image_dir,
        display_dir_name=args.display_dir,
        save_model_inputs=args.save_model_inputs,
        workers=args.workers,
        batch_size=args.batch_size,
        display_filter=args.display_filter,
        precision=args.precision,
        use_compile=args.compile,
        prob_limit_csv=args.prob_limit_csv,
    )


if __name__ == "__main__":
    main()
