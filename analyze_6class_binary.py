"""Phase 4: Binary model의 6-class fine-grained 분석

Binary 모델로 학습된 best_model.pth 을 load 해서,
test set 에서 예측을 수행한 뒤 "원래 defect class" 별로
recall / error breakdown 을 집계한다.

목적:
- binary 0.998+ 성능에서 여전히 틀리는 SPECIFIC 불량 타입 식별
- 다음 데이터/모델 개선 방향 결정

Usage:
    python analyze_6class_binary.py --log_dir logs/v9_fix_n700_s42
    python analyze_6class_binary.py --log_dir logs/v9_fix_n700_s42 --aggregate "logs/v9_fix_n700_s*"
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import timm
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

# train.py와 동일 구조 재사용 (import)
from train import ChartImageDataset, create_model


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_one_run(log_dir: Path, config, device):
    """단일 실행 폴더에서 best_model.pth 로 test 예측 + original_class 수집"""
    best_info_path = log_dir / "best_info.json"
    if not best_info_path.exists():
        return None

    with open(best_info_path, encoding="utf-8") as f:
        best_info = json.load(f)

    model_name = best_info.get("model_name", "convnextv2_tiny.fcmae_ft_in22k_in1k")
    input_size = best_info.get("input_size", [224, 224])
    normalize = best_info.get("normalize", {})
    input_mean = normalize.get("mean", [0.485, 0.456, 0.406])
    input_std = normalize.get("std", [0.229, 0.224, 0.225])

    # Data
    scen_df = pd.read_csv(config["dataset"]["scenarios_csv"])
    test_df = scen_df[scen_df["split"] == "test"]
    all_classes = config["dataset"]["classes"]

    tfm = transforms.Compose([
        transforms.Resize(tuple(input_size)),
        transforms.ToTensor(),
        transforms.Normalize(input_mean, input_std),
    ])
    ds = ChartImageDataset(
        Path(config["dataset"]["images_dir"]),
        test_df, all_classes, tfm, mode="binary"
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    # Model
    num_classes = 2  # binary
    weights_path = "weights/convnextv2_tiny_pretrained.pth" if "convnextv2_tiny" in model_name else None
    model = create_model(num_classes, model_name, device, weights_path, dropout=0.0)
    state = torch.load(str(log_dir / "best_model.pth"), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Predict
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())  # P(abnormal)
            all_labels.extend(labels.tolist())

    return {
        "log_dir": str(log_dir),
        "model_name": model_name,
        "original_classes": ds.original_classes,
        "preds": all_preds,
        "probs_abnormal": all_probs,
        "labels": all_labels,
    }


def summarize(runs):
    """여러 run 결과 병합해서 per-original-class 집계"""
    # 각 original class 별 샘플 수, FN(실제 abn → normal 예측), TN/TP
    per_class = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0,
                                       "fn_ids": [], "fp_ids": []})
    for run in runs:
        origs = run["original_classes"]
        preds = run["preds"]
        labels = run["labels"]
        for i, (orig, pred, label) in enumerate(zip(origs, preds, labels)):
            per_class[orig]["total"] += 1
            if pred == label:
                per_class[orig]["correct"] += 1
            else:
                per_class[orig]["errors"] += 1
                if label == 1 and pred == 0:  # FN: true abn → pred normal
                    per_class[orig]["fn_ids"].append(i)
                elif label == 0 and pred == 1:  # FP: true normal → pred abn
                    per_class[orig]["fp_ids"].append(i)

    print(f"\n{'='*70}")
    print(f"  6-CLASS FINE-GRAINED ANALYSIS (binary model)")
    print(f"{'='*70}")
    print(f"  Runs aggregated: {len(runs)}")
    for r in runs:
        print(f"    - {r['log_dir']}")
    print()
    print(f"  {'Original class':<22} {'N':>6} {'Correct':>8} {'Errors':>8} {'Recall':>8}")
    print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    cls_order = ["normal", "mean_shift", "standard_deviation", "spike", "drift", "context"]
    for cls in cls_order:
        if cls not in per_class:
            continue
        d = per_class[cls]
        recall = d["correct"] / max(d["total"], 1)
        err_detail = ""
        if cls == "normal" and d["fp_ids"]:
            err_detail = f" (FP {len(d['fp_ids'])})"
        elif cls != "normal" and d["fn_ids"]:
            err_detail = f" (FN {len(d['fn_ids'])})"
        print(f"  {cls:<22} {d['total']:>6} {d['correct']:>8} {d['errors']:>8} {recall:>8.4f}{err_detail}")

    # Abnormal 클래스 한정 aggregate
    abn_total = sum(per_class[c]["total"] for c in cls_order if c != "normal" and c in per_class)
    abn_correct = sum(per_class[c]["correct"] for c in cls_order if c != "normal" and c in per_class)
    nor_total = per_class.get("normal", {}).get("total", 0)
    nor_correct = per_class.get("normal", {}).get("correct", 0)
    print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'ABNORMAL (agg)':<22} {abn_total:>6} {abn_correct:>8} {abn_total-abn_correct:>8} "
          f"{abn_correct/max(abn_total,1):>8.4f}")
    print(f"  {'NORMAL':<22} {nor_total:>6} {nor_correct:>8} {nor_total-nor_correct:>8} "
          f"{nor_correct/max(nor_total,1):>8.4f}")
    print()

    return per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None,
                        help="단일 실행 폴더")
    parser.add_argument("--aggregate", type=str, default=None,
                        help="glob 패턴 (예: 'logs/v9_fix_n700_s*')")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.aggregate:
        log_dirs = [Path(p) for p in sorted(glob.glob(args.aggregate))]
    elif args.log_dir:
        log_dirs = [Path(args.log_dir)]
    else:
        raise SystemExit("--log_dir 또는 --aggregate 필요")

    print(f"  Device: {device}")
    print(f"  Target log dirs: {len(log_dirs)}")

    runs = []
    for ld in log_dirs:
        if not (ld / "best_model.pth").exists():
            print(f"  [SKIP] {ld} (no best_model.pth)")
            continue
        print(f"  Running {ld}...")
        result = evaluate_one_run(ld, config, device)
        if result is not None:
            runs.append(result)

    if not runs:
        print("No valid runs.")
        return

    summarize(runs)


if __name__ == "__main__":
    main()
