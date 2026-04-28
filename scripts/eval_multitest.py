"""Multi-severity test evaluation: load each best_model.pth from vd080_bc runs
and evaluate on test sets at difficulty 0.5/0.7/0.8/0.9.

Produces:
  logs/paper_tables/multitest_severity_curve.csv
  logs/paper_tables/multitest_severity_curve.md
  validations/multitest_severity.txt

Per-model: (run_name, test_diff, test_F1, FN, FP, abn_R, nor_R)

Usage:
  python scripts/eval_multitest.py --prefix vd080_bc
  python scripts/eval_multitest.py --prefix vd080_bc --run-filter 's42'
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import helpers from train.py
import train as train_mod

TEST_CONFIGS = {
    "0.5": ("dataset_multitest05.yaml", "images_mt05", "data_mt05"),
    "0.7": ("dataset_multitest07.yaml", "images_mt07", "data_mt07"),
    "0.8": ("dataset.yaml",             "images_vd080", "data_vd080"),   # reference
    "0.9": ("dataset_multitest09.yaml", "images_mt09", "data_mt09"),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="vd080_bc", help="Run folder prefix")
    ap.add_argument("--run-filter", default=None, help="Substring filter for run names")
    ap.add_argument("--out-csv", default="logs/paper_tables/multitest_severity_curve.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def ensure_test_sets():
    missing = []
    for diff, (cfg, imgdir, _) in TEST_CONFIGS.items():
        test_dir = ROOT / imgdir / "test"
        if not test_dir.exists() or len(list(test_dir.glob("*/*.png"))) < 1500:
            missing.append(f"test_diff={diff} ({imgdir})")
    if missing:
        print("[multitest] missing datasets:", missing)
        print("[multitest] run: bash scripts/gen_multitest.sh")
        sys.exit(1)


def load_model(run_dir: Path, device):
    """Load best_model.pth from a run directory. Returns (model, classes, mode)."""
    cfg_used = run_dir / "train_config_used.yaml"
    with open(cfg_used, encoding="utf-8") as f:
        tcfg = yaml.safe_load(f)
    mode = tcfg.get("mode", "binary")
    data_cfg_path = run_dir / "data_config_used.yaml"
    with open(data_cfg_path, encoding="utf-8") as f:
        dcfg = yaml.safe_load(f)
    classes = dcfg["dataset"]["classes"]
    all_classes = classes if mode != "binary" else ["normal", "abnormal"]
    model_name = tcfg.get("model_name", "convnextv2_tiny.fcmae_ft_in22k_in1k")
    dropout = float(tcfg.get("dropout", 0.0))
    sdr = float(tcfg.get("stochastic_depth_rate", 0.0))
    model = train_mod.create_model(
        num_classes=len(all_classes), model_name=model_name, device=device,
        dropout=dropout, stochastic_depth_rate=sdr,
    )
    state = torch.load(run_dir / "best_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, all_classes, mode


def eval_run_on_difficulty(run_dir: Path, diff: str, device):
    cfg_path, imgdir, datadir = TEST_CONFIGS[diff]
    sc_df = pd.read_csv(ROOT / datadir / "scenarios.csv")
    test_df = sc_df[sc_df["split"] == "test"].reset_index(drop=True)

    model, all_classes, mode = load_model(run_dir, device)

    # Standard eval transform (must match train_mod.val_transform — use ImageNet norm)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = train_mod.ChartImageDataset(ROOT / imgdir, test_df, all_classes, tf, mode=mode)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0], batch[1]
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Binary-view confusion
    import numpy as np
    preds = np.asarray(all_preds)
    labs = np.asarray(all_labels)
    if mode == "binary":
        bp, bl = preds, labs
    else:
        bp = (preds != 0).astype(int)
        bl = (labs != 0).astype(int)
    cm = train_mod._compute_binary_confusion(bp, bl)
    tn, fn, fp, tp = cm["tn"], cm["fn"], cm["fp"], cm["tp"]
    abn_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    nor_r = tn / (tn + fp) if (tn + fp) > 0 else 0
    # binary F1 (class-agnostic binary view)
    prec_a = tp / (tp + fp) if (tp + fp) > 0 else 0
    prec_n = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1_a = 2 * prec_a * abn_r / (prec_a + abn_r) if (prec_a + abn_r) > 0 else 0
    f1_n = 2 * prec_n * nor_r / (prec_n + nor_r) if (prec_n + nor_r) > 0 else 0
    f1 = (f1_a + f1_n) / 2

    return {
        "f1": round(f1, 4),
        "abn_r": round(abn_r, 4),
        "nor_r": round(nor_r, 4),
        "fn": int(fn),
        "fp": int(fp),
        "tp": int(tp),
        "tn": int(tn),
    }


def main():
    args = parse_args()
    ensure_test_sets()
    device = torch.device(args.device)
    print(f"[multitest] device={device}")

    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    logs = ROOT / "logs"
    runs = sorted(
        d for d in logs.iterdir()
        if d.is_dir() and args.prefix in d.name
        and (d / "best_model.pth").exists()
        and (d / "best_info.json").exists()
    )
    if args.run_filter:
        runs = [d for d in runs if args.run_filter in d.name]
    print(f"[multitest] runs to evaluate: {len(runs)}")

    rows = []
    for i, run in enumerate(runs, 1):
        m = re.search(r"vd080_bc_(?:tie_)?gc([^_]+)_sw([^_]+)_n\d+_s(\d+)", run.name)
        if not m:
            continue
        gc, sw, seed = m.group(1), m.group(2), m.group(3)
        for diff in TEST_CONFIGS:
            try:
                r = eval_run_on_difficulty(run, diff, device)
            except Exception as e:
                print(f"  [err] {run.name} @ {diff}: {e}")
                continue
            rows.append({
                "run": run.name,
                "gc": gc,
                "sw": sw,
                "seed": int(seed),
                "test_diff": float(diff),
                **r,
            })
        print(f"[{i}/{len(runs)}] {run.name[:80]}")

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())) if rows else None
        if w:
            w.writeheader()
            w.writerows(rows)
    print(f"[multitest] wrote {out_csv} ({len(rows)} rows)")

    # Simple md summary
    if rows:
        import statistics as st
        from collections import defaultdict
        per_diff = defaultdict(list)
        for r in rows:
            per_diff[(r["gc"], r["sw"], r["test_diff"])].append(r)
        md_lines = ["# Multi-severity test evaluation\n"]
        md_lines.append("## Mean F1 / FN by (gc, sw, test_diff)")
        md_lines.append("")
        md_lines.append("| gc | sw | diff | n | F1_mean | FN_μ | FP_μ |")
        md_lines.append("|---|---|---|---|---|---|---|")
        for (gc, sw, df), lst in sorted(per_diff.items()):
            md_lines.append(
                f"| {gc} | {sw} | {df} | {len(lst)} | "
                f"{st.mean([x['f1'] for x in lst]):.4f} | "
                f"{st.mean([x['fn'] for x in lst]):.1f} | "
                f"{st.mean([x['fp'] for x in lst]):.1f} |"
            )
        md = "\n".join(md_lines)
        (out_csv.with_suffix(".md")).write_text(md, encoding="utf-8")
        print(f"[multitest] wrote {out_csv.with_suffix('.md')}")


if __name__ == "__main__":
    main()
