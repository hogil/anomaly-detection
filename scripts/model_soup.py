"""Greedy Model Soup over a group of runs.

Input: list of run folders (each containing best_model.pth + best_info.json).
Procedure:
1. Rank by val_f1 descending.
2. Greedy-add: initialize soup = best-ranked weights.
   For each next candidate, try average(soup, candidate); if val_f1 improves, keep.
3. Evaluate final soup on test set.

Usage:
  python scripts/model_soup.py --pattern 'vd080_bc_gc0p5_sw1raw' --out logs/soup_v1
"""
import argparse
import copy
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as train_mod
import yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="Substring to match run folder names")
    ap.add_argument("--out", default=None, help="Output dir for soup weights + report")
    ap.add_argument("--data-config", default="dataset.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_model_instance(run_dir: Path, device):
    cfg_used = run_dir / "train_config_used.yaml"
    with open(cfg_used, encoding="utf-8") as f:
        tcfg = yaml.safe_load(f)
    data_cfg_path = run_dir / "data_config_used.yaml"
    with open(data_cfg_path, encoding="utf-8") as f:
        dcfg = yaml.safe_load(f)
    classes = dcfg["dataset"]["classes"]
    mode = tcfg.get("mode", "binary")
    all_classes = classes if mode != "binary" else ["normal", "abnormal"]
    model_name = tcfg.get("model_name", "convnextv2_tiny.fcmae_ft_in22k_in1k")
    dropout = float(tcfg.get("dropout", 0.0))
    sdr = float(tcfg.get("stochastic_depth_rate", 0.0))
    model = train_mod.create_model(
        num_classes=len(all_classes), model_name=model_name, device=device,
        dropout=dropout, stochastic_depth_rate=sdr,
    )
    return model, all_classes, mode, dcfg


def evaluate_on_split(model, scenarios_df, split, image_dir, all_classes, mode, device):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    df = scenarios_df[scenarios_df["split"] == split].reset_index(drop=True)
    ds = train_mod.ChartImageDataset(Path(image_dir), df, all_classes, tf, mode=mode)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    import numpy as np
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0], batch[1]
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    preds = np.asarray(all_preds); labs = np.asarray(all_labels)
    if mode == "binary":
        bp, bl = preds, labs
    else:
        bp = (preds != 0).astype(int); bl = (labs != 0).astype(int)
    cm = train_mod._compute_binary_confusion(bp, bl)
    tn, fn, fp, tp = cm["tn"], cm["fn"], cm["fp"], cm["tp"]
    abn_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    nor_r = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec_a = tp / (tp + fp) if (tp + fp) > 0 else 0
    prec_n = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1_a = 2 * prec_a * abn_r / (prec_a + abn_r) if (prec_a + abn_r) > 0 else 0
    f1_n = 2 * prec_n * nor_r / (prec_n + nor_r) if (prec_n + nor_r) > 0 else 0
    f1 = (f1_a + f1_n) / 2
    return {"f1": f1, "fn": int(fn), "fp": int(fp), "tn": int(tn), "tp": int(tp)}


def avg_state_dicts(state_dicts):
    """Uniform average of state_dicts."""
    n = len(state_dicts)
    keys = list(state_dicts[0].keys())
    out = OrderedDict()
    for k in keys:
        stacked = torch.stack([sd[k].float() for sd in state_dicts])
        out[k] = stacked.mean(dim=0).to(state_dicts[0][k].dtype)
    return out


def main():
    args = parse_args()
    device = torch.device(args.device)
    logs = ROOT / "logs"
    candidates = sorted(d for d in logs.iterdir()
                       if d.is_dir() and args.pattern in d.name
                       and (d / "best_model.pth").exists()
                       and (d / "best_info.json").exists())
    if not candidates:
        print(f"No runs matching '{args.pattern}'")
        return 1
    print(f"[soup] {len(candidates)} candidates matching '{args.pattern}'")

    # Rank by val_f1
    def val_f1(d):
        return json.load(open(d / "best_info.json")).get("val_f1", 0.0)
    candidates.sort(key=val_f1, reverse=True)
    for d in candidates:
        vf = val_f1(d)
        print(f"  val_f1={vf:.4f}  {d.name}")

    # Use first candidate to load model architecture + data config
    model, all_classes, mode, dcfg = load_model_instance(candidates[0], device)
    data_dir = Path(dcfg["output"]["data_dir"])
    image_dir = Path(dcfg["output"]["image_dir"])
    scenarios_df = pd.read_csv(data_dir / "scenarios.csv")

    # Load all state dicts
    state_dicts = [torch.load(d / "best_model.pth", map_location=device) for d in candidates]

    # Evaluate each individually on val + test to know baseline
    results = []
    for d, sd in zip(candidates, state_dicts):
        model.load_state_dict(sd)
        vr = evaluate_on_split(model, scenarios_df, "val", image_dir, all_classes, mode, device)
        tr = evaluate_on_split(model, scenarios_df, "test", image_dir, all_classes, mode, device)
        results.append({"run": d.name, "val_f1": vr["f1"], "test_f1": tr["f1"],
                        "test_fn": tr["fn"], "test_fp": tr["fp"]})
        print(f"  [single] {d.name[-40:]}: val_F1={vr['f1']:.4f} test_F1={tr['f1']:.4f} FN={tr['fn']} FP={tr['fp']}")

    # Greedy soup: start with top-1, add if val improves
    soup_state = state_dicts[0]
    soup_members = [candidates[0].name]
    model.load_state_dict(soup_state)
    soup_val = evaluate_on_split(model, scenarios_df, "val", image_dir, all_classes, mode, device)
    best_val_f1 = soup_val["f1"]
    print(f"\n[soup] init with top-1 val_F1={best_val_f1:.4f}")

    for i in range(1, len(state_dicts)):
        candidate_sd = state_dicts[i]
        # try new soup with this added (uniform avg of all selected)
        pool = [state_dicts[j] for j, d in enumerate(candidates) if d.name in soup_members] + [candidate_sd]
        trial_sd = avg_state_dicts(pool)
        model.load_state_dict(trial_sd)
        trial_val = evaluate_on_split(model, scenarios_df, "val", image_dir, all_classes, mode, device)
        if trial_val["f1"] >= best_val_f1:
            best_val_f1 = trial_val["f1"]
            soup_state = trial_sd
            soup_members.append(candidates[i].name)
            print(f"[soup] + {candidates[i].name[-40:]}  -> val_F1={best_val_f1:.4f} (ADDED, {len(soup_members)} members)")
        else:
            print(f"[soup] - {candidates[i].name[-40:]}  -> val_F1={trial_val['f1']:.4f} (rejected)")

    # Final soup on test
    model.load_state_dict(soup_state)
    final = evaluate_on_split(model, scenarios_df, "test", image_dir, all_classes, mode, device)
    print(f"\n[SOUP FINAL] members={len(soup_members)} test_F1={final['f1']:.4f} FN={final['fn']} FP={final['fp']}")

    # Save
    out_dir = Path(args.out) if args.out else ROOT / "logs" / f"soup_{args.pattern}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(soup_state, out_dir / "soup_weights.pth")
    report = {
        "pattern": args.pattern,
        "candidates": [r["run"] for r in results],
        "single_results": results,
        "soup_members": soup_members,
        "soup_n": len(soup_members),
        "soup_test_f1": final["f1"],
        "soup_test_fn": final["fn"],
        "soup_test_fp": final["fp"],
        "soup_val_f1": best_val_f1,
        "baseline_best_single_test_f1": max(r["test_f1"] for r in results),
        "baseline_best_single_test_fn_fp_err": min((r["test_fn"] + r["test_fp"]) for r in results),
    }
    with open(out_dir / "soup_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[soup] saved to {out_dir}/")


if __name__ == "__main__":
    sys.exit(main() or 0)
