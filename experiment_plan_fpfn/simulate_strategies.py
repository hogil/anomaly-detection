"""
Post-hoc simulation: 저장된 per-epoch checkpoints 를 이용해 18 strategies 비교.

조합:
- metric smoothing:   raw | mean3 | median3    (3)
- weight aggregation: raw | mean3 | median3    (3)
- tie handling:       strict | tie_save        (2)
= 18 strategies per batch size run

Usage:
    python experiment_plan_fpfn/simulate_strategies.py
    python experiment_plan_fpfn/simulate_strategies.py --runs v9_phase1d_bs32_n700_s1
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_tie import create_model, ChartImageDataset, _compute_binary_confusion  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"


# =============================================================================
# Helpers
# =============================================================================
def load_history(log_dir: Path):
    """history.json → per-epoch dict list"""
    h_file = log_dir / "history.json"
    if not h_file.exists():
        return None
    h = json.loads(h_file.read_text(encoding="utf-8"))
    if isinstance(h, list):
        return h
    n = len(next(iter(h.values())))
    return [{k: v[i] for k, v in h.items()} for i in range(n)]


def smoothed_val(val_series, method, window):
    """List[val_f1] → smoothed version."""
    out = []
    for i in range(len(val_series)):
        start = max(0, i - window + 1)
        win = val_series[start : i + 1]
        if method == "median":
            out.append(float(np.median(win)))
        elif method == "mean":
            out.append(float(np.mean(win)))
        else:
            out.append(val_series[i])
    return out


def select_best_epoch(val_smoothed, tie_save, min_epoch=1):
    """
    Given smoothed val metric per epoch, return the selected best epoch (1-indexed).

    tie_save:
      False (strict): 첫 max 값에서 freeze
      True  (tie):    equal 값 나올 때마다 update (last-tie wins)
    """
    best_val = -float("inf")
    best_ep = None
    for i, v in enumerate(val_smoothed, start=1):
        if i < min_epoch:
            continue
        if v > best_val or (tie_save and v == best_val):
            best_val = v
            best_ep = i
    return best_ep


def aggregate_weights(state_dicts, method):
    """List[state_dict] → aggregated state_dict."""
    if len(state_dicts) == 1 or method == "raw":
        return copy.deepcopy(state_dicts[-1])

    result = {}
    for key in state_dicts[0]:
        v0 = state_dicts[0][key]
        if not torch.is_tensor(v0):
            result[key] = v0
            continue
        if v0.dtype in (torch.long, torch.int32, torch.int64, torch.int16, torch.int8, torch.bool):
            # int 는 집계 불가 → 최신값 사용
            result[key] = state_dicts[-1][key].clone()
            continue
        stacked = torch.stack([s[key].float() for s in state_dicts])
        if method == "median":
            result[key] = stacked.median(dim=0)[0].to(v0.dtype)
        elif method == "mean":
            result[key] = stacked.mean(dim=0).to(v0.dtype)
        else:
            result[key] = state_dicts[-1][key].clone()
    return result


# =============================================================================
# Strategy matrix
# =============================================================================
METRIC_METHODS = ["raw", "mean3", "median3"]
WEIGHT_METHODS = ["raw", "mean3", "median3"]
TIE_MODES = ["strict", "tie"]


def parse_method_window(tag):
    if tag == "raw":
        return "raw", 1
    # "mean3", "median3"
    if tag.startswith("mean"):
        return "mean", int(tag[4:])
    if tag.startswith("median"):
        return "median", int(tag[6:])
    return "raw", 1


def simulate_one_strategy(history, checkpoints_dir, classes, eval_fn,
                          metric_tag, weight_tag, tie_mode):
    """Return dict: strategy, selected_ep, val_f1, test_f1, fn, fp."""
    # 1. metric smoothing
    val_f1_list = [e.get("val_f1", 0) for e in history]
    m_method, m_window = parse_method_window(metric_tag)
    val_smoothed = smoothed_val(val_f1_list, m_method, m_window)

    # 2. select best epoch
    tie_save = (tie_mode == "tie")
    best_ep = select_best_epoch(val_smoothed, tie_save=tie_save, min_epoch=1)
    if best_ep is None:
        return None

    # 3. weight aggregation (last N epochs ending at best_ep)
    w_method, w_window = parse_method_window(weight_tag)
    start_ep = max(1, best_ep - w_window + 1)
    ckpt_eps = list(range(start_ep, best_ep + 1))
    ckpt_paths = [checkpoints_dir / f"ep_{ep:03d}.pth" for ep in ckpt_eps]
    if not all(p.exists() for p in ckpt_paths):
        return None

    state_dicts = [torch.load(p, map_location="cpu", weights_only=True) for p in ckpt_paths]
    aggregated = aggregate_weights(state_dicts, w_method)

    # 4. evaluate
    test_metrics = eval_fn(aggregated)

    return {
        "strategy": f"m_{metric_tag}+w_{weight_tag}+{tie_mode}",
        "metric": metric_tag,
        "weight": weight_tag,
        "tie": tie_mode,
        "selected_ep": best_ep,
        "ckpt_eps": ckpt_eps,
        "val_f1_smoothed": round(val_smoothed[best_ep - 1], 4),
        "val_f1_raw": round(val_f1_list[best_ep - 1], 4),
        "test_f1": round(test_metrics["test_f1"], 4),
        "abn_R": round(test_metrics["abn_R"], 4),
        "nor_R": round(test_metrics["nor_R"], 4),
        "fn": test_metrics["fn"],
        "fp": test_metrics["fp"],
    }


def make_eval_fn(model_template, test_loader, device, amp_dtype=torch.float16):
    """Given a state_dict → return test metrics dict."""
    def _eval(state_dict):
        model = copy.deepcopy(model_template)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to(device)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                images = images.to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(images)
                    preds = logits.argmax(dim=1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        cm = _compute_binary_confusion(all_preds, all_labels)
        tp = cm["tp"]; tn = cm["tn"]; fp = cm["fp"]; fn = cm["fn"]
        n_normal = tn + fp
        n_abnormal = fn + tp
        abn_R = tp / n_abnormal if n_abnormal > 0 else 0.0
        nor_R = tn / n_normal if n_normal > 0 else 0.0
        # F1 macro
        prec_abn = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_abn = abn_R
        f1_abn = 2 * prec_abn * rec_abn / (prec_abn + rec_abn) if (prec_abn + rec_abn) > 0 else 0.0
        prec_nor = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec_nor = nor_R
        f1_nor = 2 * prec_nor * rec_nor / (prec_nor + rec_nor) if (prec_nor + rec_nor) > 0 else 0.0
        test_f1 = (f1_abn + f1_nor) / 2

        del model
        torch.cuda.empty_cache()
        return {"test_f1": test_f1, "abn_R": abn_R, "nor_R": nor_R, "fn": fn, "fp": fp}
    return _eval


# =============================================================================
# Main
# =============================================================================
def run_simulation(log_dir: Path, device):
    print(f"\n{'='*70}")
    print(f" {log_dir.name}")
    print(f"{'='*70}")

    history = load_history(log_dir)
    if not history:
        print("  NO history.json"); return []

    checkpoints_dir = log_dir / "checkpoints"
    ckpt_files = sorted(checkpoints_dir.glob("ep_*.pth"))
    print(f"  history: {len(history)} epochs, checkpoints: {len(ckpt_files)}")
    if not ckpt_files:
        print("  NO checkpoints (need --save_every_epoch)"); return []

    # load config for dataset + model
    import yaml
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    all_classes = cfg["dataset"]["classes"]
    classes = ["normal", "abnormal"]
    image_dir = Path(cfg["output"]["image_dir"])
    data_dir = Path(cfg["output"]["data_dir"])

    import pandas as pd
    sc_df = pd.read_csv(data_dir / "scenarios.csv")
    test_df = sc_df[sc_df["split"] == "test"].reset_index(drop=True)

    _tmp = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=False)
    data_cfg = timm.data.resolve_model_data_config(_tmp)
    input_size = data_cfg.get("input_size", (3, 224, 224))
    mean = list(data_cfg.get("mean", (0.485, 0.456, 0.406)))
    std = list(data_cfg.get("std", (0.229, 0.224, 0.225)))
    del _tmp

    val_transform = transforms.Compose([
        transforms.Resize((input_size[1], input_size[2])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_ds = ChartImageDataset(image_dir, test_df, all_classes, val_transform, mode="binary")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # model template (meta — weights replaced per strategy)
    model_template = create_model(2, "convnextv2_tiny.fcmae_ft_in22k_in1k", device, dropout=0.0)
    model_template = model_template.to("cpu")  # template on CPU

    eval_fn = make_eval_fn(model_template, test_loader, device)

    results = []
    t0 = time.time()
    for metric in METRIC_METHODS:
        for weight in WEIGHT_METHODS:
            for tie in TIE_MODES:
                r = simulate_one_strategy(
                    history, checkpoints_dir, classes,
                    eval_fn, metric, weight, tie,
                )
                if r:
                    r["run"] = log_dir.name
                    results.append(r)
                    target = " 🎯" if (r["fn"] <= 2 and r["fp"] <= 2) else ""
                    print(f"  {r['strategy']:<32} ep={r['selected_ep']:>2} f1={r['test_f1']:.4f} fn={r['fn']:>3} fp={r['fp']:>3}{target}")
    print(f"  (elapsed {time.time()-t0:.1f}s, {len(results)} strategies)")
    return results


def print_summary(all_results):
    print(f"\n{'='*90}")
    print(" SUMMARY — 전체 조합 랭킹 (FN+FP 오름차순)")
    print(f"{'='*90}")

    sorted_r = sorted(all_results, key=lambda r: (r["fn"] + r["fp"], -r["test_f1"]))
    print(f"\n  {'rank':>4} {'run':<30} {'strategy':<32} {'ep':>3} {'f1%':>6} {'FN':>3} {'FP':>3}")
    print("  " + "-" * 88)
    for i, r in enumerate(sorted_r[:20], start=1):
        target = " 🎯" if (r["fn"] <= 2 and r["fp"] <= 2) else "  "
        print(f"  {i:>4} {r['run']:<30} {r['strategy']:<32} {r['selected_ep']:>3} {r['test_f1']*100:>6.2f} {r['fn']:>3} {r['fp']:>3}{target}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="run name(s) (default: v9_phase1d_bs*)")
    ap.add_argument("--out", default="experiment_plan_fpfn/simulation_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.runs:
        dirs = [LOGS / n for n in args.runs]
    else:
        dirs = sorted(LOGS.glob("v9_phase1d_*"))
    dirs = [d for d in dirs if d.is_dir()]
    print(f"Runs: {[d.name for d in dirs]}")

    all_results = []
    for d in dirs:
        all_results.extend(run_simulation(d, device))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] {args.out}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
