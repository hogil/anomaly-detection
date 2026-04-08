"""
3차 실험 - 수정된 데이터 + AMP + Mixup + 다양한 조합
"""

import subprocess
import json
import time
from pathlib import Path

EXPERIMENTS = [
    # Best from round2 재현 (새 데이터)
    {
        "name": "r3_steplr_15_05_hiLR",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp",
    },
    # StepLR + Mixup
    {
        "name": "r3_steplr_mixup",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --use_mixup",
    },
    # StepLR 10ep/50% + AMP
    {
        "name": "r3_steplr_10_05",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 5e-5 --lr_head 5e-4 --warmup_epochs 5 --scheduler step --step_size 10 --step_gamma 0.5 --use_amp",
    },
    # Cosine + AMP (baseline 재현)
    {
        "name": "r3_cosine_amp",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 5e-5 --lr_head 5e-4 --warmup_epochs 5 --scheduler cosine --use_amp",
    },
    # Cosine + Mixup
    {
        "name": "r3_cosine_mixup",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 5e-5 --lr_head 5e-4 --warmup_epochs 5 --scheduler cosine --use_amp --use_mixup",
    },
    # StepLR + Label Smoothing
    {
        "name": "r3_steplr_labelsmooth",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --label_smoothing 0.1",
    },
    # Batch 64 + gradient 효과
    {
        "name": "r3_batch64",
        "args": "--epochs 50 --batch_size 64 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp",
    },
    # Swin + best settings
    {
        "name": "r3_swin_steplr",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 5e-5 --lr_head 5e-4 --warmup_epochs 5 --model_name swin_tiny_patch4_window7_224.ms_in22k_ft_in1k --scheduler step --step_size 10 --step_gamma 0.5 --use_amp",
    },
]


def run_experiment(exp):
    name = exp["name"]
    log_dir = Path("logs") / name

    # Skip if already done
    if (log_dir / "best_info.json").exists():
        with open(log_dir / "best_info.json") as f:
            info = json.load(f)
        print(f"  [SKIP] {name}: test_R={info['test_recall']:.3f}")
        return {"name": name, "val_recall": info["val_recall"], "val_f1": info["val_f1"],
                "test_recall": info["test_recall"], "test_f1": info["test_f1"],
                "best_epoch": info["epoch"], "elapsed": 0}

    print(f"\n  [RUN] {name}")
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = f"python train.py {exp['args']} --log_dir {log_dir}"
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0

    if (log_dir / "best_info.json").exists():
        with open(log_dir / "best_info.json") as f:
            info = json.load(f)
        r = {"name": name, "val_recall": info["val_recall"], "val_f1": info["val_f1"],
             "test_recall": info["test_recall"], "test_f1": info["test_f1"],
             "best_epoch": info["epoch"], "elapsed": round(elapsed, 1)}
        print(f"  [OK] {name}: val_R={r['val_recall']:.3f} test_R={r['test_recall']:.3f} F1={r['test_f1']:.3f} ({elapsed:.0f}s)")
        return r
    else:
        err = result.stderr[-300:] if result.stderr else "unknown"
        print(f"  [FAIL] {name}: {err}")
        return {"name": name, "error": True}


def main():
    results = []
    for exp in EXPERIMENTS:
        try:
            r = run_experiment(exp)
            results.append(r)
        except Exception as e:
            print(f"  [ERR] {exp['name']}: {e}")
            results.append({"name": exp["name"], "error": True})

    # Summary
    print(f"\n{'='*80}")
    print(f"  3rd Round Results")
    print(f"{'='*80}")
    print(f"  {'Name':>30s} | {'Val_R':>6s} | {'Test_R':>6s} | {'Test_F1':>7s} | {'Ep':>3s}")
    print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*3}")
    for r in sorted(results, key=lambda x: x.get("test_recall", 0), reverse=True):
        if "error" in r:
            print(f"  {r['name']:>30s} | FAILED")
        else:
            print(f"  {r['name']:>30s} | {r['val_recall']:6.3f} | {r['test_recall']:6.3f} | {r['test_f1']:7.3f} | {r['best_epoch']:3d}")

    with open("logs/round3_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
