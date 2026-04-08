"""
4차 실험 - 1000개/class, binary 수정, best 조합 재시도
"""

import subprocess
import json
import time
from pathlib import Path

EXPERIMENTS = [
    # Multiclass best 재현 (1000개 데이터)
    {
        "name": "r4_multiclass_best",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --mode multiclass",
    },
    # Binary (alpha 수정됨)
    {
        "name": "r4_binary",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --mode binary",
    },
    # Anomaly type
    {
        "name": "r4_anomaly_type",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --mode anomaly_type",
    },
    # Multiclass + LabelSmoothing 0.1
    {
        "name": "r4_multiclass_ls",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --mode multiclass --label_smoothing 0.1",
    },
    # Multiclass + Mixup
    {
        "name": "r4_multiclass_mixup",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --mode multiclass --use_mixup",
    },
    # Multiclass + StepLR 10/0.5 (더 자주 감소)
    {
        "name": "r4_multiclass_step10",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 10 --step_gamma 0.5 --use_amp --mode multiclass",
    },
    # Multiclass lower LR
    {
        "name": "r4_multiclass_lowlr",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 5e-5 --lr_head 5e-4 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --mode multiclass",
    },
]


def run_experiment(exp):
    name = exp["name"]
    log_dir = Path("logs") / name

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
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
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
        err = result.stderr[-500:] if result.stderr else "unknown"
        print(f"  [FAIL] {name}: {err}")
        return {"name": name, "error": True}


def main():
    results = []
    for exp in EXPERIMENTS:
        try:
            results.append(run_experiment(exp))
        except Exception as e:
            print(f"  [ERR] {exp['name']}: {e}")
            results.append({"name": exp["name"], "error": True})

    print(f"\n{'='*80}")
    print(f"  Round 4 Results (1000/class)")
    print(f"{'='*80}")
    print(f"  {'Name':>25s} | {'Val_R':>6s} | {'Test_R':>6s} | {'Test_F1':>7s} | {'Ep':>3s}")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*3}")
    for r in sorted(results, key=lambda x: x.get("test_recall", 0), reverse=True):
        if "error" in r:
            print(f"  {r['name']:>25s} | FAILED")
        else:
            ep = r.get("best_epoch", r.get("epoch", 0))
            print(f"  {r['name']:>25s} | {r['val_recall']:6.3f} | {r['test_recall']:6.3f} | {r['test_f1']:7.3f} | {ep:3d}")

    with open("logs/round4_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
