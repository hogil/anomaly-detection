"""
실험 2차 - StepLR, ReduceLROnPlateau 추가
"""

import subprocess
import json
import time
from pathlib import Path

EXPERIMENTS = [
    # StepLR: 10ep마다 50% 감소
    {
        "name": "convnextv2_steplr_10_05",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 5e-5 --lr_head 5e-4 --warmup_epochs 5 --scheduler step --step_size 10 --step_gamma 0.5",
    },
    # StepLR: 7ep마다 30% 감소
    {
        "name": "convnextv2_steplr_7_07",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 5e-5 --lr_head 5e-4 --warmup_epochs 5 --scheduler step --step_size 7 --step_gamma 0.7",
    },
    # StepLR: 15ep마다 50% 감소 + 높은 LR
    {
        "name": "convnextv2_steplr_15_05_hiLR",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5",
    },
    # ReduceLROnPlateau
    {
        "name": "convnextv2_plateau",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 5e-5 --lr_head 5e-4 --warmup_epochs 5 --scheduler plateau --step_gamma 0.5",
    },
    # Swin + StepLR
    {
        "name": "swin_steplr",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 3e-5 --lr_head 3e-4 --warmup_epochs 5 --model_name swin_tiny_patch4_window7_224.ms_in22k_ft_in1k --scheduler step --step_size 10 --step_gamma 0.5",
    },
]


def run_experiment(exp):
    name = exp["name"]
    print(f"\n{'='*60}")
    print(f"  실험: {name}")
    print(f"{'='*60}\n")

    log_dir = Path("logs") / name
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = f"python train.py {exp['args']} --log_dir {log_dir}"
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0

    best_file = log_dir / "best_info.json"
    if best_file.exists():
        with open(best_file) as f:
            info = json.load(f)
        return {
            "name": name,
            "val_recall": info["val_recall"],
            "val_f1": info["val_f1"],
            "test_recall": info["test_recall"],
            "test_f1": info["test_f1"],
            "best_epoch": info["epoch"],
            "elapsed": round(elapsed, 1),
        }
    else:
        print(f"  FAILED: {result.stderr[-500:]}")
        return {"name": name, "error": True}


def main():
    # 1차 결과도 로드
    results = []
    round1_path = Path("logs/experiment_results.json")
    if round1_path.exists():
        with open(round1_path) as f:
            results = json.load(f)
        print(f"  1차 실험 {len(results)}개 로드")

    for exp in EXPERIMENTS:
        try:
            r = run_experiment(exp)
            results.append(r)
            if "error" not in r:
                print(f"\n  [OK] {r['name']}: val_R={r['val_recall']:.3f} test_R={r['test_recall']:.3f} F1={r['test_f1']:.3f} ({r['elapsed']:.0f}s)")
        except Exception as e:
            print(f"\n  [FAIL] {exp['name']}: {e}")
            results.append({"name": exp["name"], "error": str(e)})

    # 전체 결과
    print(f"\n\n{'='*80}")
    print(f"  전체 실험 결과")
    print(f"{'='*80}")
    print(f"  {'Name':>30s} | {'Val_R':>6s} | {'Val_F1':>6s} | {'Test_R':>6s} | {'Test_F1':>6s} | {'Ep':>3s}")
    print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*3}")

    for r in sorted(results, key=lambda x: x.get("test_recall", 0), reverse=True):
        if "error" in r:
            print(f"  {r['name']:>30s} | FAILED")
        else:
            print(f"  {r['name']:>30s} | {r['val_recall']:6.3f} | {r.get('val_f1',0):6.3f} | {r['test_recall']:6.3f} | {r['test_f1']:6.3f} | {r['best_epoch']:3d}")

    with open("logs/all_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
