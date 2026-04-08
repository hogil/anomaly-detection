"""
실험 자동화 - 다양한 모델/하이퍼파라미터 조합 비교

epoch 수 고정 (50), 나머지 변경:
- 모델: ConvNeXtV2-Tiny, EfficientNetV2-S, Swin-Tiny
- LR: 다양
- Backbone freeze 여부
- Label smoothing
"""

import subprocess
import json
import time
from pathlib import Path

EXPERIMENTS = [
    # 실험 1: Baseline (현재 best)
    {
        "name": "convnextv2_lr5e5",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 5e-5 --lr_head 5e-4 --warmup_epochs 5",
        "model_override": None,
    },
    # 실험 2: ConvNeXtV2 더 낮은 LR
    {
        "name": "convnextv2_lr2e5",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 2e-5 --lr_head 2e-4 --warmup_epochs 5",
        "model_override": None,
    },
    # 실험 3: ConvNeXtV2 더 높은 LR
    {
        "name": "convnextv2_lr1e4",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 8",
        "model_override": None,
    },
    # 실험 4: ConvNeXtV2 freeze backbone 10ep
    {
        "name": "convnextv2_freeze10",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 5e-5 --lr_head 5e-4 --warmup_epochs 5 --freeze_backbone_epochs 10",
        "model_override": None,
    },
    # 실험 5: EfficientNetV2-S
    {
        "name": "efficientnetv2s",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 3e-5 --lr_head 3e-4 --warmup_epochs 5 --model_name tf_efficientnetv2_s.in21k_ft_in1k",
        "model_override": "tf_efficientnetv2_s.in21k_ft_in1k",
    },
    # 실험 6: Swin-Tiny
    {
        "name": "swin_tiny",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 3e-5 --lr_head 3e-4 --warmup_epochs 5 --model_name swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "model_override": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    },
    # 실험 7: ConvNeXtV2 + Label Smoothing
    {
        "name": "convnextv2_labelsmooth",
        "args": "--epochs 50 --batch_size 32 --lr_backbone 5e-5 --lr_head 5e-4 --warmup_epochs 5 --label_smoothing 0.1",
        "model_override": None,
    },
]


def run_experiment(exp):
    name = exp["name"]
    print(f"\n{'='*60}")
    print(f"  실험: {name}")
    print(f"  args: {exp['args']}")
    print(f"{'='*60}\n")

    log_dir = Path("logs") / name
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = f"python train.py {exp['args']} --log_dir {log_dir}"
    t0 = time.time()

    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=1800
    )

    elapsed = time.time() - t0

    # 결과 파일 확인
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
            "test_metrics": info["test_metrics"],
        }
    else:
        print(f"  FAILED: {result.stderr[-500:]}")
        return {"name": name, "val_recall": 0, "test_recall": 0, "error": True}


def main():
    results = []

    for exp in EXPERIMENTS:
        # 이미 완료된 실험은 건너뛰기
        existing = Path("logs") / exp["name"] / "best_info.json"
        if existing.exists():
            with open(existing) as f:
                info = json.load(f)
            r = {"name": exp["name"], "val_recall": info["val_recall"], "val_f1": info["val_f1"],
                 "test_recall": info["test_recall"], "test_f1": info["test_f1"],
                 "best_epoch": info["epoch"], "elapsed": 0, "test_metrics": info["test_metrics"]}
            results.append(r)
            print(f"\n  [SKIP] {exp['name']}: already done (test_R={r['test_recall']:.3f})")
            continue
        try:
            r = run_experiment(exp)
            results.append(r)
            if "error" not in r:
                print(f"\n  [OK] {r['name']}: val_R={r['val_recall']:.3f} test_R={r['test_recall']:.3f} F1={r['test_f1']:.3f} ({r['elapsed']:.0f}s)")
        except Exception as e:
            print(f"\n  [FAIL] {exp['name']}: {e}")
            results.append({"name": exp["name"], "error": str(e)})

    # 결과 요약
    print(f"\n\n{'='*80}")
    print(f"  실험 결과 요약")
    print(f"{'='*80}")
    print(f"  {'Name':>25s} | {'Val_R':>6s} | {'Val_F1':>6s} | {'Test_R':>6s} | {'Test_F1':>6s} | {'Epoch':>5s} | {'Time':>6s}")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}")

    for r in sorted(results, key=lambda x: x.get("test_recall", 0), reverse=True):
        if "error" in r:
            print(f"  {r['name']:>25s} | FAILED")
        else:
            print(f"  {r['name']:>25s} | {r['val_recall']:6.3f} | {r['val_f1']:6.3f} | {r['test_recall']:6.3f} | {r['test_f1']:6.3f} | {r['best_epoch']:5d} | {r['elapsed']:5.0f}s")

    # 저장
    with open("logs/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  저장: logs/experiment_results.json")


if __name__ == "__main__":
    main()
