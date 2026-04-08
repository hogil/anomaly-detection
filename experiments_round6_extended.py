"""
6차 확장 실험 - 다양한 조합 (R6 기본 실험 완료 후 실행)

실험 축:
1. Mixup (alpha 0.2, 0.4)
2. Label Smoothing (0.05, 0.1)
3. Focal Loss gamma (1.0, 2.0, 3.0)
4. Dropout (0.3, 0.5)
5. Scheduler (step, cosine)
6. Normal Threshold 다양한 값 (평가 시 자동 적용)
"""
import subprocess, json, time
from pathlib import Path

# 기본 설정
BASE = "--epochs 50 --batch_size 32 --warmup_epochs 5 --use_amp"
LR = "--lr_backbone 1e-4 --lr_head 1e-3"

EXPERIMENTS = [
    # === Binary 모드 ===
    # Mixup variations
    {"name": "r6b_bin_mixup02", "args": f"{BASE} {LR} --scheduler step --step_size 15 --step_gamma 0.5 --mode binary --use_mixup --mixup_alpha 0.2"},
    {"name": "r6b_bin_mixup04", "args": f"{BASE} {LR} --scheduler step --step_size 15 --step_gamma 0.5 --mode binary --use_mixup --mixup_alpha 0.4"},
    # Label Smoothing
    {"name": "r6b_bin_ls005", "args": f"{BASE} {LR} --scheduler step --step_size 15 --step_gamma 0.5 --mode binary --label_smoothing 0.05"},
    {"name": "r6b_bin_ls01", "args": f"{BASE} {LR} --scheduler step --step_size 15 --step_gamma 0.5 --mode binary --label_smoothing 0.1"},
    # Cosine scheduler
    {"name": "r6b_bin_cosine", "args": f"{BASE} {LR} --scheduler cosine --mode binary"},
    # Dropout
    {"name": "r6b_bin_drop03", "args": f"{BASE} {LR} --scheduler step --step_size 15 --step_gamma 0.5 --mode binary --dropout 0.3"},

    # === Anomaly Type 모드 ===
    # Mixup
    {"name": "r6b_at_mixup02", "args": f"{BASE} {LR} --scheduler step --step_size 15 --step_gamma 0.5 --mode anomaly_type --use_mixup --mixup_alpha 0.2"},
    {"name": "r6b_at_mixup04", "args": f"{BASE} {LR} --scheduler step --step_size 15 --step_gamma 0.5 --mode anomaly_type --use_mixup --mixup_alpha 0.4"},
    # Label Smoothing
    {"name": "r6b_at_ls005", "args": f"{BASE} {LR} --scheduler step --step_size 15 --step_gamma 0.5 --mode anomaly_type --label_smoothing 0.05"},
    {"name": "r6b_at_ls01", "args": f"{BASE} {LR} --scheduler step --step_size 15 --step_gamma 0.5 --mode anomaly_type --label_smoothing 0.1"},
    # Cosine
    {"name": "r6b_at_cosine", "args": f"{BASE} {LR} --scheduler cosine --mode anomaly_type"},
    # Dropout
    {"name": "r6b_at_drop03", "args": f"{BASE} {LR} --scheduler step --step_size 15 --step_gamma 0.5 --mode anomaly_type --dropout 0.3"},
    # Focal Loss gamma
    {"name": "r6b_at_gamma1", "args": f"{BASE} {LR} --scheduler step --step_size 15 --step_gamma 0.5 --mode anomaly_type --focal_gamma 1.0"},
    {"name": "r6b_at_gamma3", "args": f"{BASE} {LR} --scheduler step --step_size 15 --step_gamma 0.5 --mode anomaly_type --focal_gamma 3.0"},
    # Low LR
    {"name": "r6b_at_lowlr", "args": f"{BASE} --lr_backbone 5e-5 --lr_head 5e-4 --scheduler step --step_size 15 --step_gamma 0.5 --mode anomaly_type"},

    # === Multiclass 비교 ===
    {"name": "r6b_mc_mixup", "args": f"{BASE} {LR} --scheduler step --step_size 15 --step_gamma 0.5 --mode multiclass --use_mixup"},
    {"name": "r6b_mc_cosine", "args": f"{BASE} {LR} --scheduler cosine --mode multiclass"},
    {"name": "r6b_mc_ls01", "args": f"{BASE} {LR} --scheduler step --step_size 15 --step_gamma 0.5 --mode multiclass --label_smoothing 0.1"},
]


def run():
    results = []
    total = len(EXPERIMENTS)

    for i, exp in enumerate(EXPERIMENTS):
        name = exp["name"]
        log_dir = Path("logs") / name

        if (log_dir / "best_info.json").exists():
            print(f"[SKIP] {name}")
            info = json.load(open(log_dir / "best_info.json"))
            results.append({"name": name, "test_recall": info.get("test_recall"), "test_f1": info.get("test_f1"), "epoch": info.get("epoch")})
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {name}")
        print(f"  {exp['args']}")
        print(f"{'='*60}")

        cmd = f'python train.py {exp["args"]} --log_dir logs/{name}'
        t0 = time.time()
        subprocess.run(cmd, shell=True, cwd="D:/project/anomaly-detection")
        elapsed = time.time() - t0

        if (log_dir / "best_info.json").exists():
            info = json.load(open(log_dir / "best_info.json"))
            r = {"name": name, "test_recall": info.get("test_recall"), "test_f1": info.get("test_f1"),
                 "epoch": info.get("epoch"), "elapsed": round(elapsed, 1)}
            results.append(r)
            print(f"  → test_R={r['test_recall']:.4f}, F1={r['test_f1']:.4f} ({elapsed:.0f}s)")
        else:
            print(f"  → FAILED")
            results.append({"name": name, "test_recall": 0, "test_f1": 0, "epoch": 0})

    with open("logs/round6b_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  6차 확장 실험 결과")
    print(f"{'='*60}")
    print(f"  {'Name':>25s} | {'Test_R':>6s} | {'Test_F1':>7s} | {'Ep':>3s}")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*7}-+-{'-'*3}")
    for r in sorted(results, key=lambda x: x.get("test_recall", 0), reverse=True):
        print(f"  {r['name']:>25s} | {r.get('test_recall',0):6.4f} | {r.get('test_f1',0):7.4f} | {r.get('epoch','?'):>3}")


if __name__ == "__main__":
    run()
