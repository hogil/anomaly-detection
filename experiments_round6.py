"""
6차 실험 - 2단계 전략 (binary + anomaly_type) × 학습 갯수별 비교
"""
import subprocess, json, time
from pathlib import Path

# 학습 갯수: 100, 200, 350, 500, 700(전체)
SIZES = [100, 200, 350, 500, 0]  # 0 = 전체 (~700/class)

# 기본 설정 (best known)
BASE = "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp"

EXPERIMENTS = []

for size in SIZES:
    label = f"{size}" if size > 0 else "full"
    mpc = f"--max_per_class {size}" if size > 0 else ""

    # Binary
    EXPERIMENTS.append({
        "name": f"r6_binary_{label}",
        "args": f"{BASE} --mode binary {mpc}",
    })
    # Anomaly type
    EXPERIMENTS.append({
        "name": f"r6_at_{label}",
        "args": f"{BASE} --mode anomaly_type {mpc}",
    })

# Multiclass 비교 (full만)
EXPERIMENTS.append({
    "name": "r6_multiclass_full",
    "args": f"{BASE} --mode multiclass",
})

def run():
    results = []
    for i, exp in enumerate(EXPERIMENTS):
        name = exp["name"]
        log_dir = Path("logs") / name

        if (log_dir / "best_info.json").exists():
            print(f"[SKIP] {name} (already done)")
            info = json.load(open(log_dir / "best_info.json"))
            results.append({"name": name, **{k: info[k] for k in ["test_recall", "test_f1", "epoch"] if k in info}})
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(EXPERIMENTS)}] {name}")
        print(f"{'='*60}")

        cmd = f'python train.py {exp["args"]} --log_dir logs/{name}'
        t0 = time.time()
        ret = subprocess.run(cmd, shell=True, cwd="D:/project/anomaly-detection")
        elapsed = time.time() - t0

        if (log_dir / "best_info.json").exists():
            info = json.load(open(log_dir / "best_info.json"))
            results.append({
                "name": name,
                "test_recall": info.get("test_recall"),
                "test_f1": info.get("test_f1"),
                "epoch": info.get("epoch"),
                "elapsed": round(elapsed, 1),
            })
            print(f"  → test_R={info.get('test_recall', 0):.4f}, F1={info.get('test_f1', 0):.4f}")
        else:
            print(f"  → FAILED (no best_info.json)")

    # 결과 저장
    with open("logs/round6_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"  6차 실험 결과")
    print(f"{'='*60}")
    print(f"  {'Name':>25s} | {'Test_R':>6s} | {'Test_F1':>7s} | {'Ep':>3s}")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*7}-+-{'-'*3}")
    for r in sorted(results, key=lambda x: x.get("test_recall", 0), reverse=True):
        print(f"  {r['name']:>25s} | {r.get('test_recall',0):6.4f} | {r.get('test_f1',0):7.4f} | {r.get('epoch','?'):>3}")


if __name__ == "__main__":
    run()
