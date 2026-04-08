"""v8_init multi-seed 검증 — 5 normal_counts × 5 seeds = 25 독립 trial

각 trial: 1 train + 1 deterministic prediction (NT=0.5).
같은 (normal_count, seed) 조합은 매번 동일 결과 보장 (seed deterministic).
서로 다른 seed는 모델 init/data shuffle/loader order 모두 다름.

목적: v8_init 결과의 분산 측정 (single seed의 운 vs 진짜 효과).
"""
import subprocess, json, os

NORMAL_COUNTS = [700, 2100, 2800, 3500]  # n=1400은 이미 완료, 제외
SEEDS = [1, 42, 2, 3, 4]

BASE = (
    "python train.py --epochs 20 --mode binary "
    "--scheduler cosine --warmup_epochs 5 --use_amp --num_workers 4 "
    "--batch_size 32 "
    "--lr_backbone 5e-5 --lr_head 5e-4 "
    "--dropout 0.0 "
    "--focal_gamma 0.0 "
    "--abnormal_weight 1.0 "
    "--min_epochs 10 "
)

total = len(NORMAL_COUNTS) * len(SEEDS)
done = 0
for nratio in NORMAL_COUNTS:
    for seed in SEEDS:
        done += 1
        name = f"v8seed_n{nratio}_s{seed}"
        log_dir = f"logs/{name}"
        if os.path.exists(f"{log_dir}/best_info.json"):
            print(f"[SKIP {done}/{total}] {name}")
            continue
        print(f"\n{'='*60}\n[START {done}/{total}] {name} (normal={nratio}, seed={seed})\n{'='*60}")
        cmd = f"{BASE} --normal_ratio {nratio} --seed {seed} --log_dir {log_dir}"
        print(f"  CMD: {cmd}")
        subprocess.run(cmd, shell=True)

# Summary
print("\n\n" + "="*80)
print("RESULTS — per (normal_count, seed)")
print("="*80)
print(f"{'name':<22} {'normal':>7} {'seed':>5} {'ep':>3} {'abn_R':>8} {'nor_R':>8} {'F1':>8}")
print("-"*80)
for nratio in NORMAL_COUNTS:
    for seed in SEEDS:
        name = f"v8seed_n{nratio}_s{seed}"
        info_path = f"logs/{name}/best_info.json"
        if not os.path.exists(info_path):
            print(f"{name:<22} {nratio:>7} {seed:>5} (missing)")
            continue
        with open(info_path) as f:
            bi = json.load(f)
        ep = bi.get("epoch", "-")
        abn_r = bi["test_metrics"]["abnormal"]["recall"]
        nor_r = bi["test_metrics"]["normal"]["recall"]
        f1 = bi.get("test_f1", 0)
        print(f"{name:<22} {nratio:>7} {seed:>5} {ep:>3} {abn_r:>8.4f} {nor_r:>8.4f} {f1:>8.4f}")

# Aggregation
import statistics
print("\n\n" + "="*80)
print("AGGREGATED — per normal_count (mean ± std over 5 seeds)")
print("="*80)
print(f"{'normal':>7} {'abn_R mean':>11} {'abn_R std':>11} {'nor_R mean':>11} {'nor_R std':>11} {'F1 mean':>10} {'F1 std':>10}")
print("-"*80)
for nratio in NORMAL_COUNTS:
    abn_rs, nor_rs, f1s = [], [], []
    for seed in SEEDS:
        info_path = f"logs/v8seed_n{nratio}_s{seed}/best_info.json"
        if not os.path.exists(info_path):
            continue
        with open(info_path) as f:
            bi = json.load(f)
        abn_rs.append(bi["test_metrics"]["abnormal"]["recall"])
        nor_rs.append(bi["test_metrics"]["normal"]["recall"])
        f1s.append(bi.get("test_f1", 0))
    if not abn_rs:
        print(f"{nratio:>7} (no results)")
        continue
    print(f"{nratio:>7} "
          f"{statistics.mean(abn_rs):>11.4f} {statistics.stdev(abn_rs) if len(abn_rs)>1 else 0:>11.4f} "
          f"{statistics.mean(nor_rs):>11.4f} {statistics.stdev(nor_rs) if len(nor_rs)>1 else 0:>11.4f} "
          f"{statistics.mean(f1s):>10.4f} {statistics.stdev(f1s) if len(f1s)>1 else 0:>10.4f}")

print("\nDONE")
