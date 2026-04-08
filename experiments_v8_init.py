"""v8 initial baseline + normal 갯수 변화 실험

핵심: 가장 단순한 설정 (dropout 0, focal_gamma 0, abnormal_weight 1.0)
     + normal 데이터 갯수 변화 (700 ~ 3500)

새 데이터: normal pool=3500 train (config samples_per_class.normal=5000).
각 실험은 normal pool에서 --normal_ratio로 subsample (random_state=42).
"""
import subprocess, json, os

# (name, normal_ratio)
EXPERIMENTS = [
    ("v8_init",    700),    # 1x (baseline, 기존 v8과 동일 규모)
    ("v8_init_n2", 1400),   # 2x
    ("v8_init_n3", 2100),   # 3x
    ("v8_init_n4", 2800),   # 4x
    ("v8_init_n5", 3500),   # 5x (full)
]

BASE = (
    "python train.py --epochs 50 --mode binary "
    "--scheduler cosine --warmup_epochs 5 --use_amp --num_workers 4 "
    "--batch_size 32 "
    "--lr_backbone 1e-4 --lr_head 1e-3 "
    "--dropout 0.0 "
    "--focal_gamma 0.0 "
    "--abnormal_weight 1.0 "
)

for name, nratio in EXPERIMENTS:
    log_dir = f"logs/{name}"
    if os.path.exists(f"{log_dir}/best_info.json"):
        print(f"[SKIP] {name}")
        continue
    print(f"\n{'='*60}\n[START] {name} (normal_ratio={nratio})\n{'='*60}")
    extra = f"--normal_ratio {nratio}"
    cmd = f"{BASE} {extra} --log_dir {log_dir}"
    print(f"  CMD: {cmd}")
    subprocess.run(cmd, shell=True)

# Summary
print("\n\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"{'Name':<14} {'normal':<7} {'ep':<4} {'val_rec':<9} {'val_f1':<9} {'test_rec':<10} {'test_f1':<9}")
print("-"*70)
for name, nratio in EXPERIMENTS:
    info_path = f"logs/{name}/best_info.json"
    if not os.path.exists(info_path):
        print(f"{name:<14} {nratio:<7} (no result)")
        continue
    with open(info_path) as f:
        bi = json.load(f)
    ep = bi.get("epoch", "-")
    vr = bi.get("val_recall", 0)
    vf = bi.get("val_f1", 0)
    tr = bi.get("test_recall", 0)
    tf = bi.get("test_f1", 0)
    print(f"{name:<14} {nratio:<7} {ep:<4} {vr:<9.4f} {vf:<9.4f} {tr:<10.4f} {tf:<9.4f}")

print("\nDONE")
