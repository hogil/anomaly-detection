"""v8 데이터 위에서 다양한 binary 학습 실험 sequence

목표: v8_base (test recall 0.999/F1 0.997 @ NT0.6) 대비 성능 향상
"""
import subprocess, json, os
from pathlib import Path

EXPERIMENTS = [
    # name,                args (train.py 추가 옵션)
    ("v8_g15",     "--focal_gamma 1.5"),
    ("v8_g25",     "--focal_gamma 2.5"),
    ("v8_g30",     "--focal_gamma 3.0"),
    ("v8_mixup",   "--use_mixup --mixup_alpha 0.2"),
    ("v8_aw12",    "--abnormal_weight 1.2"),
    ("v8_lr5e5",   "--lr_backbone 5e-5 --lr_head 5e-4"),
    ("v8_g25_mix", "--focal_gamma 2.5 --use_mixup --mixup_alpha 0.2"),
]

BASE = (
    "python train.py --epochs 50 --mode binary "
    "--scheduler cosine --warmup_epochs 5 --use_amp --num_workers 4 "
    "--lr_backbone 1e-4 --lr_head 1e-3 "
)

results = []
for name, extra in EXPERIMENTS:
    log_dir = f"logs/{name}"
    if os.path.exists(f"{log_dir}/best_info.json"):
        print(f"[SKIP] {name} already done")
        continue
    print(f"\n{'='*60}\n[START] {name}\n{'='*60}")
    cmd = f"{BASE} {extra} --log_dir {log_dir}"
    print(f"  CMD: {cmd}")
    subprocess.run(cmd, shell=True)

# Summary
print("\n\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'Name':<15} {'ep':<4} {'val_rec':<9} {'val_f1':<9} {'test_rec':<9} {'test_f1':<9}")
print("-"*60)
for name, _ in EXPERIMENTS:
    info_path = f"logs/{name}/best_info.json"
    if not os.path.exists(info_path):
        print(f"{name:<15} (no result)")
        continue
    with open(info_path) as f:
        bi = json.load(f)
    ep = bi.get("epoch", "-")
    vr = bi.get("val_recall", 0)
    vf = bi.get("val_f1", 0)
    tr = bi.get("test_recall", 0)
    tf = bi.get("test_f1", 0)
    print(f"{name:<15} {ep:<4} {vr:<9.4f} {vf:<9.4f} {tr:<9.4f} {tf:<9.4f}")

print("\nDONE")
