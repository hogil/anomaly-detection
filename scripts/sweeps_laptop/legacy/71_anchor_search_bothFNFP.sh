#!/usr/bin/env bash
# Anchor search: find config where n=5 mean FN>=5 AND FP>=5 (balanced degradation).
# Total total error >= 10 with balanced split.
#
# Hypothesis:
#   - High FN ← under-training (few epochs, small data, low lr)
#   - High FP ← over-regularization (high wd) or small normal pool
#   - Balanced ← stack under-train + small normal + moderate reg
#
# 6 candidates × 5 seeds = 30 runs (~7.5h @ 15min/run)
set -u
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
STDOUT_DIR=validations/sweep_logs
mkdir -p "$STDOUT_DIR"

V11_BASE=(
  --mode binary
  --config configs/datasets/dataset_v11.yaml
  --epochs 20 --patience 5
  --smooth_window 3 --smooth_method median
  --lr_backbone 2e-5 --lr_head 2e-4
  --warmup_epochs 5 --grad_clip 1.0 --weight_decay 0.01
  --ema_decay 0.0 --normal_ratio 700
  --batch_size 32 --dropout 0.0
  --precision fp16 --num_workers 0
)

run_one() {
  local name="$1"; shift
  local existing
  existing=$(ls -d "logs/"*"_${name}_F"* 2>/dev/null | head -1 || true)
  if [ -n "$existing" ]; then echo "[SKIP] $name"; return 0; fi
  echo "[RUN ] $name"
  python train.py "${V11_BASE[@]}" --log_dir "$name" "$@" \
    > "$STDOUT_DIR/${name}.stdout.log" 2>&1 || echo "[FAIL] $name rc=$?"
}

SEEDS=(42 1 2 3 4)

# NOTE: min_epochs=7 (smoothed best save threshold) → epochs >= 10 required.
# ep5 / ep7 skipped (no best model saved, useless).

# Candidate A: ep10 + warm0 (under-train)
for s in "${SEEDS[@]}"; do
  run_one "v11as_A_ep10_warm0_n700_s${s}" \
    --epochs 10 --warmup_epochs 0 --seed "$s"
done

# Candidate B: wd005 + warm0 + ep10 (wd005 FP spike + under-train for FN)
for s in "${SEEDS[@]}"; do
  run_one "v11as_B_wd005_warm0_ep10_n700_s${s}" \
    --weight_decay 0.05 --warmup_epochs 0 --epochs 10 --seed "$s"
done

# Candidate C: nr200 + wd005 + warm0 (small normal + over-reg)
for s in "${SEEDS[@]}"; do
  run_one "v11as_C_nr200_wd005_warm0_n700_s${s}" \
    --normal_ratio 200 --weight_decay 0.05 --warmup_epochs 0 --seed "$s"
done

# Candidate D: nr200 + warm0 (small normal only)
for s in "${SEEDS[@]}"; do
  run_one "v11as_D_nr200_warm0_n700_s${s}" \
    --normal_ratio 200 --warmup_epochs 0 --seed "$s"
done

# Candidate E: lr1em5 + warm0 (slow lr + no warmup)
for s in "${SEEDS[@]}"; do
  run_one "v11as_E_lr1em5_warm0_n700_s${s}" \
    --lr_backbone 1e-5 --lr_head 1e-4 --warmup_epochs 0 --seed "$s"
done

# Candidate F: ep10 + nr300 + wd005 + warm0 (full stack: under-train + small data + over-reg)
for s in "${SEEDS[@]}"; do
  run_one "v11as_F_ep10_nr300_wd005_warm0_n700_s${s}" \
    --epochs 10 --normal_ratio 300 --weight_decay 0.05 --warmup_epochs 0 --seed "$s"
done

echo "[anchor-search] done at $(date)"
