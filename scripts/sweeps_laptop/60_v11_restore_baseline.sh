#!/usr/bin/env bash
# v11 RESTORE — reproduce the 2026-04-13 champion setup exactly on freshly-generated v11 data.
# Champion: logs/260413_081307_fresh0413_reset_v11_n700_s3 F=0.9993
#
# v11 training config (NOT the current laptop baseline):
#   batch_size: 32   (current laptop default = 192)
#   dropout: 0.0     (train.py default, pass explicitly for safety)
#   lr_backbone: 2e-5
#   lr_head:     2e-4
#   warmup: 5, grad_clip: 1.0, weight_decay: 0.01
#   scheduler: cosine
#   epochs: 20, patience: 5, smooth_window: 3, smooth_method: median
#
# Data config: configs/datasets/dataset_v11.yaml (val_difficulty_scale=1.0, data_v11/, images_v11/)
#
# This script DOES NOT reuse _common.sh (which hardcodes batch=192). It calls train.py directly
# with v11 era flags.

set -u
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1

STDOUT_DIR=validations/sweep_logs
mkdir -p "$STDOUT_DIR"

V11_BASE=(
  --mode binary
  --config configs/datasets/dataset_v11.yaml
  --epochs 20
  --patience 5
  --smooth_window 3
  --smooth_method median
  --lr_backbone 2e-5
  --lr_head 2e-4
  --warmup_epochs 5
  --grad_clip 1.0
  --weight_decay 0.01
  --ema_decay 0.0
  --normal_ratio 700
  --batch_size 32
  --dropout 0.0
  --precision fp16
  --num_workers 6
  --prefetch_factor 4
)

run_v11() {
  local name="$1"; shift
  local existing
  existing=$(ls -d "logs/"*"_${name}_F"* 2>/dev/null | head -1 || true)
  if [ -n "$existing" ]; then
    echo "[SKIP] $name -> $(basename "$existing")"
    return 0
  fi
  echo "[RUN ] $name"
  python train.py "${V11_BASE[@]}" --log_dir "$name" "$@" \
    > "$STDOUT_DIR/${name}.stdout.log" 2>&1
}

# Champion reproduction: 3 seeds (s42, s1, s3 — s3 was the F=0.9993 winner)
for s in 42 1 3; do
  run_v11 "v11r_baseline_n700_s${s}" --seed "$s"
done

echo "[v11-restore] done at $(date)"
