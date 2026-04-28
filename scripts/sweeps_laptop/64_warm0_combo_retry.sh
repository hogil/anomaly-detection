#!/usr/bin/env bash
# Retry warm0-combo pilot runs that crashed earlier (GPU contention on 2026-04-23).
# Goal: find anchor with mean_err > 7 (current warm0=7.0), ideally 10+.
#
# Base: V11_BASE (batch=32, dropout=0, lr=2e-5, num_workers=0)
# Prefix: v11s_ (analyzer matches this)
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

# REF1 extension to n=5
for s in 3 4; do
  run_one "v11s_warm0_n700_s${s}" --warmup_epochs 0 --seed "$s"
done

# warm0 + lr combos (2 seeds each)
for s in 42 1; do
  run_one "v11s_warm0_lr5e5_n700_s${s}" \
    --warmup_epochs 0 --lr_backbone 5e-5 --lr_head 5e-4 --seed "$s"
done
for s in 42 1; do
  run_one "v11s_warm0_lr7e5_n700_s${s}" \
    --warmup_epochs 0 --lr_backbone 7e-5 --lr_head 7e-4 --seed "$s"
done

# warm0 + sched combos
for s in 42 1; do
  run_one "v11s_warm0_step_n700_s${s}" \
    --warmup_epochs 0 --scheduler step --step_size 5 --step_gamma 0.5 --seed "$s"
done
for s in 42 1; do
  run_one "v11s_warm0_plateau_n700_s${s}" \
    --warmup_epochs 0 --scheduler plateau --seed "$s"
done

# Stronger combos (new): stacked degradation
for s in 42 1; do
  run_one "v11s_warm0_plateau_lr5e5_n700_s${s}" \
    --warmup_epochs 0 --scheduler plateau \
    --lr_backbone 5e-5 --lr_head 5e-4 --seed "$s"
done
for s in 42 1; do
  run_one "v11s_warm0_ep10_n700_s${s}" \
    --warmup_epochs 0 --epochs 10 --seed "$s"
done

echo "[combo-retry] done at $(date)"
