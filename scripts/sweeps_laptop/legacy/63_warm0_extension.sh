#!/usr/bin/env bash
# Extension pilot: (1) grow REF1 (warm0) to n=5 seeds  (2) test stronger degradation
# combos (warm0 + lr/sched stack) to find mean_err ~10-12 anchor.
#
# Base: V11_BASE  (batch=32, dropout=0, lr=2e-5)
# Prefix: v11s_  (matches analyzer)
#
# Runs (~11, ~3h at ~18 min/run):
#   REF1 expansion:
#     warm0 s3, s4             (2)
#   Stronger degradation combos:
#     warm0 + lr5e5  s42, s1   (2)
#     warm0 + lr7e5  s42, s1   (2)
#     warm0 + sch_step  s42, s1 (2)
#     warm0 + sch_plateau s42, s1 (2)
#     warm0 + bs16 s42          (1, short batch = gradient noise)

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
  --precision fp16 --num_workers 6 --prefetch_factor 4
)

run_one() {
  local name="$1"; shift
  local existing
  existing=$(ls -d "logs/"*"_${name}_F"* 2>/dev/null | head -1 || true)
  if [ -n "$existing" ]; then echo "[SKIP] $name"; return 0; fi
  echo "[RUN ] $name"
  python train.py "${V11_BASE[@]}" --log_dir "$name" "$@" \
    > "$STDOUT_DIR/${name}.stdout.log" 2>&1
}

# (1) REF1 expansion — warm0 only, seeds 3 and 4
for s in 3 4; do
  run_one "v11s_warm0_n700_s${s}" --warmup_epochs 0 --seed "$s"
done

# (2) Stronger degradation combos — warm0 + extra
# Note: later --warmup_epochs wins in argparse (=0 used)

# warm0 + lr5e5
for s in 42 1; do
  run_one "v11s_warm0_lr5e5_n700_s${s}" \
    --warmup_epochs 0 --lr_backbone 5e-5 --lr_head 5e-4 --seed "$s"
done

# warm0 + lr7e5
for s in 42 1; do
  run_one "v11s_warm0_lr7e5_n700_s${s}" \
    --warmup_epochs 0 --lr_backbone 7e-5 --lr_head 7e-4 --seed "$s"
done

# warm0 + step scheduler (harsh drops)
for s in 42 1; do
  run_one "v11s_warm0_step_n700_s${s}" \
    --warmup_epochs 0 --scheduler step --step_size 5 --step_gamma 0.5 --seed "$s"
done

# warm0 + plateau
for s in 42 1; do
  run_one "v11s_warm0_plateau_n700_s${s}" \
    --warmup_epochs 0 --scheduler plateau --seed "$s"
done

# warm0 + bs16 (noisier grad, smaller batch)
run_one "v11s_warm0_bs16_n700_s42" \
  --warmup_epochs 0 --batch_size 16 --seed 42

echo "[ext-pilot] done at $(date)"
