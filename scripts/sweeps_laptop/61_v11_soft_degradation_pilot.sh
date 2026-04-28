#!/usr/bin/env bash
# Stage 2: v11 soft-degradation pilot.
# Find LR/scheduler anchor that produces mean_err ~10 on v11 data WITHOUT gradient explosion.
#
# Base: V11_BASE (batch=32, dropout=0, lr=2e-5, v11 data) — change only lr/sched.
# 8 configs × 3 seeds = 24 runs. ~10h total (~25 min/run at batch=32).
#
# Prefix: v11s_  (v11 stress)
# Goal:   pick anchor with mean(FN+FP) ∈ [8, 15], std < 4
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

run_v11s() {
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
  local rc=$?
  [ $rc -ne 0 ] && echo "[FAIL] $name (rc=$rc)"
}

SEEDS=(42 1 2)

# LR variants (gentle, not explosive — stay within proven stable range)
LR_VARIANTS=(
  "lr3e5       --lr_backbone 3e-5 --lr_head 3e-4"
  "lr5e5       --lr_backbone 5e-5 --lr_head 5e-4"
  "lr7e5       --lr_backbone 7e-5 --lr_head 7e-4"
  "lr1e4       --lr_backbone 1e-4 --lr_head 1e-3"
)
for v in "${LR_VARIANTS[@]}"; do
  tag=$(echo "$v" | awk '{print $1}')
  args=$(echo "$v" | sed 's/^[^ ]* //')
  for s in "${SEEDS[@]}"; do
    run_v11s "v11s_${tag}_n700_s${s}" $args --seed "$s"
  done
done

# Scheduler / warmup variants (lr=2e-5 default preserved)
SCH_VARIANTS=(
  "sch_step_s5g05   --scheduler step    --step_size 5  --step_gamma 0.5"
  "sch_step_s10g07  --scheduler step    --step_size 10 --step_gamma 0.7"
  "sch_plateau      --scheduler plateau"
  "warm0            --warmup_epochs 0"
)
for v in "${SCH_VARIANTS[@]}"; do
  tag=$(echo "$v" | awk '{print $1}')
  args=$(echo "$v" | sed 's/^[^ ]* //')
  for s in "${SEEDS[@]}"; do
    run_v11s "v11s_${tag}_n700_s${s}" $args --seed "$s"
  done
done

echo "[v11-soft-deg] done at $(date)"
