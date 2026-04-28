#!/usr/bin/env bash
# Axis ablation at winning baseline (vd080).
# Goal: prove each option's effect on test FN+FP, pick best levels for golden recipe.
# Baseline: gc=1.0, sw=3med, lr_bb=2e-5, lr_h=2e-4, wd=0.01, fg=0, abw=1.0, dp=0.5
# For each axis: 3 levels × 3 seeds = 9 runs. 7 axes × 9 = 63 runs.
# Prefix: vd080_ax_<axis><level>
source "$(dirname "$0")/_common.sh"

SEEDS=(42 1 2)

# wd — weight decay
WD_VARIANTS=(
  "wd000   --weight_decay 0"
  "wd001   --weight_decay 0.01"
  "wd005   --weight_decay 0.05"
)
# grad_clip
GC_VARIANTS=(
  "gc05    --grad_clip 0.5"
  "gc10    --grad_clip 1.0"
  "gc20    --grad_clip 2.0"
)
# lr (lr_backbone × 10 for lr_head)
LR_VARIANTS=(
  "lr1e5   --lr_backbone 1e-5 --lr_head 1e-4"
  "lr2e5   --lr_backbone 2e-5 --lr_head 2e-4"
  "lr5e5   --lr_backbone 5e-5 --lr_head 5e-4"
)
# focal_gamma
FG_VARIANTS=(
  "fg0     --focal_gamma 0"
  "fg0p5   --focal_gamma 0.5"
  "fg2     --focal_gamma 2.0"
)
# abnormal_weight
ABW_VARIANTS=(
  "abw08   --abnormal_weight 0.8"
  "abw10   --abnormal_weight 1.0"
  "abw20   --abnormal_weight 2.0"
)
# EMA decay
EMA_VARIANTS=(
  "ema0    --ema_decay 0.0"
  "ema99   --ema_decay 0.99"
  "ema999  --ema_decay 0.999"
)

run_axis() {
  local axis="$1"; shift
  local -a variants=("$@")
  for v in "${variants[@]}"; do
    tag=$(echo "$v" | awk '{print $1}')
    args=$(echo "$v" | sed 's/^[^ ]* //')
    for s in "${SEEDS[@]}"; do
      run_one "vd080_ax_${tag}_n700_s${s}" \
        --config dataset.yaml \
        --grad_clip 1.0 \
        --smooth_window 3 --smooth_method median \
        $args \
        --seed "$s"
    done
  done
}

run_axis wd  "${WD_VARIANTS[@]}"
run_axis gc  "${GC_VARIANTS[@]}"
run_axis lr  "${LR_VARIANTS[@]}"
run_axis fg  "${FG_VARIANTS[@]}"
run_axis abw "${ABW_VARIANTS[@]}"
run_axis ema "${EMA_VARIANTS[@]}"
