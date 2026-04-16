#!/usr/bin/env bash
# head:backbone LR ratio: 5 ratios x 5 seeds = 25 runs
# backbone fixed at 5e-5
source "$(dirname "$0")/_common.sh"

# tag        head_lr
PAIRS=(
  "1to5     2.5e-4"
  "1to10    5e-4"
  "1to20    1e-3"
  "1to50    2.5e-3"
  "1to100   5e-3"
)
SEEDS=(42 1 2 3 4)

for p in "${PAIRS[@]}"; do
  read tag lrh <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "srv0416_ratio_${tag}_s${s}" \
      --lr_backbone 5e-5 --lr_head "$lrh" --seed "$s"
  done
done
