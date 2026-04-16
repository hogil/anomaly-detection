#!/usr/bin/env bash
# Per-class × LR 2D (does pc=900 optimum shift at different LR?):
# 4 × 4 × 5 seeds = 80 runs
source "$(dirname "$0")/_common.sh"

PCS=(300 500 700 900)
# tag   backbone  head
LRS=(
  "2e5   2e-5  2e-4"
  "3e5   3e-5  3e-4"
  "5e5   5e-5  5e-4"
  "1e4   1e-4  1e-3"
)
SEEDS=(42 1 2 3 4)

for pc in "${PCS[@]}"; do
  for l in "${LRS[@]}"; do
    read ltag lrb lrh <<<"$l"
    for s in "${SEEDS[@]}"; do
      run_one "srv0416_pc${pc}_lr${ltag}_n700_s${s}" \
        --max_per_class "$pc" \
        --lr_backbone "$lrb" --lr_head "$lrh" \
        --seed "$s"
    done
  done
done
