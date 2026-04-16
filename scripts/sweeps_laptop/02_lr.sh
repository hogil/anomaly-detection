#!/usr/bin/env bash
# LR fine grid around optimum (5e-5): 9 levels x 5 seeds = 45 runs
# head is 10x backbone
source "$(dirname "$0")/_common.sh"

# tag     backbone  head
PAIRS=(
  "1p5e5  1.5e-5  1.5e-4"
  "2e5    2e-5    2e-4"
  "3e5    3e-5    3e-4"
  "4e5    4e-5    4e-4"
  "5e5    5e-5    5e-4"
  "6e5    6e-5    6e-4"
  "7e5    7e-5    7e-4"
  "8e5    8e-5    8e-4"
  "1e4    1e-4    1e-3"
)
SEEDS=(42 1 2 3 4)

for p in "${PAIRS[@]}"; do
  read tag lrb lrh <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_lr${tag}_n700_s${s}" \
      --lr_backbone "$lrb" --lr_head "$lrh" --seed "$s"
  done
done
