#!/usr/bin/env bash
# 2D LR x WD grid: 5 x 5 x 5 seeds = 125 runs
source "$(dirname "$0")/_common.sh"

# tag    backbone  head
LRS=(
  "2e5   2e-5  2e-4"
  "3e5   3e-5  3e-4"
  "5e5   5e-5  5e-4"
  "7e5   7e-5  7e-4"
  "1e4   1e-4  1e-3"
)
WDS=(
  "0      0.0"
  "0p005  0.005"
  "0p01   0.01"
  "0p05   0.05"
  "0p1    0.1"
)
SEEDS=(42 1 2 3 4)

for lr in "${LRS[@]}"; do
  read lrtag lrb lrh <<<"$lr"
  for wd in "${WDS[@]}"; do
    read wdtag wdv <<<"$wd"
    for s in "${SEEDS[@]}"; do
      run_one "srv0416_lr${lrtag}_wd${wdtag}_n700_s${s}" \
        --lr_backbone "$lrb" --lr_head "$lrh" \
        --weight_decay "$wdv" --seed "$s"
    done
  done
done
