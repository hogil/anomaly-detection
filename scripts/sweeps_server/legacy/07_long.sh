#!/usr/bin/env bash
# Longer training variants: 3 settings x 5 seeds = 15 runs
source "$(dirname "$0")/_common.sh"

SEEDS=(42 1 2 3 4)

for s in "${SEEDS[@]}"; do
  run_one "srv0416_long_ep40_s${s}" \
    --epochs 40 --patience 12 --seed "$s"
done
for s in "${SEEDS[@]}"; do
  run_one "srv0416_long_ep60_s${s}" \
    --epochs 60 --patience 15 --seed "$s"
done
for s in "${SEEDS[@]}"; do
  run_one "srv0416_long_ema_s${s}" \
    --epochs 40 --patience 12 --ema_decay 0.999 --seed "$s"
done
