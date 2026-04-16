#!/usr/bin/env bash
# Mixup augmentation: off + 4 alpha levels = 5 configs x 5 seeds = 25 runs
source "$(dirname "$0")/_common.sh"

SEEDS=(42 1 2 3 4)

# baseline (no mixup)
for s in "${SEEDS[@]}"; do
  run_one "ex0416_mixoff_n700_s${s}" --seed "$s"
done

# mixup with varying alpha
ALPHAS=(
  "0p1   0.1"
  "0p2   0.2"
  "0p4   0.4"
  "0p8   0.8"
)
for p in "${ALPHAS[@]}"; do
  read tag val <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_mix${tag}_n700_s${s}" \
      --use_mixup --mixup_alpha "$val" --seed "$s"
  done
done
