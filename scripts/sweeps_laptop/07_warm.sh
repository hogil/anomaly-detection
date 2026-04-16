#!/usr/bin/env bash
# Warmup epochs: 5 levels x 5 seeds = 25 runs
# Prior data showed Δ<0.0004 between 3/8. Close the book with 0/2/5/8/12.
source "$(dirname "$0")/_common.sh"

WARMS=(0 2 5 8 12)
SEEDS=(42 1 2 3 4)

for w in "${WARMS[@]}"; do
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_warm${w}_n700_s${s}" \
      --warmup_epochs "$w" --seed "$s"
  done
done
