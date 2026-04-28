#!/usr/bin/env bash
# Weight decay extreme log-scale (extends 01_wd.sh):
# 6 new levels (very small + very large) x 5 seeds = 30 runs
# 01_wd.sh already covered {0, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2}.
# Here we add the extreme tails.
source "$(dirname "$0")/_common.sh"

PAIRS=(
  "0p0001  0.0001"
  "0p0005  0.0005"
  "0p5     0.5"
  "1p0     1.0"
  "2p0     2.0"
  "5p0     5.0"
)
SEEDS=(42 1 2 3 4)

for p in "${PAIRS[@]}"; do
  read tag val <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_wdx${tag}_n700_s${s}" \
      --weight_decay "$val" --seed "$s"
  done
done
