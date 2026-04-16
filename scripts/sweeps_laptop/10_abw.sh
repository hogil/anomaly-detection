#!/usr/bin/env bash
# Abnormal class weight (alpha): 7 levels x 5 seeds = 35 runs
# In binary mode abnormal_weight becomes focal alpha = [1.0, W].
# W>1 → biases toward higher abnormal recall (FN↓ FP↑)
# W<1 → biases toward normal precision (FP↓ FN↑)
source "$(dirname "$0")/_common.sh"

PAIRS=(
  "0p5    0.5"
  "0p8    0.8"
  "1p0    1.0"
  "1p2    1.2"
  "1p5    1.5"
  "2p0    2.0"
  "3p0    3.0"
)
SEEDS=(42 1 2 3 4)

for p in "${PAIRS[@]}"; do
  read tag val <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_abw${tag}_n700_s${s}" \
      --abnormal_weight "$val" --seed "$s"
  done
done
