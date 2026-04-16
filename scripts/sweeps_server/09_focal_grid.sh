#!/usr/bin/env bash
# Focal loss gamma × abnormal_weight 2D grid: 5 × 4 × 5 seeds = 100 runs
# Memory says "gamma is key", alpha (abnormal_weight) controls FN/FP trade.
source "$(dirname "$0")/_common.sh"

# tag    gamma
GAMMAS=(
  "0      0.0"
  "1p0    1.0"
  "1p5    1.5"
  "2p0    2.0"
  "3p0    3.0"
)
# tag    alpha
ALPHAS=(
  "0p8    0.8"
  "1p0    1.0"
  "1p5    1.5"
  "2p0    2.0"
)
SEEDS=(42 1 2 3 4)

for g in "${GAMMAS[@]}"; do
  read gtag gval <<<"$g"
  for a in "${ALPHAS[@]}"; do
    read atag aval <<<"$a"
    for s in "${SEEDS[@]}"; do
      run_one "srv0416_fg${gtag}_abw${atag}_n700_s${s}" \
        --focal_gamma "$gval" --abnormal_weight "$aval" --seed "$s"
    done
  done
done
