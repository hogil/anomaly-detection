#!/usr/bin/env bash
# Online Hard Example Mining ratio: 5 levels x 5 seeds = 25 runs
# ohem_ratio=0 → use all, 0.5 → top-50% hardest, 0.25 → top-25%
source "$(dirname "$0")/_common.sh"

PAIRS=(
  "0      0.0"
  "0p25   0.25"
  "0p5    0.5"
  "0p75   0.75"
  "1p0    1.0"
)
SEEDS=(42 1 2 3 4)

for p in "${PAIRS[@]}"; do
  read tag val <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_ohem${tag}_n700_s${s}" \
      --ohem_ratio "$val" --seed "$s"
  done
done
