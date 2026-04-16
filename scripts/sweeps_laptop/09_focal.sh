#!/usr/bin/env bash
# Focal loss gamma: 7 levels x 5 seeds = 35 runs
# Memory note: "gamma is key" — never properly swept in binary mode.
# gamma=0 → CE; higher gamma → focus on hard examples.
source "$(dirname "$0")/_common.sh"

PAIRS=(
  "0      0.0"
  "0p5    0.5"
  "1p0    1.0"
  "1p5    1.5"
  "2p0    2.0"
  "2p5    2.5"
  "3p0    3.0"
)
SEEDS=(42 1 2 3 4)

for p in "${PAIRS[@]}"; do
  read tag val <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_fg${tag}_n700_s${s}" \
      --focal_gamma "$val" --seed "$s"
  done
done
