#!/usr/bin/env bash
# Classifier dropout (head dropout, separate from stochastic_depth_rate):
# 6 levels x 5 seeds = 30 runs
source "$(dirname "$0")/_common.sh"

PAIRS=(
  "0      0.0"
  "0p1    0.1"
  "0p2    0.2"
  "0p3    0.3"
  "0p5    0.5"
  "0p7    0.7"
)
SEEDS=(42 1 2 3 4)

for p in "${PAIRS[@]}"; do
  read tag val <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_do${tag}_n700_s${s}" \
      --dropout "$val" --seed "$s"
  done
done
