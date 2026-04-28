#!/usr/bin/env bash
# Weight decay fine grid: 8 levels x 5 seeds = 40 runs
source "$(dirname "$0")/_common.sh"

PAIRS=(
  "0      0"
  "0p001  0.001"
  "0p005  0.005"
  "0p01   0.01"
  "0p02   0.02"
  "0p05   0.05"
  "0p1    0.1"
  "0p2    0.2"
)
SEEDS=(42 1 2 3 4)

for p in "${PAIRS[@]}"; do
  read tag val <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_wd${tag}_n700_s${s}" --weight_decay "$val" --seed "$s"
  done
done
