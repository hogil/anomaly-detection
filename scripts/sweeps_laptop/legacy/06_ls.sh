#!/usr/bin/env bash
# Label smoothing: 5 levels x 5 seeds = 25 runs
# Prior ls=0.1 showed FP_max=23 (blowup). Test lower values.
source "$(dirname "$0")/_common.sh"

PAIRS=(
  "0     0.0"
  "0p02  0.02"
  "0p05  0.05"
  "0p08  0.08"
  "0p1   0.1"
)
SEEDS=(42 1 2 3 4)

for p in "${PAIRS[@]}"; do
  read tag val <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_ls${tag}_n700_s${s}" \
      --label_smoothing "$val" --seed "$s"
  done
done
