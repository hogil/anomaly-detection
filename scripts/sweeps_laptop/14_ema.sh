#!/usr/bin/env bash
# EMA decay: 5 levels x 5 seeds = 25 runs
# Memory says EMA is #1 overfitting technique.
source "$(dirname "$0")/_common.sh"

PAIRS=(
  "0       0.0"
  "0p99    0.99"
  "0p995   0.995"
  "0p999   0.999"
  "0p9995  0.9995"
)
SEEDS=(42 1 2 3 4)

for p in "${PAIRS[@]}"; do
  read tag val <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_ema${tag}_n700_s${s}" \
      --ema_decay "$val" --seed "$s"
  done
done
