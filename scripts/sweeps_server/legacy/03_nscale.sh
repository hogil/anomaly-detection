#!/usr/bin/env bash
# Extended N-saturation curve: 10 levels x 5 seeds = 50 runs
# Goal: draw log-log error vs N curve past current saturation point (N=3500)
source "$(dirname "$0")/_common.sh"

NS=(175 350 700 1050 1400 2100 2800 3500 5000 7000)
SEEDS=(42 1 2 3 4)

for n in "${NS[@]}"; do
  for s in "${SEEDS[@]}"; do
    run_one "srv0416_nscale_n${n}_s${s}" \
      --normal_ratio "$n" --seed "$s"
  done
done
