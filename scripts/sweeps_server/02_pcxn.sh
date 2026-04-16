#!/usr/bin/env bash
# per-class x total-N interaction: 4 x 4 x 5 seeds = 80 runs
source "$(dirname "$0")/_common.sh"

PCS=(300 500 700 900)
NS=(700 1400 2800 5000)
SEEDS=(42 1 2 3 4)

for pc in "${PCS[@]}"; do
  for n in "${NS[@]}"; do
    for s in "${SEEDS[@]}"; do
      run_one "srv0416_pc${pc}_n${n}_s${s}" \
        --max_per_class "$pc" --normal_ratio "$n" --seed "$s"
    done
  done
done
