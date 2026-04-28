#!/usr/bin/env bash
# Per-class count fine grid: 19 levels (50~950 step 50) x 3 seeds = 57 runs
source "$(dirname "$0")/_common.sh"

PCS=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950)
SEEDS=(42 1 2)

for pc in "${PCS[@]}"; do
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_pc${pc}_n700_s${s}" \
      --max_per_class "$pc" --seed "$s"
  done
done
