#!/usr/bin/env bash
# Scheduler comparison: cosine (baseline) vs step with various decay configs.
# 1 cosine baseline + 4 step variants = 5 configs x 5 seeds = 25 runs
source "$(dirname "$0")/_common.sh"

SEEDS=(42 1 2 3 4)

# baseline cosine (re-run for fair comparison with seeds)
for s in "${SEEDS[@]}"; do
  run_one "ex0416_schedcos_n700_s${s}" \
    --scheduler cosine --seed "$s"
done

# step variants (step_size × step_gamma)
STEP_CONFS=(
  "s5g05    5   0.5"
  "s5g07    5   0.7"
  "s10g05   10  0.5"
  "s10g07   10  0.7"
)
for c in "${STEP_CONFS[@]}"; do
  read tag size gamma <<<"$c"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_schedstep_${tag}_n700_s${s}" \
      --scheduler step --step_size "$size" --step_gamma "$gamma" --seed "$s"
  done
done
