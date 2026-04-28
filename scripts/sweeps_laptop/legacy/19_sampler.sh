#!/usr/bin/env bash
# Train sampler strategies: shuffle / balanced_binary / balanced_original
# 3 strategies x 5 seeds = 15 runs
source "$(dirname "$0")/_common.sh"

SAMPLERS=(shuffle balanced_binary balanced_original)
SEEDS=(42 1 2 3 4)

for samp in "${SAMPLERS[@]}"; do
  tag="${samp//_/-}"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_samp_${tag}_n700_s${s}" \
      --train_sampler "$samp" --seed "$s"
  done
done
