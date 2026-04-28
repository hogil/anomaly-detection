#!/usr/bin/env bash
# Batch size sweep: 4 levels x 5 seeds = 20 runs
# 4060 Ti 16GB limit: bs<=64 safe at fp16
source "$(dirname "$0")/_common.sh"

BSS=(8 16 32 64)
SEEDS=(42 1 2 3 4)

for bs in "${BSS[@]}"; do
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_bs${bs}_n700_s${s}" \
      --batch_size "$bs" --seed "$s"
  done
done
