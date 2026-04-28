#!/usr/bin/env bash
# Freeze backbone for first N epochs (head-only warmup):
# 5 levels x 5 seeds = 25 runs
source "$(dirname "$0")/_common.sh"

FREEZES=(0 1 2 5 10)
SEEDS=(42 1 2 3 4)

for f in "${FREEZES[@]}"; do
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_frz${f}_n700_s${s}" \
      --freeze_backbone_epochs "$f" --seed "$s"
  done
done
