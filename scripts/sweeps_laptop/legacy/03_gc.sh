#!/usr/bin/env bash
# Gradient-clip rescue at lr=1e-4 (intentionally unstable): 7 levels x 5 seeds = 35 runs
# Rationale: lr=1e-4 shows max_val_f1_drop=0.67 → does gc actually stabilize?
source "$(dirname "$0")/_common.sh"

PAIRS=(
  "0p1   0.1"
  "0p25  0.25"
  "0p5   0.5"
  "1p0   1.0"
  "2p0   2.0"
  "5p0   5.0"
  "10p0  10.0"
)
SEEDS=(42 1 2 3 4)

for p in "${PAIRS[@]}"; do
  read tag val <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_gc${tag}_lr1em4_n700_s${s}" \
      --lr_backbone 1e-4 --lr_head 1e-3 \
      --grad_clip "$val" --seed "$s"
  done
done
