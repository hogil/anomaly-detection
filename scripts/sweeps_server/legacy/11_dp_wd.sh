#!/usr/bin/env bash
# Dropout × weight_decay 2D (interaction between explicit and implicit reg):
# 4 × 4 × 5 seeds = 80 runs
source "$(dirname "$0")/_common.sh"

# tag   stoch_depth
DPS=(
  "0      0.0"
  "0p1    0.1"
  "0p2    0.2"
  "0p4    0.4"
)
# tag     wd
WDS=(
  "0      0.0"
  "0p005  0.005"
  "0p02   0.02"
  "0p1    0.1"
)
SEEDS=(42 1 2 3 4)

for dp in "${DPS[@]}"; do
  read dtag dval <<<"$dp"
  for wd in "${WDS[@]}"; do
    read wtag wval <<<"$wd"
    for s in "${SEEDS[@]}"; do
      run_one "srv0416_dp${dtag}_wd${wtag}_n700_s${s}" \
        --stochastic_depth_rate "$dval" --weight_decay "$wval" --seed "$s"
    done
  done
done
