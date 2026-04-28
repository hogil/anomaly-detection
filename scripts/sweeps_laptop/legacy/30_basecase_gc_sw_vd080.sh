#!/usr/bin/env bash
# Basecase (winning.yaml defaults) sweep on vd080 dataset (val_difficulty=test_difficulty=0.8):
#   - gc (grad_clip): 0.5 / 1.0 / 2.0 / 5.0  (4 levels)
#   - sw (smooth_window × method): 1raw / 3med / 3mean / 5med / 5mean  (5 levels)
# Cross product: 4 × 5 × 5 seeds = 100 runs. Seeds 42/1/2/3/4.
#
# Purpose: re-evaluate gc and smoothing method on stable basecase with
#   val difficulty matching test (so val selection signal is meaningful),
#   and with new train.py VAL_PEAK logging (ep 1+).
#
# Prefix: vd080_bc (basecase)
source "$(dirname "$0")/_common.sh"

GCS=(
  "0p5   0.5"
  "1p0   1.0"
  "2p0   2.0"
  "5p0   5.0"
)
SWS=(
  "1raw   1  median"
  "3med   3  median"
  "3mean  3  mean"
  "5med   5  median"
  "5mean  5  mean"
)
SEEDS=(42 1 2 3 4)

for g in "${GCS[@]}"; do
  read gtag gval <<<"$g"
  for w in "${SWS[@]}"; do
    read wtag win meth <<<"$w"
    for s in "${SEEDS[@]}"; do
      run_one "vd080_bc_gc${gtag}_sw${wtag}_n700_s${s}" \
        --config dataset.yaml \
        --grad_clip "$gval" \
        --smooth_window "$win" --smooth_method "$meth" \
        --seed "$s"
    done
  done
done
