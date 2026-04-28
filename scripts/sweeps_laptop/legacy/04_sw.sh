#!/usr/bin/env bash
# Val-smoothing on unstable baseline (N=2100, which averaged F1=0.955):
# 6 smoothing configs x 5 seeds = 30 runs
# Rationale: smoothing should rescue bad-case seeds, not just raise mean.
source "$(dirname "$0")/_common.sh"

# tag    window  method
CONFS=(
  "1raw   1  median"
  "3med   3  median"
  "3mean  3  mean"
  "5med   5  median"
  "5mean  5  mean"
  "7med   7  median"
)
SEEDS=(42 1 2 3 4)

for c in "${CONFS[@]}"; do
  read tag win meth <<<"$c"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_sw${tag}_n2100_s${s}" \
      --normal_ratio 2100 \
      --smooth_window "$win" --smooth_method "$meth" \
      --seed "$s"
  done
done
