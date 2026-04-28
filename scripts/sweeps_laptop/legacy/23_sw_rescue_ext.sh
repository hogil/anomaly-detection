#!/usr/bin/env bash
# Extended SW rescue: smoothing on n=3500 (also unstable, mean F1=0.965)
# AND on lr=1e-4 (unstable learning).
# 6 sw configs x 5 seed x 2 settings = 60 runs.
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

# (a) sw @ unstable n=3500
for c in "${CONFS[@]}"; do
  read tag win meth <<<"$c"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_sw${tag}_n3500_s${s}" \
      --normal_ratio 3500 \
      --smooth_window "$win" --smooth_method "$meth" --seed "$s"
  done
done

# (b) sw @ unstable lr=1e-4
for c in "${CONFS[@]}"; do
  read tag win meth <<<"$c"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_sw${tag}_lr1em4_n700_s${s}" \
      --lr_backbone 1e-4 --lr_head 1e-3 \
      --smooth_window "$win" --smooth_method "$meth" --seed "$s"
  done
done
