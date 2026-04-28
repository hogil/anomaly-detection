#!/usr/bin/env bash
# vstress (val_diff=1.0, test_diff=0.7) smoothing ablation:
# Hypothesis: on stress dataset where val saturates at 1.0 early, smoothing
# should rescue ties and outperform raw. If raw still wins here too, smoothing
# is obsolete on current training regime.
#
# sw 5 levels × 5 seeds = 25 runs. Fixed gc=1.0 (winning default).
# Prefix: vstress_sw
source "$(dirname "$0")/_common.sh"

SWS=(
  "1raw   1  median"
  "3med   3  median"
  "3mean  3  mean"
  "5med   5  median"
  "5mean  5  mean"
)
SEEDS=(42 1 2 3 4)

for w in "${SWS[@]}"; do
  read wtag win meth <<<"$w"
  for s in "${SEEDS[@]}"; do
    run_one "vstress_sw${wtag}_n700_s${s}" \
      --config dataset_vstress.yaml \
      --grad_clip 1.0 \
      --smooth_window "$win" --smooth_method "$meth" \
      --seed "$s"
  done
done
