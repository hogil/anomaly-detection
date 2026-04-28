#!/usr/bin/env bash
# Tie-save vs strict-save ablation on vd080.
# Hypothesis (per memory feedback_val_loss_oscillation_root_cause.md):
#   tie-save causes bad-weight saves on oscillating val. But on clean vd080
#   (val_diff=test_diff=0.8), raw val_f1 doesn't saturate → ties are rarer
#   and smoother, so tie-save may actually help pick later/better weights.
#
# Design: gc=1.0 fixed, 5 sw levels × 5 seeds × {strict, tie}
#   Strict runs already exist in pilot+ext — only need to add tie runs.
#   sw 5 × seeds 5 = 25 runs on vd080 with --allow_tie_save.
#
# Prefix: vd080_bc_tie
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
    run_one "vd080_bc_tie_gc1p0_sw${wtag}_n700_s${s}" \
      --config dataset.yaml \
      --grad_clip 1.0 \
      --smooth_window "$win" --smooth_method "$meth" \
      --allow_tie_save \
      --seed "$s"
  done
done
