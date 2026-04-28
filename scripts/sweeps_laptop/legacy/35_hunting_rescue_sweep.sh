#!/usr/bin/env bash
# Hunting stress + rescue combinations on vd080.
# Goal: measure whether (smoothing × tie-save) rescues oscillating val_f1 runs.
#
# Induce hunting via lr=1e-4 (5× winning default).
#   Prior stability analysis: lr=1e-4 on fresh0413_reset_v11 had
#   max val_f1 drop 0.67, loss_jump 0.904, drops>1% freq 0.67.
#
# Grid: lr=1e-4 × 5 sw × 2 {strict, tie} × 3 seeds = 30 runs (~7h).
# Metric focus: worst-seed FN/FP (rescue = reduce worst case).
#
# Prefix: vd080_hunt_lr1e4_<sw>_<tie>
source "$(dirname "$0")/_common.sh"

SWS=(
  "1raw   1  median"
  "3med   3  median"
  "3mean  3  mean"
  "5med   5  median"
  "5mean  5  mean"
)
TIES=(
  "strict  "
  "tie     --allow_tie_save"
)
SEEDS=(42 1 2)

for w in "${SWS[@]}"; do
  read wtag win meth <<<"$w"
  for t in "${TIES[@]}"; do
    ttag=$(echo "$t" | awk '{print $1}')
    targ=$(echo "$t" | awk '{for(i=2;i<=NF;i++) printf "%s ",$i}')
    for s in "${SEEDS[@]}"; do
      run_one "vd080_hunt_lr1e4_sw${wtag}_${ttag}_s${s}" \
        --config dataset.yaml \
        --lr_backbone 1e-4 --lr_head 1e-3 \
        --grad_clip 1.0 \
        --smooth_window "$win" --smooth_method "$meth" \
        $targ \
        --seed "$s"
    done
  done
done
