#!/usr/bin/env bash
# Extension of the basecase pilot: seeds 1, 2, 3, 4 for all (gc, sw) combos.
# Will skip any run already present (via run_one's skip-existing check).
# Run AFTER 30_basecase_pilot_s42.sh completes.
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
SEEDS=(1 2 3 4)

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
