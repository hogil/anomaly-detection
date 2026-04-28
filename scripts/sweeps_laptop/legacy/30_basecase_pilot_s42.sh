#!/usr/bin/env bash
# Pilot (seed 42 only) for basecase gc × sw on vd080: 4 × 5 = 20 runs.
# Full sweep with all 5 seeds lives in 30_basecase_gc_sw_vd080.sh.
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

for g in "${GCS[@]}"; do
  read gtag gval <<<"$g"
  for w in "${SWS[@]}"; do
    read wtag win meth <<<"$w"
    run_one "vd080_bc_gc${gtag}_sw${wtag}_n700_s42" \
      --config dataset.yaml \
      --grad_clip "$gval" \
      --smooth_window "$win" --smooth_method "$meth" \
      --seed 42
  done
done
