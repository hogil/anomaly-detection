#!/usr/bin/env bash
# Extend seeds {5, 6, 7} for top-3 candidate cells to reach n=8 for McNemar/Wilcoxon.
# Target: gc=0.5 sw=1raw, gc=0.5 sw=3mean, gc=1.0 sw=3med (current 5-seed leaders).
# Cost: 3 cells × 3 seeds = 9 runs (~2.2h)
source "$(dirname "$0")/_common.sh"

CELLS=(
  "gc0p5_sw1raw   --grad_clip 0.5 --smooth_window 1 --smooth_method median"
  "gc0p5_sw3mean  --grad_clip 0.5 --smooth_window 3 --smooth_method mean"
  "gc1p0_sw3med   --grad_clip 1.0 --smooth_window 3 --smooth_method median"
)
SEEDS=(5 6 7)

for c in "${CELLS[@]}"; do
  tag=$(echo "$c" | awk '{print $1}')
  args=$(echo "$c" | sed 's/^[^ ]* //')
  for s in "${SEEDS[@]}"; do
    run_one "vd080_bc_${tag}_n700_s${s}" \
      --config dataset.yaml $args --seed "$s"
  done
done
