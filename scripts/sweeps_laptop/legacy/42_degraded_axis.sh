#!/usr/bin/env bash
# Degraded-baseline axis re-sweep on vd080.
# Purpose: run all remaining axis experiments at a deliberately degraded
#          LR/scheduler anchor so axis effects rise above the saturated
#          noise floor (mean error ~10-20/run instead of ~3).
#
# The ANCHOR is injected via env var ANCHOR_ARGS, chosen from stress-sweep
# results by scripts/analyze_stress_anchor.py. Default placeholder:
#   ANCHOR_ARGS="--lr_backbone 1e-4 --lr_head 1e-3"
#
# Axes tested at anchor (each var appended/overriding ANCHOR_ARGS):
#   gc  (grad_clip): 0.5, 1.0, 2.0, 5.0            (4 levels)
#   fg  (focal_gamma): 0, 0.5, 2.0                  (3 levels)
#   abw (abnormal_weight): 0.8, 1.0, 2.0            (3 levels)
#   wd  (weight_decay): 0, 0.01, 0.05               (3 levels)
#   dp  (dropout): 0.0, 0.3, 0.5                    (3 levels)
#   sw  (smooth_window/method): 1raw, 3med, 5mean   (3 levels)
# Total: 19 cells × 5 seeds = 95 runs
# Est: ~90 min/run × 95 ≈ 142h on laptop
#
# Prefix: vd080_deg_  (degraded baseline)

set -u
source "$(dirname "$0")/_common.sh"

: "${ANCHOR_ARGS:=--lr_backbone 1e-4 --lr_head 1e-3}"
: "${ANCHOR_TAG:=lr1em4}"

echo "[deg-axis] ANCHOR_TAG=${ANCHOR_TAG}  ANCHOR_ARGS='${ANCHOR_ARGS}'"

SEEDS=(42 1 2 3 4)

run_axis_at_anchor() {
  local axis="$1"; shift
  local -a variants=("$@")
  for v in "${variants[@]}"; do
    tag=$(echo "$v" | awk '{print $1}')
    args=$(echo "$v" | sed 's/^[^ ]* //')
    for s in "${SEEDS[@]}"; do
      run_one "vd080_deg_${ANCHOR_TAG}_${tag}_n700_s${s}" \
        --config dataset.yaml \
        $ANCHOR_ARGS \
        $args \
        --seed "$s"
    done
  done
}

GC_VARIANTS=(
  "gc0p5    --grad_clip 0.5"
  "gc1p0    --grad_clip 1.0"
  "gc2p0    --grad_clip 2.0"
  "gc5p0    --grad_clip 5.0"
)
FG_VARIANTS=(
  "fg0      --focal_gamma 0"
  "fg0p5    --focal_gamma 0.5"
  "fg2      --focal_gamma 2.0"
)
ABW_VARIANTS=(
  "abw08    --abnormal_weight 0.8"
  "abw10    --abnormal_weight 1.0"
  "abw20    --abnormal_weight 2.0"
)
WD_VARIANTS=(
  "wd000    --weight_decay 0"
  "wd001    --weight_decay 0.01"
  "wd005    --weight_decay 0.05"
)
DP_VARIANTS=(
  "dp00     --dropout 0.0"
  "dp03     --dropout 0.3"
  "dp05     --dropout 0.5"
)
SW_VARIANTS=(
  "sw1raw   --smooth_window 1 --smooth_method median"
  "sw3med   --smooth_window 3 --smooth_method median"
  "sw5mean  --smooth_window 5 --smooth_method mean"
)

run_axis_at_anchor gc  "${GC_VARIANTS[@]}"
run_axis_at_anchor fg  "${FG_VARIANTS[@]}"
run_axis_at_anchor abw "${ABW_VARIANTS[@]}"
run_axis_at_anchor wd  "${WD_VARIANTS[@]}"
run_axis_at_anchor dp  "${DP_VARIANTS[@]}"
run_axis_at_anchor sw  "${SW_VARIANTS[@]}"

echo "[deg-axis] done at $(date)"
