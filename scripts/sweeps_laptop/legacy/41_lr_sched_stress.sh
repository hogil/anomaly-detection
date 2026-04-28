#!/usr/bin/env bash
# vd080 LR + Scheduler stress sweep (experimental testbed, NOT golden recipe).
# Goal: produce runs with varied FN/FP across error regimes so other axis
#       experiments can be evaluated against a realistic error distribution,
#       not saturated baseline only.
#
# Prefix: vd080_stress_  (distinct from _bc / _ax / _color)
# Seeds : 42, 1, 2, 3, 4  (n=5 each)
#
# Groups (7 levels × 5 = 35 runs):
#   LR extremes (3 levels): force under-train (FN↑) / mild over (mixed) / collapse (both↑)
#     lr5em6  : 5e-6  (backbone ~frozen, expect FN>>FP)
#     lr1em4  : 1e-4  (10× default, mild over — mixed outcomes)
#     lr2em4  : 2e-4  (20× default, expect collapse)
#
#   Scheduler (4 levels): keep lr2e5 default, vary LR schedule
#     sch_step_s5g05   : step_size=5  gamma=0.5 (sharp drops)
#     sch_step_s10g07  : step_size=10 gamma=0.7 (mild drops)
#     sch_plateau      : plateau (no scheduled drop, adaptive)
#     sch_cos_warm0    : cosine + warmup=0 (harsh start)
#
# Est: 35 runs × ~90min ≈ 53h on RTX 4060 Ti laptop.

set -u
source "$(dirname "$0")/_common.sh"

# PILOT mode: n=2 per config for fast anchor screening (was n=5 full).
# Rationale: user priority is finding a degraded ref condition quickly.
# After anchor picked, degraded axis sweep runs at full n=5.
SEEDS=(42 1)

# ------ LR extremes ----------------------------------------------------------
LR_VARIANTS=(
  "lr5em6    --lr_backbone 5e-6  --lr_head 5e-5"
  "lr1em4    --lr_backbone 1e-4  --lr_head 1e-3"
  "lr2em4    --lr_backbone 2e-4  --lr_head 2e-3"
)
for v in "${LR_VARIANTS[@]}"; do
  tag=$(echo "$v" | awk '{print $1}')
  args=$(echo "$v" | sed 's/^[^ ]* //')
  for s in "${SEEDS[@]}"; do
    run_one "vd080_stress_${tag}_n700_s${s}" \
      --config dataset.yaml \
      --grad_clip 1.0 \
      --smooth_window 3 --smooth_method median \
      $args \
      --seed "$s"
  done
done

# ------ Scheduler ------------------------------------------------------------
SCH_VARIANTS=(
  "sch_step_s5g05   --scheduler step    --step_size 5  --step_gamma 0.5"
  "sch_step_s10g07  --scheduler step    --step_size 10 --step_gamma 0.7"
  "sch_plateau      --scheduler plateau"
  "sch_cos_warm0    --scheduler cosine  --warmup_epochs 0"
)
for v in "${SCH_VARIANTS[@]}"; do
  tag=$(echo "$v" | awk '{print $1}')
  args=$(echo "$v" | sed 's/^[^ ]* //')
  for s in "${SEEDS[@]}"; do
    run_one "vd080_stress_${tag}_n700_s${s}" \
      --config dataset.yaml \
      --grad_clip 1.0 \
      --smooth_window 3 --smooth_method median \
      $args \
      --seed "$s"
  done
done

echo "[stress-sweep] done at $(date)"
