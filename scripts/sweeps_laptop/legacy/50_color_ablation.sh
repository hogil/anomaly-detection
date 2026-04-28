#!/usr/bin/env bash
# Stage 14: Visual color/visibility ablation on vd080 data.
# 3 variants × 5 seeds = 15 runs. baseline (c00) reuses existing vd080_bc_gc1p0_sw3med runs.
#
# Variants:
#   c01: target=#E43320 red, fleet alpha unchanged at 0.4
#   c02: target=#4878CF blue, fleet alpha reduced to 0.15
#   c03: target=#E43320 red, fleet alpha reduced to 0.15
#
# Training uses images_c01_vd080 / c02 / c03 dirs (pre-generated via generate_images.py).
# Same training condition as bc sweep baseline cell (gc=1.0, sw=3med, wd=0.01).

set -u
source "$(dirname "$0")/_common.sh"

SEEDS=(42 1 2 3 4)
CONFIGS=(
  "c01  configs/datasets/color_c01.yaml"
  "c02  configs/datasets/color_c02.yaml"
  "c03  configs/datasets/color_c03.yaml"
)

for entry in "${CONFIGS[@]}"; do
  tag=$(echo "$entry" | awk '{print $1}')
  cfg=$(echo "$entry" | awk '{print $2}')
  # skip if color dataset yaml missing (image gen not done yet)
  if [ ! -f "$cfg" ]; then
    echo "[SKIP-CFG] $tag  $cfg not found"
    continue
  fi
  # skip if target image dir missing
  img_dir=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['output']['image_dir'])")
  if [ ! -d "$img_dir" ]; then
    echo "[SKIP-IMG] $tag  $img_dir missing (run generate_images.py --config $cfg first)"
    continue
  fi

  for s in "${SEEDS[@]}"; do
    run_one "vd080_color_${tag}_n700_s${s}" \
      --config "$cfg" \
      --grad_clip 1.0 \
      --smooth_window 3 --smooth_method median \
      --weight_decay 0.01 \
      --seed "$s"
  done
done

echo "[color-ablation] done at $(date)"
