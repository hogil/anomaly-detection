#!/usr/bin/env bash
# Generate images for color ablation variants (c01, c02, c03) sequentially.
# Uses shared data_vd080 tabular data (scenarios + timeseries).
# Runs in background; log to validations/color_imagegen.log
set -u
cd "$(dirname "$0")/.."
exec >> validations/color_imagegen.log 2>&1

echo "[color-imagegen] started at $(date)"

# Link data_vd080 to default data_dir location expected by generate_images.py
# (variant yamls reuse data_vd080 — no data regen needed)
for tag in c01 c02 c03; do
  cfg="configs/datasets/color_${tag}.yaml"
  echo ""
  echo "[color-imagegen] === ${tag} === $(date)"
  echo "[color-imagegen] config: $cfg"
  # Single worker — leave full CPU headroom for train.py DataLoader (GPU starvation fix)
  python generate_images.py --config "$cfg" --workers 1
  echo "[color-imagegen] ${tag} done at $(date)"
done

echo ""
echo "[color-imagegen] all variants done at $(date)"
