#!/usr/bin/env bash
# Generate v11 color variants (c01/c02/c03) on CPU in background.
# Runs in parallel with Stage 2 GPU training (CPU-bound matplotlib, minimal contention).
# Uses fewer workers (4) to avoid starving train.py DataLoader.
set -u
cd "$(dirname "$0")/../.."
exec >> validations/v11_color_imagegen.log 2>&1

echo "[v11-color-imagegen] started at $(date)"

for tag in c01 c02 c03; do
  cfg="configs/datasets/color_${tag}_v11.yaml"
  img_dir=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['output']['image_dir'])")

  # Skip if already fully generated (train + val + test each have 6 classes)
  complete=1
  for split in train val test; do
    n=$(ls "$img_dir/$split" 2>/dev/null | wc -l)
    [ "$n" -lt 6 ] && complete=0
  done
  if [ "$complete" = "1" ]; then
    echo "[v11-color-imagegen] SKIP $tag (already complete)"
    continue
  fi

  echo "[v11-color-imagegen] generating $tag at $(date)"
  python generate_images.py --config "$cfg" --workers 4
done

echo "[v11-color-imagegen] done at $(date)"
