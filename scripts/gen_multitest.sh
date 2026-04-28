#!/usr/bin/env bash
# Generate test sets at multiple difficulty levels (0.5/0.7/0.9) for
# multi-severity evaluation of existing best_model.pth checkpoints.
# Runs serially to avoid CPU contention with GPU training.
set -eu
cd "$(dirname "$0")/.."

for diff in 05 07 09; do
  cfg="dataset_multitest${diff}.yaml"
  dataroot=$(python -c "import yaml; print(yaml.safe_load(open('$cfg',encoding='utf-8'))['output']['data_dir'])")
  imgroot=$(python -c "import yaml; print(yaml.safe_load(open('$cfg',encoding='utf-8'))['output']['image_dir'])")
  if [ -f "$dataroot/scenarios.csv" ] && [ "$(find $imgroot/test -name '*.png' 2>/dev/null | wc -l)" -ge 1500 ]; then
    echo "[gen_multitest] $diff already done, skipping"
    continue
  fi
  echo "[gen_multitest] $(date +%H:%M) generating $cfg (test_diff=0.${diff:1:1})"
  python generate_data.py --config "$cfg" --workers 6 2>&1 | tail -3
  python generate_images.py --config "$cfg" --workers 4 2>&1 | tail -3
done
echo "[gen_multitest] all 3 multitest datasets ready at $(date +%H:%M)"
