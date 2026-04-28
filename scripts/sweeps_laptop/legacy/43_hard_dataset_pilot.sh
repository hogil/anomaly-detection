#!/usr/bin/env bash
# Hard-dataset fallback pilot.
# Runs when vd080 stress pilot fails to find a degraded anchor in [8, 20] range.
#
# Two variants, stronger first:
#   vd050            : test/val_difficulty = 0.5 (anomalies halved in eval)
#   vd_noise_hi      : baseline noise doubled (harder signal separation)
#
# For each variant: 1 dataset-gen (CPU) → 7 stress configs × 2 seeds (GPU)
# Stop on first variant that produces a valid anchor.

set -u
cd "$(dirname "$0")/../.."
exec >> validations/hard_dataset_pilot.log 2>&1

echo "[hard-pilot] started at $(date)"

try_variant() {
  local name="$1"       # vd050 / vd_noise_hi
  local cfg="configs/datasets/dataset_${name}.yaml"
  local data_dir
  data_dir=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['output']['data_dir'])")
  local img_dir
  img_dir=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['output']['image_dir'])")

  echo "[hard-pilot] === variant=$name  cfg=$cfg ==="

  # 1. Data gen
  if [ ! -f "$data_dir/timeseries.csv" ] || [ ! -f "$data_dir/scenarios.csv" ]; then
    echo "[hard-pilot] generating data for $name at $(date)"
    python generate_data.py --config "$cfg" --workers 0 || { echo "[hard-pilot] data gen FAIL"; return 1; }
  fi

  # 2. Image gen (train + val + test)
  local missing=0
  for split in train val test; do
    n=$(ls "$img_dir/$split" 2>/dev/null | wc -l)
    [ "$n" -lt 6 ] && missing=1
  done
  if [ "$missing" = "1" ]; then
    echo "[hard-pilot] generating images for $name at $(date)"
    python generate_images.py --config "$cfg" --workers 0 || { echo "[hard-pilot] image gen FAIL"; return 1; }
  fi
  # Verify
  for split in train val test; do
    n=$(ls "$img_dir/$split" 2>/dev/null | wc -l)
    if [ "$n" -lt 6 ]; then
      echo "[hard-pilot] FAIL: $img_dir/$split has $n classes"
      return 1
    fi
  done

  # 3. Stress pilot on this variant (7 configs × 2 seeds = 14 runs)
  echo "[hard-pilot] launching stress pilot on $name at $(date)"
  source scripts/sweeps_laptop/legacy/_common.sh

  local SEEDS=(42 1)
  local LR_VARIANTS=(
    "lr5em6    --lr_backbone 5e-6  --lr_head 5e-5"
    "lr1em4    --lr_backbone 1e-4  --lr_head 1e-3"
    "lr2em4    --lr_backbone 2e-4  --lr_head 2e-3"
  )
  local SCH_VARIANTS=(
    "sch_step_s5g05   --scheduler step    --step_size 5  --step_gamma 0.5"
    "sch_step_s10g07  --scheduler step    --step_size 10 --step_gamma 0.7"
    "sch_plateau      --scheduler plateau"
    "sch_cos_warm0    --scheduler cosine  --warmup_epochs 0"
  )

  for v in "${LR_VARIANTS[@]}" "${SCH_VARIANTS[@]}"; do
    tag=$(echo "$v" | awk '{print $1}')
    args=$(echo "$v" | sed 's/^[^ ]* //')
    for s in "${SEEDS[@]}"; do
      run_one "${name}_stress_${tag}_n700_s${s}" \
        --config "$cfg" \
        --grad_clip 1.0 \
        --smooth_window 3 --smooth_method median \
        $args \
        --seed "$s"
    done
  done

  # 4. Check for anchor
  echo "[hard-pilot] analyzing $name stress results at $(date)"
  python scripts/analyze_stress_anchor.py \
    --prefix "${name}_stress" \
    --target_lo 8 --target_hi 20 --std_max 6 \
    --out "validations/stress_anchor_${name}.json" || true
  return 0
}

# Priority order (based on user direction 2026-04-22 07:00):
#   1. vd_hard         — defects weakened at generation source (principled)
#   2. vd050           — test/val_difficulty halved (legacy difficulty_scale route)
#   3. vd_noise_hi     — baseline noise doubled (last resort)

try_variant vd_hard
if [ -s validations/stress_anchor_vd_hard.json ]; then
  tag=$(python -c "import json; print(json.load(open('validations/stress_anchor_vd_hard.json')).get('anchor_tag'))")
  if [ -n "$tag" ] && [ "$tag" != "None" ]; then
    echo "[hard-pilot] ANCHOR FOUND via vd_hard: $tag — STOPPING fallback chain"
    exit 0
  fi
fi

try_variant vd050
if [ -s validations/stress_anchor_vd050.json ]; then
  tag=$(python -c "import json; print(json.load(open('validations/stress_anchor_vd050.json')).get('anchor_tag'))")
  if [ -n "$tag" ] && [ "$tag" != "None" ]; then
    echo "[hard-pilot] ANCHOR FOUND via vd050: $tag — STOPPING fallback chain"
    exit 0
  fi
fi

try_variant vd_noise_hi

echo "[hard-pilot] done at $(date)"
