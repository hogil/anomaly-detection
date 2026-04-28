#!/usr/bin/env bash
# H200 migration dry-run helper.
# Does NOT copy anything, does NOT launch training. Only prints what WOULD
# happen so the operator can review before running the real migration.
#
# Usage:
#   REMOTE=user@h200.internal:/home/user/anomaly-detection \
#     bash scripts/migrate_h200_dryrun.sh
#
# Output: human-readable plan + rsync --dry-run transcripts + a sanity
# summary of dataset/weight sizes.
set -u

LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${REMOTE:-user@h200:/home/user/anomaly-detection}"

echo "=============================================================="
echo "H200 migration DRY-RUN"
echo "  local  : $LOCAL"
echo "  remote : $REMOTE"
echo "  time   : $(date)"
echo "=============================================================="

echo
echo "[1/5] Local dataset / weight footprint"
for d in data_vd080 images_vd080 \
         data_mt05 images_mt05 \
         data_mt07 images_mt07 \
         data_mt09 images_mt09 \
         data_vstress images_vstress \
         weights src configs scripts; do
  if [ -d "$LOCAL/$d" ] || [ -f "$LOCAL/$d" ]; then
    du -sh "$LOCAL/$d" 2>/dev/null || true
  else
    echo "  [missing]  $d"
  fi
done

echo
echo "[2/5] rsync --dry-run: code + configs (no actual transfer)"
rsync -avzn \
  --include="src/***" --include="scripts/***" --include="configs/***" \
  --include="train.py" --include="generate_data.py" \
  --include="generate_images.py" --include="inference.py" \
  --include="run_experiments_v11.py" \
  --include="dataset.yaml" --include="config.yaml" \
  --include="dataset_multitest05.yaml" --include="dataset_multitest07.yaml" \
  --include="dataset_multitest09.yaml" --include="dataset_vstress.yaml" \
  --include="requirements.txt" --include="download.py" \
  --exclude="*" \
  "$LOCAL/" "$REMOTE/" 2>&1 | tail -20 || true

echo
echo "[3/5] rsync --dry-run: one dataset shard (vd080) as a sample"
rsync -avzn --partial \
  "$LOCAL/data_vd080/" "$REMOTE/data_vd080/" 2>&1 | tail -10 || true
rsync -avzn --partial \
  "$LOCAL/images_vd080/" "$REMOTE/images_vd080/" 2>&1 | tail -10 || true

echo
echo "[4/5] Would-be commands for sweep execution on H200"
cat <<'EOF'
  # stage 1 (ablation) — uses both H200 GPUs via run_experiments_v11.py
  python run_experiments_v11.py --server h200 --gpus 2 \
    --groups lr gc wd smooth reg --base_n 700 --num_workers 16 \
    --name-prefix h200_sweep

  # stage 2 (tie + hunt)
  python run_experiments_v11.py --server h200 --gpus 2 \
    --groups tie hunt --base_n 700 --num_workers 16 \
    --name-prefix h200_sweep

  # stage 3 (golden recipe)
  python scripts/build_golden_recipe.py
  python run_experiments_v11.py --server h200 --gpus 2 \
    --groups golden --num_workers 16 --name-prefix h200_sweep
EOF

echo
echo "[5/5] Pre-flight checks to run on the remote BEFORE real migration:"
cat <<'EOF'
  nvidia-smi | head -20
  df -h ~ | head -2
  python3 -c "import torch; print('cuda=', torch.cuda.is_available(),
    'devices=', torch.cuda.device_count())"
EOF

echo
echo "Dry-run complete. No files copied, no training launched."
