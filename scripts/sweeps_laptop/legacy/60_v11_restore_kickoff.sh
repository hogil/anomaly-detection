#!/usr/bin/env bash
# Wait for v11 image gen to finish, then run v11 champion-reproduction baseline (3 seeds).
set -u
cd "$(dirname "$0")/../.."
exec >> validations/v11_restore_kickoff.log 2>&1

echo "[v11-kickoff] started at $(date)"

# Wait for v11 image gen
while ps -ef 2>/dev/null | grep -q "[g]enerate_images.py --config configs/datasets/dataset_v11"; do
  sleep 30
done
echo "[v11-kickoff] image gen finished at $(date)"

# Verify
for split in train val test; do
  n=$(ls images_v11/$split 2>/dev/null | wc -l)
  if [ "$n" -lt 6 ]; then
    echo "[v11-kickoff] FAIL: images_v11/$split has $n classes"
    exit 1
  fi
done
echo "[v11-kickoff] v11 image dir verified"

# Run baseline
echo "[v11-kickoff] launching v11 baseline at $(date)"
bash scripts/sweeps_laptop/legacy/60_v11_restore_baseline.sh

echo "[v11-kickoff] done at $(date)"
