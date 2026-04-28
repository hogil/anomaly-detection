#!/usr/bin/env bash
# Kickoff Stage 1 baseline once images_v11 is fully generated (folder-based signal).
set -u
cd "$(dirname "$0")/../.."
exec >> validations/v11_baseline_kickoff.log 2>&1

echo "[v11-baseline-kickoff] started at $(date)"

# Wait until train/val/test each have 6 class subdirs (= imagegen complete)
while true; do
  ok=1
  for split in train val test; do
    n=$(ls images_v11/$split 2>/dev/null | wc -l)
    [ "$n" -lt 6 ] && ok=0
  done
  [ "$ok" = "1" ] && break
  sleep 30
done
echo "[v11-baseline-kickoff] images_v11 complete at $(date)"

echo "[v11-baseline-kickoff] launching Stage 1 baseline"
bash scripts/sweeps_laptop/60_v11_restore_baseline.sh

echo "[v11-baseline-kickoff] done at $(date)"
