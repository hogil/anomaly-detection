#!/usr/bin/env bash
# Color ablation kickoff: wait for current queue (axis + soup) + image gen to finish, then run color sweep.
set -u
cd "$(dirname "$0")/../.."
exec >> validations/color_kickoff.log 2>&1

echo "[color-kickoff] started at $(date)"

# 1. Wait until image dirs exist for all 3 variants
for tag in c01 c02 c03; do
  while [ ! -d "images_${tag}_vd080/train" ]; do
    echo "[color-kickoff] waiting for images_${tag}_vd080/train at $(date)"
    sleep 120
  done
done
echo "[color-kickoff] all 3 image dirs ready at $(date)"

# 2. Wait for axis ablation + model soup queue to finish (no train.py running)
while ps -ef 2>/dev/null | grep -q "[t]rain.py"; do
  sleep 60
done
echo "[color-kickoff] train.py idle, launching color sweep at $(date)"

# 3. Run color sweep
bash scripts/sweeps_laptop/50_color_ablation.sh

echo "[color-kickoff] done at $(date)"
