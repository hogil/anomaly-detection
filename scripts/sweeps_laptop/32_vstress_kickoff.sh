#!/usr/bin/env bash
# Waits until all 100 vd080_bc runs complete (pilot + ext), then launches
# vstress sw sweep. Independent from the chain script — safe even if chain
# is killed/restarted.
set -u
cd "$(dirname "$0")/../.."

echo "[vstress-kickoff] started at $(date)"
prev=0
while true; do
  n=$(ls -d logs/*vd080_bc_*_F* 2>/dev/null | wc -l)
  if [ "$n" != "$prev" ]; then
    echo "[vstress-kickoff] vd080_bc completed=$n/100 at $(date +%H:%M)"
    prev=$n
  fi
  if [ "$n" -ge 100 ]; then
    break
  fi
  sleep 120
done

# Also ensure nothing else is training
while ps -ef 2>/dev/null | grep -q "[t]rain.py"; do sleep 30; done

# Ensure vstress images exist
if [ ! -d images_vstress/train/normal ] || \
   [ "$(ls images_vstress/train/normal/*.png 2>/dev/null | wc -l)" -lt 3000 ]; then
  echo "[vstress-kickoff] vstress images not ready — aborting"
  exit 1
fi

echo "[vstress-kickoff] launching vstress sweep at $(date)"
exec bash scripts/sweeps_laptop/31_vstress_sw_sweep.sh
