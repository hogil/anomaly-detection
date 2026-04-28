#!/usr/bin/env bash
# Kicks off axis ablation sweep after vd080 pilot+ext (100 bc runs).
set -u
cd "$(dirname "$0")/../.."

echo "[axis-kickoff] started at $(date)"
while true; do
  bc=$(ls -d logs/*vd080_bc_*_F* 2>/dev/null | grep -v vd080_bc_tie | wc -l)
  if [ "$bc" -ge 100 ]; then break; fi
  sleep 180
done
while ps -ef 2>/dev/null | grep -q "[t]rain.py"; do sleep 30; done
echo "[axis-kickoff] launching axis ablation at $(date)"
exec bash scripts/sweeps_laptop/legacy/40_axis_ablation.sh
