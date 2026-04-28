#!/usr/bin/env bash
# Golden recipe kickoff: waits for axis + tie + hunting done, then
# builds golden recipe from axis results and runs 10 seeds vs 10 baseline.
set -u
cd "$(dirname "$0")/../.."

echo "[golden-kickoff] started at $(date)"
while true; do
  ax=$(ls -d logs/*vd080_ax_*_F* 2>/dev/null | wc -l)
  tie=$(ls -d logs/*vd080_bc_tie_*_F* 2>/dev/null | wc -l)
  hu=$(ls -d logs/*vd080_hunt_*_F* 2>/dev/null | wc -l)
  if [ "$ax" -ge 63 ] && [ "$tie" -ge 25 ] && [ "$hu" -ge 30 ]; then break; fi
  sleep 180
done
while ps -ef 2>/dev/null | grep -q "[t]rain.py"; do sleep 30; done

echo "[golden-kickoff] building golden recipe at $(date)"
python scripts/build_golden_recipe.py

echo "[golden-kickoff] launching golden+control sweep at $(date)"
exec bash scripts/sweeps_laptop/legacy/43_golden_recipe.sh
