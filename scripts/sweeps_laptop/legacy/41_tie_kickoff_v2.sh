#!/usr/bin/env bash
# Tie-save kickoff (updated — removed vstress dependency).
# Gate: 100 vd080_bc + 63 axis ablation runs.
set -u
cd "$(dirname "$0")/../.."

echo "[tie-kickoff-v2] started at $(date)"
while true; do
  bc=$(ls -d logs/*vd080_bc_*_F* 2>/dev/null | grep -v vd080_bc_tie | wc -l)
  ax=$(ls -d logs/*vd080_ax_*_F* 2>/dev/null | wc -l)
  if [ "$bc" -ge 100 ] && [ "$ax" -ge 63 ]; then break; fi
  sleep 180
done
while ps -ef 2>/dev/null | grep -q "[t]rain.py"; do sleep 30; done
echo "[tie-kickoff-v2] launching tie-save sweep at $(date)"
exec bash scripts/sweeps_laptop/legacy/33_tie_save_sweep.sh
