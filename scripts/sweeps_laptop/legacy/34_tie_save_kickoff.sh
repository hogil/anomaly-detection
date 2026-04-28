#!/usr/bin/env bash
# Waits until vstress sweep complete, then kicks off tie-save sweep.
# Independent from other watchers.
set -u
cd "$(dirname "$0")/../.."

echo "[tie-kickoff] started at $(date)"
# Stage gates: 100 vd080_bc + 25 vstress_sw must be done
while true; do
  bc=$(ls -d logs/*vd080_bc_*_F* 2>/dev/null | grep -v "vd080_bc_tie" | wc -l)
  vs=$(ls -d logs/*vstress_sw*_F* 2>/dev/null | wc -l)
  if [ "$bc" -ge 100 ] && [ "$vs" -ge 25 ]; then
    break
  fi
  sleep 180
done

while ps -ef 2>/dev/null | grep -q "[t]rain.py"; do sleep 30; done

echo "[tie-kickoff] launching tie-save sweep at $(date)"
exec bash scripts/sweeps_laptop/legacy/33_tie_save_sweep.sh
