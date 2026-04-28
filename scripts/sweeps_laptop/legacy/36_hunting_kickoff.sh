#!/usr/bin/env bash
# Waits until tie-save sweep complete, then runs hunting rescue sweep.
set -u
cd "$(dirname "$0")/../.."

echo "[hunt-kickoff] started at $(date)"
while true; do
  bc=$(ls -d logs/*vd080_bc_*_F* 2>/dev/null | grep -v vd080_bc_tie | wc -l)
  vs=$(ls -d logs/*vstress_sw*_F* 2>/dev/null | wc -l)
  tie=$(ls -d logs/*vd080_bc_tie_*_F* 2>/dev/null | wc -l)
  if [ "$bc" -ge 100 ] && [ "$vs" -ge 25 ] && [ "$tie" -ge 25 ]; then
    break
  fi
  sleep 180
done

while ps -ef 2>/dev/null | grep -q "[t]rain.py"; do sleep 30; done

echo "[hunt-kickoff] launching hunting rescue sweep at $(date)"
exec bash scripts/sweeps_laptop/legacy/35_hunting_rescue_sweep.sh
