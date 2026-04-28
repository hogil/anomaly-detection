#!/usr/bin/env bash
# Hunting kickoff (updated — sequential after tie-save).
set -u
cd "$(dirname "$0")/../.."

echo "[hunt-kickoff-v2] started at $(date)"
while true; do
  tie=$(ls -d logs/*vd080_bc_tie_*_F* 2>/dev/null | wc -l)
  if [ "$tie" -ge 25 ]; then break; fi
  sleep 180
done
while ps -ef 2>/dev/null | grep -q "[t]rain.py"; do sleep 30; done
echo "[hunt-kickoff-v2] launching hunting rescue sweep at $(date)"
exec bash scripts/sweeps_laptop/legacy/35_hunting_rescue_sweep.sh
