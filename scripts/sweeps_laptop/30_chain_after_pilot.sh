#!/usr/bin/env bash
# Watches for sweep scripts to finish sequentially, then kicks off:
#   Stage 1: seeds 1..4 extension (80 runs on vd080)
#   Stage 2: vstress sw ablation (25 runs on stress dataset)
# Safe to re-run.
set -u
cd "$(dirname "$0")/../.."

wait_idle() {
  while true; do
    if ps -ef 2>/dev/null | grep -qE "[3]0_basecase_pilot_s42.sh|[3]0_basecase_seeds_1234.sh|[3]1_vstress_sw_sweep.sh"; then
      sleep 60; continue
    fi
    if ps -ef 2>/dev/null | grep -q "[t]rain.py"; then
      sleep 60; continue
    fi
    break
  done
}

wait_idle
echo "[chain] stage 1 — seeds 1..4 extension at $(date)"
bash scripts/sweeps_laptop/30_basecase_seeds_1234.sh

wait_idle
echo "[chain] stage 2 — vstress sw ablation at $(date)"
if [ ! -d images_vstress/train/normal ] || [ "$(ls images_vstress/train/normal/*.png 2>/dev/null | wc -l)" -lt 3000 ]; then
  echo "[chain] vstress images not ready, skipping stage 2"
  exit 0
fi
exec bash scripts/sweeps_laptop/31_vstress_sw_sweep.sh
