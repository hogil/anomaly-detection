#!/usr/bin/env bash
# Resume: run remaining pilot s=42 (skips 5 already-done), then seeds 1..4.
# CRITICAL: do NOT pipe this script through head/tail — SIGPIPE will kill it.
set -u
cd "$(dirname "$0")/../.."

LOGF="validations/sweep_logs/resume_$(date +%Y%m%d_%H%M).log"
mkdir -p "$(dirname "$LOGF")"

{
  echo "[resume] start $(date)"
  bash scripts/sweeps_laptop/legacy/30_basecase_pilot_s42.sh
  echo "[resume] pilot done at $(date)"
  bash scripts/sweeps_laptop/legacy/30_basecase_seeds_1234.sh
  echo "[resume] seeds 1..4 done at $(date)"
} >> "$LOGF" 2>&1
