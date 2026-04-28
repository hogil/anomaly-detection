#!/usr/bin/env bash
# Wait for Stage 3 (62_v11_axis_full.sh) to finish, then run combo retry pilot.
# Does NOT auto-switch anchor — only reports. User decides whether to re-run Stage 3
# at a stronger anchor.
set -u
cd "$(dirname "$0")/../.."
exec >> validations/combo_retry_chain.log 2>&1

echo "[combo-chain] started at $(date)"

# Wait for Stage 3 to be idle (no 62_v11_axis_full.sh process + no train.py with v11ax prefix)
while ps -ef 2>/dev/null | grep -q "[6]2_v11_axis_full"; do
  sleep 120
done
while ps -ef 2>/dev/null | grep -q "[t]rain.py.*v11ax_"; do
  sleep 60
done
echo "[combo-chain] Stage 3 finished at $(date)"

# Run retry pilot
bash scripts/sweeps_laptop/legacy/64_warm0_combo_retry.sh

# Analyze — wider target [5, 20] since we want stronger anchor than current 7
echo "[combo-chain] analyzing at $(date)"
PYTHONIOENCODING=utf-8 python scripts/analyze_v11_anchor.py \
  --target_lo 7 --target_hi 20 --std_max 5 \
  --out validations/v11_anchor_v2.json || true

echo "[combo-chain] result:"
cat validations/v11_anchor_v2.json 2>/dev/null

echo "[combo-chain] done at $(date). User should review v11_anchor_v2.json before re-running Stage 3 at new anchor."
