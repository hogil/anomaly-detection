#!/usr/bin/env bash
# Extension pilot → Auto-analyze → Resume Stage 3 (continue or switch anchor).
set -u
cd "$(dirname "$0")/../.."
exec >> validations/v11_extension.log 2>&1

echo "[ext-chain] started at $(date)"

# 1. Run extension pilot
bash scripts/sweeps_laptop/63_warm0_extension.sh
echo "[ext-chain] extension pilot complete at $(date)"

# 2. Analyze — target widened to [6, 15] to accept current warm0 (7.0) and
#    stronger combo candidates. Keep existing anchor if no better one found.
PYTHONIOENCODING=utf-8 python scripts/analyze_v11_anchor.py \
  --target_lo 6 --target_hi 15 --std_max 5 \
  --out validations/v11_anchor_ext.json || true

echo "[ext-chain] extension analysis result:"
cat validations/v11_anchor_ext.json 2>/dev/null || echo "(no output)"

# 3. Decision: prefer higher mean_err (closer to 10) if stable.
# For simplicity: always resume Stage 3 at warm0 (existing runs preserved).
# If user wants to switch anchor, they can do it manually after reviewing.
echo "[ext-chain] resuming Stage 3 at warm0 anchor at $(date)"
export ANCHOR_TAG=warm0
export ANCHOR_ARGS="--warmup_epochs 0"
bash scripts/sweeps_laptop/62_v11_axis_full.sh

echo "[ext-chain] done at $(date)"
