#!/usr/bin/env bash
# Kickoff: wait for v11 baseline (Stage 1) to finish, then run Stage 2 soft-degradation pilot,
# then analyze anchor and launch Stage 3 axis sweep.
set -u
cd "$(dirname "$0")/../.."
exec >> validations/v11_chain.log 2>&1

echo "[v11-chain] started at $(date)"

# 1. Wait for Stage 1 baseline to produce 3 completed run folders (tangible signal, no race).
echo "[v11-chain] waiting for 3 v11r_baseline_*_F* folders..."
while true; do
  n=$(ls -d logs/*v11r_baseline_n700_*_F* 2>/dev/null | wc -l)
  if [ "$n" -ge 3 ]; then break; fi
  sleep 60
done
# Also wait for any active train.py (in case last run just finished saving)
while ps -ef 2>/dev/null | grep -q "[t]rain.py.*v11r_baseline"; do
  sleep 30
done
echo "[v11-chain] Stage 1 baseline finished at $(date)"

# Show Stage 1 results
echo "[v11-chain] Stage 1 baseline runs:"
ls -d logs/*v11r_baseline*_F* 2>/dev/null | awk -F'/' '{print "  "$NF}'

# 2a. Launch color image generation in parallel (CPU, for Stage 3 color experiments)
echo "[v11-chain] launching v11 color image gen in background at $(date)"
nohup bash scripts/sweeps_laptop/legacy/61_v11_color_imagegen.sh > /dev/null 2>&1 &
echo "[v11-chain] color imagegen pid=$!"

# 2b. Run Stage 2 soft-degradation pilot
echo "[v11-chain] launching Stage 2 soft degradation pilot at $(date)"
bash scripts/sweeps_laptop/legacy/61_v11_soft_degradation_pilot.sh

# 3. Analyze anchor
echo "[v11-chain] analyzing Stage 2 results at $(date)"
python scripts/analyze_v11_anchor.py \
  --target_lo 8 --target_hi 15 --std_max 4 \
  --out validations/v11_anchor.json
rc=$?
if [ $rc -ne 0 ]; then
  echo "[v11-chain] NO VALID ANCHOR — Stage 3 skipped. Inspect validations/v11_anchor.json and re-run manually."
  exit 1
fi

ANCHOR_TAG=$(python -c "import json; print(json.load(open('validations/v11_anchor.json'))['anchor_tag'])")
ANCHOR_ARGS=$(python -c "import json; print(json.load(open('validations/v11_anchor.json'))['anchor_args'])")

if [ -z "$ANCHOR_TAG" ] || [ "$ANCHOR_TAG" = "None" ]; then
  echo "[v11-chain] anchor_tag empty — STOPPING"
  exit 1
fi

echo "[v11-chain] ANCHOR picked: ${ANCHOR_TAG}"
echo "[v11-chain] ANCHOR_ARGS: ${ANCHOR_ARGS}"

# 4. Stage 3: full axis sweep at anchor
echo "[v11-chain] launching Stage 3 axis sweep at $(date)"
export ANCHOR_TAG ANCHOR_ARGS
bash scripts/sweeps_laptop/legacy/62_v11_axis_full.sh

echo "[v11-chain] done at $(date)"
