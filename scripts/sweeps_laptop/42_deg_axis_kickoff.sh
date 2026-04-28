#!/usr/bin/env bash
# Degraded-axis kickoff: wait for stress sweep to finish, auto-analyze,
# and launch axis re-sweep at the picked anchor.
#
# Safety: if no valid anchor is found, script exits without launching and
# writes a note — user can re-run manually with explicit ANCHOR_ARGS.
set -u
cd "$(dirname "$0")/../.."
exec >> validations/deg_axis_kickoff.log 2>&1

echo "[deg-kickoff] started at $(date)"

# 1. Wait until stress sweep finishes (41_lr_sched_stress.sh no longer running)
while ps -ef 2>/dev/null | grep -q "[4]1_lr_sched_stress"; do
  sleep 120
done
echo "[deg-kickoff] stress sweep finished at $(date)"

# 2. Wait for any residual train.py
while ps -ef 2>/dev/null | grep -q "[t]rain.py"; do
  sleep 60
done

# 3. Pick anchor from stress results
echo "[deg-kickoff] analyzing stress results..."
python scripts/analyze_stress_anchor.py \
  --target_lo 8 --target_hi 20 --std_max 6 \
  --out validations/stress_anchor.json
rc=$?
if [ $rc -ne 0 ]; then
  echo "[deg-kickoff] no valid anchor (rc=$rc) — STOPPING. Inspect validations/stress_anchor.json"
  exit 1
fi

# 4. Extract anchor
ANCHOR_TAG=$(python -c "import json; print(json.load(open('validations/stress_anchor.json'))['anchor_tag'])")
ANCHOR_ARGS=$(python -c "import json; print(json.load(open('validations/stress_anchor.json'))['anchor_args'])")

if [ -z "$ANCHOR_TAG" ] || [ "$ANCHOR_TAG" = "None" ]; then
  echo "[deg-kickoff] anchor_tag empty — vd080 pilot produced no degraded ref"
  echo "[deg-kickoff] launching hard-dataset fallback at $(date)"
  nohup bash scripts/sweeps_laptop/43_hard_dataset_pilot.sh > /dev/null 2>&1 &
  echo "[deg-kickoff] 43_hard_dataset_pilot launched (pid=$!). Stopping deg kickoff."
  exit 0
fi

echo "[deg-kickoff] anchor=${ANCHOR_TAG}  args='${ANCHOR_ARGS}'"
echo "[deg-kickoff] launching 42_degraded_axis.sh at $(date)"

export ANCHOR_TAG ANCHOR_ARGS
bash scripts/sweeps_laptop/42_degraded_axis.sh

echo "[deg-kickoff] done at $(date)"
