#!/usr/bin/env bash
# Stress-sweep kickoff: wait for color sweep (and any other train.py) to finish,
# then launch LR+Scheduler stress sweep on vd080.
set -u
cd "$(dirname "$0")/../.."
exec >> validations/stress_kickoff.log 2>&1

echo "[stress-kickoff] started at $(date)"

# Wait for color ablation loop to fully finish (avoids race on next-seed launch)
while ps -ef 2>/dev/null | grep -q "[5]0_color_ablation"; do
  sleep 60
done
echo "[stress-kickoff] color_ablation loop finished at $(date)"

# Then wait for train.py idle (any residual / other sweeps)
while ps -ef 2>/dev/null | grep -q "[t]rain.py"; do
  sleep 30
done
echo "[stress-kickoff] train.py idle, launching stress sweep at $(date)"

bash scripts/sweeps_laptop/41_lr_sched_stress.sh

echo "[stress-kickoff] done at $(date)"
