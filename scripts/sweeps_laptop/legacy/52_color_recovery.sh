#!/usr/bin/env bash
# c02 recovery: wait for c02 image regeneration, verify val/test exist,
# then resume color sweep (c02 fresh + c03) and relaunch stress + deg_axis chain.
set -u
cd "$(dirname "$0")/../.."
exec >> validations/color_recovery.log 2>&1

echo "[recovery] started at $(date)"

# 1. Wait for c02 imagegen (generate_images.py --config color_c02) to finish
while ps -ef 2>/dev/null | grep -q "[g]enerate_images.py"; do
  sleep 30
done
echo "[recovery] imagegen finished at $(date)"

# 2. Verify c02 has val and test dirs
for split in train val test; do
  n=$(ls images_c02_vd080/$split 2>/dev/null | wc -l)
  if [ "$n" -lt 6 ]; then
    echo "[recovery] FAIL: images_c02_vd080/$split has $n classes (need 6)"
    exit 1
  fi
done
echo "[recovery] c02 image dir verified (train/val/test x 6 classes)"

# 3. Relaunch color_ablation (run_one will skip the 5 completed c01 runs
#    and the 5 broken c02 folders won't block re-run since they lack _F<score>)
nohup bash scripts/sweeps_laptop/legacy/50_color_ablation.sh >> validations/color_ablation_resume.log 2>&1 &
COLOR_PID=$!
echo "[recovery] resumed color_ablation pid=${COLOR_PID}"

# 4. Relaunch stress kickoff (waits on 50_color_ablation)
nohup bash scripts/sweeps_laptop/legacy/41_stress_kickoff.sh > /dev/null 2>&1 &
STRESS_PID=$!
echo "[recovery] relaunched stress_kickoff pid=${STRESS_PID}"

# 5. Relaunch degraded axis kickoff (waits on 41_lr_sched_stress)
nohup bash scripts/sweeps_laptop/legacy/42_deg_axis_kickoff.sh > /dev/null 2>&1 &
DEG_PID=$!
echo "[recovery] relaunched deg_axis_kickoff pid=${DEG_PID}"

echo "[recovery] chain relaunched at $(date)"
