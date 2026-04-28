#!/usr/bin/env bash
# Resume full queue after model_soup crash: wd=0.05 extension + axis ablation + soup retry
set -u
source "$(dirname "$0")/_common.sh"
exec >> validations/resume_queue.log 2>&1
echo "[resume v3] started at $(date) in $(pwd)"

# Part 1: wd=0.05 extension (seeds 3, 4)
for s in 3 4; do
  run_one "vd080_ax_wd005_n700_s${s}" \
    --config dataset.yaml \
    --grad_clip 1.0 --smooth_window 3 --smooth_method median \
    --weight_decay 0.05 --seed "$s"
done
echo "[resume v3] wd=0.05 ext done at $(date)"

# Part 2: remaining axis ablation (gc, lr, fg, abw, ema)
bash scripts/sweeps_laptop/legacy/40_axis_ablation.sh
echo "[resume v3] axis ablation done at $(date)"

# Part 3: Model Soup retry
for p in gc0p5_sw1raw gc0p5_sw3mean gc1p0_sw3med gc5p0_sw3mean gc5p0_sw5mean; do
  python scripts/model_soup.py --pattern "vd080_bc_${p}" \
    --out "logs/soup_${p}_v2" 2>&1 | tee -a validations/model_soup.log
done
echo "[resume v3] model soup done at $(date)"

# Part 4: Stage 14 color ablation (15 runs) — images must be pre-generated
bash scripts/sweeps_laptop/legacy/50_color_ablation.sh
echo "[resume v3] all done at $(date)"
