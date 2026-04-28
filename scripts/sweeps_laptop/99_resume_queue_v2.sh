#!/usr/bin/env bash
# Resume v2 — after GPU starvation fix (batch_size=96, other projects killed).
# Continues axis ablation from fg2 onward (fg0, fg0p5 done — fg0p5_s2 missing).
set -u
source "$(dirname "$0")/_common.sh"
exec >> validations/resume_queue.log 2>&1
echo "[resume v4] started at $(date) — batch_size=${LAPTOP_BASE[*]} | fresh source"

# Part A: missing axis runs (only uncompleted)
# fg0p5_s2 was hung — retry explicitly
for s in 2; do
  run_one "vd080_ax_fg0p5_n700_s${s}" \
    --config dataset.yaml \
    --grad_clip 1.0 --smooth_window 3 --smooth_method median \
    --focal_gamma 0.5 --seed "$s"
done

# fg2 axis (3 seeds)
for s in 42 1 2; do
  run_one "vd080_ax_fg2_n700_s${s}" \
    --config dataset.yaml \
    --grad_clip 1.0 --smooth_window 3 --smooth_method median \
    --focal_gamma 2.0 --seed "$s"
done

# abw axis
for s in 42 1 2; do
  for aw in 0.8 1.0 2.0; do
    # Folder tag: abw08/abw10/abw20
    case "$aw" in
      0.8) tag=abw08 ;;
      1.0) tag=abw10 ;;
      2.0) tag=abw20 ;;
    esac
    run_one "vd080_ax_${tag}_n700_s${s}" \
      --config dataset.yaml \
      --grad_clip 1.0 --smooth_window 3 --smooth_method median \
      --abnormal_weight "$aw" --seed "$s"
  done
done

# ema axis
for s in 42 1 2; do
  for ema in 0.0 0.99 0.999; do
    case "$ema" in
      0.0)   tag=ema0   ;;
      0.99)  tag=ema99  ;;
      0.999) tag=ema999 ;;
    esac
    run_one "vd080_ax_${tag}_n700_s${s}" \
      --config dataset.yaml \
      --grad_clip 1.0 --smooth_window 3 --smooth_method median \
      --ema_decay "$ema" --seed "$s"
  done
done

echo "[resume v4] axis ablation done at $(date)"

# Part B: Model Soup retry (now with fixed model_soup.py)
for p in gc0p5_sw1raw gc0p5_sw3mean gc1p0_sw3med gc5p0_sw3mean gc5p0_sw5mean; do
  python scripts/model_soup.py --pattern "vd080_bc_${p}" \
    --out "logs/soup_${p}_v2" 2>&1 | tee -a validations/model_soup.log
done
echo "[resume v4] model soup done at $(date)"

# Part C: Stage 14 color ablation (waits for images — color_kickoff handles that)
echo "[resume v4] color ablation gated on images_c{01,02,03}_vd080/ — run 51_color_kickoff.sh separately"
echo "[resume v4] all done at $(date)"
