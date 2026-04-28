#!/usr/bin/env bash
# Stage 3: v11 full axis sweep at picked anchor.
# ANCHOR_ARGS and ANCHOR_TAG come from scripts/analyze_v11_anchor.py output.
#
# Axes (hyperparameter + rendering):
#   Hyperparameter (7 axes × 3 levels × 5 seeds = 105 runs):
#     gc    : 0.5, 1.0, 2.0, 5.0 (4)
#     fg    : 0, 0.5, 2.0 (3)
#     abw   : 0.8, 1.0, 2.0 (3)
#     wd    : 0, 0.01, 0.05 (3)
#     dp    : 0.0, 0.3, 0.5 (3)
#     sw    : 1raw, 3med, 5mean (3)
#     ema   : 0, 0.99, 0.999 (3)
#   Rendering (3 × 5 = 15 runs):
#     color : c01, c02, c03 (requires images_c0{1,2,3}_v11/)
#
# Total: 22 hyperparam cells + 3 color cells = 25 × 5 seeds = 125 runs
# Est: ~25 min/run × 125 ≈ 52h
set -u
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1

STDOUT_DIR=validations/sweep_logs
mkdir -p "$STDOUT_DIR"

: "${ANCHOR_TAG:?ANCHOR_TAG env var required — run analyze_v11_anchor.py first}"
: "${ANCHOR_ARGS:?ANCHOR_ARGS env var required}"

echo "[v11-axis] ANCHOR_TAG=${ANCHOR_TAG}"
echo "[v11-axis] ANCHOR_ARGS='${ANCHOR_ARGS}'"

V11_BASE=(
  --mode binary
  --config configs/datasets/dataset_v11.yaml
  --epochs 20
  --patience 5
  --smooth_window 3
  --smooth_method median
  --lr_backbone 2e-5
  --lr_head 2e-4
  --warmup_epochs 5
  --grad_clip 1.0
  --weight_decay 0.01
  --ema_decay 0.0
  --normal_ratio 700
  --batch_size 32
  --dropout 0.0
  --precision fp16
  --num_workers 0
)

run_v11a() {
  local name="$1"; shift
  local existing
  existing=$(ls -d "logs/"*"_${name}_F"* 2>/dev/null | head -1 || true)
  if [ -n "$existing" ]; then
    echo "[SKIP] $name -> $(basename "$existing")"
    return 0
  fi
  echo "[RUN ] $name"
  python train.py "${V11_BASE[@]}" --log_dir "$name" "$@" \
    > "$STDOUT_DIR/${name}.stdout.log" 2>&1
}

SEEDS=(42 1 2 3 4)
PREFIX="v11ax_${ANCHOR_TAG}"

run_axis_at_anchor() {
  local -a variants=("$@")
  for v in "${variants[@]}"; do
    tag=$(echo "$v" | awk '{print $1}')
    args=$(echo "$v" | sed 's/^[^ ]* //')
    for s in "${SEEDS[@]}"; do
      run_v11a "${PREFIX}_${tag}_n700_s${s}" \
        $ANCHOR_ARGS \
        $args \
        --seed "$s"
    done
  done
}

GC_VARIANTS=(
  "gc0p5    --grad_clip 0.5"
  "gc1p0    --grad_clip 1.0"
  "gc2p0    --grad_clip 2.0"
  "gc5p0    --grad_clip 5.0"
)
FG_VARIANTS=(
  "fg0      --focal_gamma 0"
  "fg0p5    --focal_gamma 0.5"
  "fg2      --focal_gamma 2.0"
)
ABW_VARIANTS=(
  "abw08    --abnormal_weight 0.8"
  "abw10    --abnormal_weight 1.0"
  "abw20    --abnormal_weight 2.0"
)
WD_VARIANTS=(
  "wd000    --weight_decay 0"
  "wd001    --weight_decay 0.01"
  "wd005    --weight_decay 0.05"
)
DP_VARIANTS=(
  "dp00     --dropout 0.0"
  "dp03     --dropout 0.3"
  "dp05     --dropout 0.5"
)
SW_VARIANTS=(
  "sw1raw   --smooth_window 1 --smooth_method median"
  "sw3med   --smooth_window 3 --smooth_method median"
  "sw5mean  --smooth_window 5 --smooth_method mean"
)
EMA_VARIANTS=(
  "ema0     --ema_decay 0.0"
  "ema99    --ema_decay 0.99"
  "ema999   --ema_decay 0.999"
)

run_axis_at_anchor "${GC_VARIANTS[@]}"
# fg / abw / ema removed (사용자 결정 2026-04-23):
#   - fg: spread 0.60 = noise (이미 fg=0이 best로 알려짐)
#   - abw: 이미 abw=1.0 default가 best, 추가 검증 의미 없음
#   - ema: ema=0.0이 v11 default, 추가 검증 의미 없음
# run_axis_at_anchor "${FG_VARIANTS[@]}"
# run_axis_at_anchor "${ABW_VARIANTS[@]}"
run_axis_at_anchor "${WD_VARIANTS[@]}"
run_axis_at_anchor "${DP_VARIANTS[@]}"
run_axis_at_anchor "${SW_VARIANTS[@]}"
# run_axis_at_anchor "${EMA_VARIANTS[@]}"

# --- Color axis (rendering) ---
# requires: configs/datasets/color_c0{1,2,3}_v11.yaml + images_c0{1,2,3}_v11/
for ctag in c01 c02 c03; do
  cfg="configs/datasets/color_${ctag}_v11.yaml"
  if [ ! -f "$cfg" ]; then
    echo "[SKIP-COLOR] $ctag  $cfg not found (expected v11 color variant)"
    continue
  fi
  img_dir=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['output']['image_dir'])")
  if [ ! -d "$img_dir/test" ]; then
    echo "[SKIP-COLOR] $ctag  $img_dir not fully generated"
    continue
  fi
  for s in "${SEEDS[@]}"; do
    run_v11a "${PREFIX}_color_${ctag}_n700_s${s}" \
      --config "$cfg" \
      $ANCHOR_ARGS \
      --seed "$s"
  done
done

echo "[v11-axis] done at $(date)"
