#!/usr/bin/env bash
# wd005 Anchor Full Axis Sweep (논문용)
#
# Anchor: V11_BASE + --warmup_epochs 0 + --weight_decay 0.05
#   → mean test_err ~9.6 (FN 2.0 / FP 7.6) — 충분한 열화 확보
#
# 목적:
#   1) trend 불량/양호 이미지 분류 task 성능 향상 증명
#   2) val loss/f1 oscillation robustness 증명
#
# 각 축: 5 seeds × N levels. 단일 축 override.
# run prefix: v11wd_<axis>_<level>_n700_s<seed>
#
# Total:
#   lr  6×5 = 30
#   wd  6×5 = 30 (anchor 포함)
#   aw  6×5 = 30
#   color 4×5 = 20
#   nr  5×5 = 25
#   sw  5×5 = 25
#   gc  4×5 = 20
#   dp  4×5 = 20
#   ep  4×5 = 20
#  =220 runs × ~15 min = ~55h
set -u
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
STDOUT_DIR=validations/sweep_logs
mkdir -p "$STDOUT_DIR"

# Anchor-embedded base
V11_WD005=(
  --mode binary
  --config configs/datasets/dataset_v11.yaml
  --epochs 20 --patience 5
  --smooth_window 3 --smooth_method median
  --lr_backbone 2e-5 --lr_head 2e-4
  --grad_clip 1.0
  --ema_decay 0.0 --normal_ratio 700
  --batch_size 32 --dropout 0.0
  --precision fp16 --num_workers 0
  # ANCHOR:
  --warmup_epochs 0
  --weight_decay 0.05
)

run_wd() {
  local name="$1"; shift
  local existing
  existing=$(ls -d "logs/"*"_${name}_F"* 2>/dev/null | head -1 || true)
  if [ -n "$existing" ]; then echo "[SKIP] $name"; return 0; fi
  echo "[RUN ] $name"
  python train.py "${V11_WD005[@]}" --log_dir "$name" "$@" \
    > "$STDOUT_DIR/${name}.stdout.log" 2>&1 || echo "[FAIL] $name rc=$?"
}

SEEDS=(42 1 2 3 4)

run_axis() {
  local axis="$1"; shift
  for v in "$@"; do
    tag=$(echo "$v" | awk '{print $1}')
    args=$(echo "$v" | sed 's/^[^ ]* //')
    for s in "${SEEDS[@]}"; do
      run_wd "v11wd_${axis}_${tag}_n700_s${s}" $args --seed "$s"
    done
  done
}

# -------- LR axis (6 levels) ----------
run_axis lr \
  "lr1em5   --lr_backbone 1e-5 --lr_head 1e-4" \
  "lr2em5   --lr_backbone 2e-5 --lr_head 2e-4" \
  "lr3em5   --lr_backbone 3e-5 --lr_head 3e-4" \
  "lr5em5   --lr_backbone 5e-5 --lr_head 5e-4" \
  "lr7em5   --lr_backbone 7e-5 --lr_head 7e-4" \
  "lr1em4   --lr_backbone 1e-4 --lr_head 1e-3"

# -------- WD axis (6 levels, anchor=0.05 포함) ----------
run_axis wd \
  "wd0     --weight_decay 0.0" \
  "wd001   --weight_decay 0.01" \
  "wd002   --weight_decay 0.02" \
  "wd005   --weight_decay 0.05" \
  "wd01    --weight_decay 0.1" \
  "wd02    --weight_decay 0.2"

# -------- AW (abnormal_weight) axis (6 levels) ----------
run_axis aw \
  "aw05    --abnormal_weight 0.5" \
  "aw08    --abnormal_weight 0.8" \
  "aw10    --abnormal_weight 1.0" \
  "aw15    --abnormal_weight 1.5" \
  "aw20    --abnormal_weight 2.0" \
  "aw30    --abnormal_weight 3.0"

# -------- Color axis (4 levels) ----------
# c00 = v11 default (anchor config), c01/c02/c03 = v11 color variants
run_axis color \
  "c00     " \
  "c01     --config configs/datasets/color_c01_v11.yaml" \
  "c02     --config configs/datasets/color_c02_v11.yaml" \
  "c03     --config configs/datasets/color_c03_v11.yaml"

# -------- Normal ratio axis (5 levels) ----------
run_axis nr \
  "nr300   --normal_ratio 300" \
  "nr500   --normal_ratio 500" \
  "nr700   --normal_ratio 700" \
  "nr1000  --normal_ratio 1000" \
  "nr1500  --normal_ratio 1500"

# -------- Smoothing axis (5 levels) ----------
run_axis sw \
  "sw1raw  --smooth_window 1 --smooth_method median" \
  "sw3med  --smooth_window 3 --smooth_method median" \
  "sw3mean --smooth_window 3 --smooth_method mean" \
  "sw5med  --smooth_window 5 --smooth_method median" \
  "sw5mean --smooth_window 5 --smooth_method mean"

# -------- Grad clip axis (4 levels) ----------
run_axis gc \
  "gc05    --grad_clip 0.5" \
  "gc10    --grad_clip 1.0" \
  "gc20    --grad_clip 2.0" \
  "gc50    --grad_clip 5.0"

# -------- Dropout axis (4 levels) ----------
run_axis dp \
  "dp00    --dropout 0.0" \
  "dp02    --dropout 0.2" \
  "dp03    --dropout 0.3" \
  "dp05    --dropout 0.5"

# -------- Epochs axis (4 levels) ----------
run_axis ep \
  "ep10    --epochs 10" \
  "ep15    --epochs 15" \
  "ep20    --epochs 20" \
  "ep30    --epochs 30"

echo "[wd005-sweep] done at $(date)"
