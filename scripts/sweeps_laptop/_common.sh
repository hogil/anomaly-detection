#!/usr/bin/env bash
# Shared helpers for laptop sweeps (4060 Ti 16GB, fp16, bs=32).
# Source this from each group script.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

STDOUT_DIR="$ROOT/validations/sweep_logs"
mkdir -p "$STDOUT_DIR"

# Force unbuffered Python stdout so sweep logs show progress in real time
export PYTHONUNBUFFERED=1

LAPTOP_BASE=(
  --mode binary
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
  --batch_size 192
  --precision fp16
  --num_workers 6
  --prefetch_factor 4
)

# Call: run_one <name> <extra args...>
run_one() {
  local name="$1"; shift
  local existing
  existing=$(ls -d "logs/"*"_${name}_F"* 2>/dev/null | head -1 || true)
  if [ -n "$existing" ]; then
    echo "[SKIP] $name  -> $(basename "$existing")"
    return 0
  fi
  echo "[RUN ] $name"
  python train.py "${LAPTOP_BASE[@]}" --log_dir "$name" "$@" \
    > "$STDOUT_DIR/${name}.stdout.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL] $name (rc=$rc) — see $STDOUT_DIR/${name}.stdout.log"
  fi
  return $rc
}

# Pretty-print the decimal (0.01 -> 0p01) for folder tag stability
# (kept in each script inline — bash has no clean way via associative arrays)
