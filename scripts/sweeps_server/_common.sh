#!/usr/bin/env bash
# Shared helpers for server sweeps (H200 x2, bf16+compile+bs=256).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

STDOUT_DIR="$ROOT/validations/sweep_logs"
mkdir -p "$STDOUT_DIR"

SERVER_BASE=(
  --mode binary
  --epochs 25
  --patience 8
  --smooth_window 5
  --smooth_method median
  --lr_backbone 2e-5
  --lr_head 2e-4
  --warmup_epochs 5
  --grad_clip 1.0
  --weight_decay 0.01
  --ema_decay 0.0
  --normal_ratio 700
  --precision bf16
  --compile
  --batch_size 256
  --num_workers 16
  --prefetch_factor 8
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
  python train.py "${SERVER_BASE[@]}" --log_dir "$name" "$@" \
    > "$STDOUT_DIR/${name}.stdout.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL] $name (rc=$rc) — see $STDOUT_DIR/${name}.stdout.log"
  fi
  return $rc
}
