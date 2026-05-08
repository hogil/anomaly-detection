#!/usr/bin/env bash
# Real DDP launcher for all-dataset-backbone.sh.
#
# This wrapper only detects the visible GPU count and exports
# AD_TRAIN_DDP_NPROC. The adaptive controller then launches each train.py run
# through torch.distributed.run, while train.py keeps --batch_size as the
# global batch and uses global_batch/world_size as the rank-local micro-batch.
set -euo pipefail

# Force Python utf-8 output so Korean prints render on Windows cp949 consoles.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

D="$(cd "$(dirname "$0")" && pwd)"

PYTHON="${PYTHON:-python}"
ARGS=("$@")
idx=0
while [[ "$idx" -lt "${#ARGS[@]}" ]]; do
  case "${ARGS[$idx]}" in
    --python)
      if [[ "$((idx + 1))" -lt "${#ARGS[@]}" ]]; then
        PYTHON="${ARGS[$((idx + 1))]}"
      fi
      idx=$((idx + 2))
      ;;
    --)
      break
      ;;
    *)
      idx=$((idx + 1))
      ;;
  esac
done

detect_visible_gpus() {
  "$PYTHON" - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
}

NPROC="${AD_TRAIN_DDP_NPROC:-}"
if [[ -z "$NPROC" ]]; then
  NPROC="$(detect_visible_gpus | tr -d ' \r\n')"
fi
if ! [[ "$NPROC" =~ ^[0-9]+$ ]]; then
  NPROC=0
fi

if [[ "$NPROC" -ge 2 ]]; then
  export AD_TRAIN_DDP_NPROC="$NPROC"
  echo "[ddp] visible GPUs=$NPROC"
  echo "[ddp] each train.py run will use torchrun --nproc_per_node=$NPROC"
  echo "[ddp] --batch_size remains global; rank-local micro-batch = batch_size/$NPROC"
else
  unset AD_TRAIN_DDP_NPROC
  echo "[ddp] visible GPUs=$NPROC; falling back to single-process training" >&2
fi

exec bash "$D/all-dataset-backbone.sh" "$@"
