#!/usr/bin/env bash
# Real DDP launcher for all-dataset-backbone.sh.
#
# CUDA_VISIBLE_DEVICES is required and must be set by the caller, e.g.
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/all-dataset-backbone-ddp.sh
# The wrapper counts the comma-separated entries to set AD_TRAIN_DDP_NPROC.
# The adaptive controller then launches each train.py run through
# torch.distributed.run, while train.py keeps --batch_size as the global
# batch and uses global_batch/world_size as the rank-local micro-batch.
set -euo pipefail

# Force Python utf-8 output so Korean prints render on Windows cp949 consoles.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

D="$(cd "$(dirname "$0")" && pwd)"

# GPU count comes from CUDA_VISIBLE_DEVICES (comma-separated GPU IDs).
# Bash :? expansion errors out if it is unset or empty, so the caller MUST
# type e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3 bash $0`.
NPROC="${AD_TRAIN_DDP_NPROC:-$(echo "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must be set explicitly, e.g. CUDA_VISIBLE_DEVICES=0,1,2,3 bash $0}" | awk -F, '{print NF}')}"
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
