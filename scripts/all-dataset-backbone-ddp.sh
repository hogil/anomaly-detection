#!/usr/bin/env bash
# DDP variant of all-dataset-backbone.sh.
#
# Detects GPU count from the system, exports DDP_NPROC_PER_NODE, then exec's
# the regular wrapper unchanged. adaptive_experiment_controller.py picks up
# DDP_NPROC_PER_NODE and launches each train.py via:
#
#   torchrun --standalone --nnodes=1 --nproc_per_node=N train.py ...
#
# train.py detects LOCAL_RANK/WORLD_SIZE env vars (set by torchrun) and:
#   - splits each batch across N GPUs via DistributedSampler (data parallel)
#   - wraps the model with DistributedDataParallel (gradient sync each step)
#   - gates all file writes / checkpoint saves / final summary to rank 0
#
# Effective batch per step = args.batch_size * N (per-process batch unchanged).
# Single-GPU regression: simply call all-dataset-backbone.sh without this wrapper.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"

# Detect GPU count: prefer nvidia-smi, fallback to torch.cuda.device_count().
if command -v nvidia-smi >/dev/null 2>&1; then
  NPROC="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
else
  NPROC="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
fi

# Allow explicit override (e.g. CUDA_VISIBLE_DEVICES=0,1 + DDP_NPROC_PER_NODE=2).
if [[ -n "${DDP_NPROC_PER_NODE:-}" ]]; then
  NPROC="$DDP_NPROC_PER_NODE"
fi

if [[ "$NPROC" -lt 2 ]]; then
  echo "[ddp] system GPU count = $NPROC, need >= 2" >&2
  echo "      use scripts/all-dataset-backbone.sh for single-GPU runs," >&2
  echo "      or set DDP_NPROC_PER_NODE=N to override detection." >&2
  exit 1
fi

export DDP_NPROC_PER_NODE="$NPROC"
echo "[ddp] DDP_NPROC_PER_NODE=$NPROC (each train.py launched via torchrun)"
echo "[ddp] effective batch per step = args.batch_size * $NPROC"
echo "[ddp] all other behavior identical to all-dataset-backbone.sh"

exec bash "$D/all-dataset-backbone.sh" "$@"
