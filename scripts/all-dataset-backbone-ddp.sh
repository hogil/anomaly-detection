#!/usr/bin/env bash
# Multi-GPU variant of all-dataset-backbone.sh.
# train.py 가 visible GPU 수가 2 이상이면 자동으로 nn.DataParallel 사용.
# torchrun / NCCL / DistributedSampler 없이 단일 process 안에서 batch 가
# 모든 GPU 에 자동 scatter, gradient 는 GPU 0 에 reduce 후 optimizer step.
# 의미적으로 single-GPU 와 동일한 update.
#
# 사용:
#   bash scripts/all-dataset-backbone-ddp.sh                   # visible GPU 전부
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/all-dataset-backbone-ddp.sh   # GPU 4개
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"

if command -v nvidia-smi >/dev/null 2>&1; then
  NPROC="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
else
  NPROC=0
fi

if [[ "${NPROC:-0}" -lt 2 ]]; then
  echo "[multi-gpu] visible GPU = $NPROC → train.py 가 single-GPU 모드로 동작" >&2
else
  echo "[multi-gpu] visible GPU = $NPROC → train.py 가 nn.DataParallel 로 자동 분배"
  echo "[multi-gpu] effective batch = args.batch_size (DP 가 내부에서 $NPROC 등분)"
  echo "[multi-gpu] GPU 부분집합: CUDA_VISIBLE_DEVICES=0,1,2,3 bash $0"
fi

exec bash "$D/all-dataset-backbone.sh" "$@"
