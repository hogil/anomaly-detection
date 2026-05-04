#!/usr/bin/env bash
# Diagnose CUDA/NCCL/DDP on Ubuntu 24 before running repository training.
#
# Usage:
#   bash scripts/diagnose_ddp_ubuntu24.sh
#   GPUS=0,1 bash scripts/diagnose_ddp_ubuntu24.sh
#   PYTHON=.venv/bin/python RUN_TRAIN_SMOKE=1 bash scripts/diagnose_ddp_ubuntu24.sh
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
GPUS="${GPUS:-}"
NPROC="${NPROC:-}"
RUN_TRAIN_SMOKE="${RUN_TRAIN_SMOKE:-0}"

status=0

section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

run_cmd() {
  echo "+ $*"
  "$@"
  local code=$?
  echo "[exit=$code]"
  return "$code"
}

env_prefix() {
  local -n out=$1
  out=(
    "NCCL_DEBUG=$NCCL_DEBUG"
    "TORCH_DISTRIBUTED_DEBUG=$TORCH_DISTRIBUTED_DEBUG"
    "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
  )
  if [[ -n "$GPUS" ]]; then
    out+=("CUDA_VISIBLE_DEVICES=$GPUS")
  fi
}

section "System and Python"
run_cmd uname -a || true
run_cmd bash -lc 'test -r /etc/os-release && cat /etc/os-release || true' || true
run_cmd which "$PYTHON_BIN" || status=1
run_cmd "$PYTHON_BIN" --version || status=1
run_cmd "$PYTHON_BIN" -m pip show torch torchvision pyyaml timm numpy || true
run_cmd nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "GPUS override=${GPUS:-<none>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

section "Torch CUDA Report"
if ! run_cmd "$PYTHON_BIN" - <<'PY'
import os
import sys
import torch

print("python =", sys.executable)
print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
print("gpu count =", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
then
  status=1
fi

if [[ -z "$NPROC" ]]; then
  NPROC="$("$PYTHON_BIN" - <<'PY'
import torch
print(min(torch.cuda.device_count(), 2) if torch.cuda.is_available() else 0)
PY
)"
fi
NPROC="${NPROC//[$'\r\n\t ']}"

section "Single-process CUDA Tensor Probe"
if ! run_cmd "$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    x = torch.randn(256, 256, device="cuda")
    y = (x @ x).sum()
    torch.cuda.synchronize()
    print(f"device {i}: ok {float(y.detach().cpu()):.4f}")
PY
then
  status=1
fi

make_probe_file() {
  local path="$1"
  cat > "$path" <<'PY'
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
torch.cuda.set_device(local_rank)

print(
    {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": os.environ.get("WORLD_SIZE"),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    },
    flush=True,
)

dist.init_process_group("nccl")

x = torch.ones(1, device="cuda") * (rank + 1)
dist.all_reduce(x)
print(f"rank {rank}: all_reduce={x.item()}", flush=True)

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(local_rank)
ddp = DDP(model, device_ids=[local_rank], output_device=local_rank)
inp = torch.randn(4, 8, device="cuda")
loss = ddp(inp).sum()
loss.backward()
print(f"rank {rank}: ddp_model_ok loss={float(loss.detach().cpu()):.4f}", flush=True)

dist.destroy_process_group()
PY
}

if [[ "$NPROC" =~ ^[0-9]+$ && "$NPROC" -ge 2 ]]; then
  probe_file="$(mktemp /tmp/ddp_probe.XXXXXX.py)"
  make_probe_file "$probe_file"

  section "NCCL + DDP Probe (default transports)"
  envs=()
  env_prefix envs
  if ! env "${envs[@]}" "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$probe_file"; then
    echo "[warn] default NCCL/DDP probe failed"
    status=1

    section "NCCL + DDP Probe (P2P and IB disabled)"
    if ! env "${envs[@]}" NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$probe_file"; then
      echo "[warn] fallback NCCL/DDP probe failed"
      status=1
    else
      echo "[pass] fallback with NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 succeeded"
    fi
  else
    echo "[pass] default NCCL/DDP probe succeeded"
  fi
  rm -f "$probe_file"
else
  echo "[skip] DDP probe requires at least 2 visible CUDA devices; NPROC=$NPROC"
  status=1
fi

if [[ "$RUN_TRAIN_SMOKE" == "1" ]]; then
  section "Repository train.py DDP smoke"
  smoke_batch="$NPROC"
  if [[ ! "$smoke_batch" =~ ^[0-9]+$ || "$smoke_batch" -lt 2 ]]; then
    smoke_batch=2
  fi
  envs=()
  env_prefix envs
  if ! env "${envs[@]}" "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    train.py --config config_smoke.yaml --epochs 1 --batch_size "$smoke_batch" \
    --num_workers 0 --no_fast_exit --log_dir ddp_env_probe; then
    status=1
  fi
fi

section "Summary"
if [[ "$status" -eq 0 ]]; then
  echo "PASS: CUDA/NCCL/DDP probes succeeded in this Ubuntu24 environment."
else
  echo "FAIL: at least one probe failed. Compare this output with a working environment."
fi
exit "$status"
