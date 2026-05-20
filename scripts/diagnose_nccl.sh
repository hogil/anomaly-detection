#!/usr/bin/env bash
# H100 / multi-GPU DDP NCCL diagnostics.
#
# Captures everything needed to figure out why DDP works with 2 GPUs but
# fails with 4+ GPUs (or any "unhandled cuda error" from NCCL):
#   - NCCL init smoke test with NCCL_DEBUG=INFO
#   - fabricmanager / shm / ulimit / driver / NCCL / torch version
#   - nvidia-smi topology matrix
#   - GPU process snapshot
#
# Usage:
#   bash scripts/diagnose_nccl.sh                     # uses CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash scripts/diagnose_nccl.sh 0,1,2,3,4,5,6,7      # custom GPU set
#   bash scripts/diagnose_nccl.sh 0,1,2,3 nccl_p2p_off # also try with P2P disabled
#
# Output: validations/nccl_diag_<timestamp>/diag.log (single file, paste to chat)

set -uo pipefail

# Resolve repo root from this script's location so the diagnostic works no
# matter which directory the user calls it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SMOKE_PY="$SCRIPT_DIR/_nccl_smoke.py"
cd "$REPO_ROOT"

if [[ ! -f "$SMOKE_PY" ]]; then
  echo "[fatal] $SMOKE_PY not found." >&2
  echo "        Run 'git pull --rebase' inside $REPO_ROOT and try again." >&2
  exit 1
fi

GPUS_CSV="${1:-0,1,2,3}"
EXTRA_MODE="${2:-}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="validations/nccl_diag_${TS}"
LOG="${OUT_DIR}/diag.log"
mkdir -p "$OUT_DIR"

section() {
  {
    echo
    echo "================================================================"
    echo "## $*"
    echo "================================================================"
  } | tee -a "$LOG"
}

run() {
  echo "+ $*" | tee -a "$LOG"
  # shellcheck disable=SC2068
  $@ 2>&1 | tee -a "$LOG"
  echo | tee -a "$LOG"
}

{
  echo "# NCCL diagnostic ${TS}"
  echo "- gpus=${GPUS_CSV}"
  echo "- extra_mode=${EXTRA_MODE}"
  echo "- host=$(hostname)"
  echo "- user=$(whoami)"
} | tee "$LOG"

section "environment versions"
run uname -a
run nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null
run python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('nccl', torch.cuda.nccl.version() if torch.cuda.is_available() else 'no-cuda'); print('cuda_avail', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"

section "system limits"
run df -h /dev/shm
run bash -c 'echo "ulimit -l (locked-memory KB): $(ulimit -l)"'
run bash -c 'echo "ulimit -n (open files):       $(ulimit -n)"'
run bash -c 'echo "ulimit -s (stack KB):         $(ulimit -s)"'

section "fabricmanager (required for H100 NVSwitch boxes)"
if command -v systemctl >/dev/null 2>&1; then
  run bash -c 'systemctl is-active nvidia-fabricmanager 2>&1 || true'
  run bash -c 'systemctl status nvidia-fabricmanager --no-pager 2>&1 | head -15 || true'
else
  echo "(systemctl not available)" | tee -a "$LOG"
fi
run bash -c 'pgrep -a fabricmanager 2>&1 || echo "(no fabricmanager process)"'

section "nvidia-smi topology matrix"
run nvidia-smi topo -m

section "current GPU processes (catch zombies / contention)"
run nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

section "NCCL init smoke test (single-process spawning N ranks)"
# Single-process torch.multiprocessing test: each rank inits NCCL, does one all_reduce.
# This isolates NCCL bring-up from anything else in the training pipeline.
NPROC="$(echo "$GPUS_CSV" | awk -F, '{print NF}')"
echo "[smoke] running NCCL all_reduce smoke with nproc=${NPROC} on GPUs ${GPUS_CSV}" | tee -a "$LOG"

env_prefix=(env)
env_prefix+=("CUDA_VISIBLE_DEVICES=$GPUS_CSV")
env_prefix+=("NCCL_DEBUG=INFO")
env_prefix+=("NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV")
if [[ "$EXTRA_MODE" == "nccl_p2p_off" ]]; then
  env_prefix+=("NCCL_P2P_DISABLE=1")
  echo "[smoke] mode=nccl_p2p_off — NCCL_P2P_DISABLE=1 set" | tee -a "$LOG"
fi
if [[ "$EXTRA_MODE" == "nccl_all_off" ]]; then
  env_prefix+=("NCCL_P2P_DISABLE=1" "NCCL_SHM_DISABLE=1" "NCCL_IB_DISABLE=1")
  echo "[smoke] mode=nccl_all_off — P2P + SHM + IB all disabled" | tee -a "$LOG"
fi

echo "[smoke] launching torchrun-style test (this is what crashes for you)" | tee -a "$LOG"
echo "+ ${env_prefix[*]} python -m torch.distributed.run --nproc_per_node=${NPROC} --master_port=29501 $SMOKE_PY" | tee -a "$LOG"

"${env_prefix[@]}" python -m torch.distributed.run \
  --nproc_per_node="$NPROC" \
  --master_port=29501 \
  "$SMOKE_PY" 2>&1 | tee -a "$LOG"
smoke_rc=${PIPESTATUS[0]}
echo "[smoke] exit_code=${smoke_rc}" | tee -a "$LOG"

section "summary"
{
  echo "- log_file:   $LOG"
  echo "- gpus:       $GPUS_CSV"
  echo "- nproc:      $NPROC"
  echo "- smoke_rc:   $smoke_rc"
  echo
  echo "Next step: paste $LOG (or just the NCCL init smoke section) to chat."
  echo "Common quick retries:"
  echo "  bash scripts/diagnose_nccl.sh $GPUS_CSV nccl_p2p_off"
  echo "  bash scripts/diagnose_nccl.sh $GPUS_CSV nccl_all_off"
} | tee -a "$LOG"

echo
echo "saved: $LOG"
