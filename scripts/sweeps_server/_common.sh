#!/usr/bin/env bash
# Shared helpers for the paper server sweep wrappers.
#
# - detect_profile: read GPU memory + CPU count, set sane num_workers/prefetch
#   defaults so the same scripts work on a multi-H200 server and a desktop GPU
#   without separate "pc safe" plumbing. Users can still override via flags.
# - run_paper_stage / run_round1_axes: thin wrappers over run_paper_server_all.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PAPER_RUNNER="$ROOT_DIR/scripts/run_paper_server_all.sh"

cd "$ROOT_DIR"

# Auto-enable real DDP (torchrun) when 2+ GPUs are visible on Linux.
#
# Without this, train.py runs as a single Python process and silently wraps the
# model with nn.DataParallel, which suffers severe scaling loss past 2 GPUs
# (GPU 0 gather/scatter bottleneck). We export AD_TRAIN_DDP_NPROC so the
# adaptive controller launches each train.py through torch.distributed.run.
#
# Override:
#   AD_NO_DDP=1     leave AD_TRAIN_DDP_NPROC unset (forces DataParallel / single)
#   AD_TRAIN_DDP_NPROC=<N>  already set by caller; respected as-is.
#   CUDA_VISIBLE_DEVICES   limits the count we auto-detect.
auto_enable_ddp() {
  if [[ -n "${AD_TRAIN_DDP_NPROC:-}" ]]; then
    echo "[ddp] AD_TRAIN_DDP_NPROC=${AD_TRAIN_DDP_NPROC} (already set; honoring caller)"
    return 0
  fi
  if [[ "${AD_NO_DDP:-0}" == "1" ]]; then
    echo "[ddp] AD_NO_DDP=1; staying on single-process / DataParallel"
    return 0
  fi
  # NCCL backend is Linux-only; skip on Windows/Cygwin/Mac.
  case "${OSTYPE:-}" in
    msys*|cygwin*|win32*|darwin*)
      return 0 ;;
  esac
  local gpu_count=0
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{n=0; for (i=1; i<=NF; i++) if ($i != "") n++; print n}')
  elif command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | awk '{print $1}')
  fi
  if [[ -z "$gpu_count" ]] || ! [[ "$gpu_count" =~ ^[0-9]+$ ]]; then
    gpu_count=0
  fi
  if [[ "$gpu_count" -ge 2 ]]; then
    export AD_TRAIN_DDP_NPROC="$gpu_count"
    echo "[ddp] auto-enabled: $gpu_count GPUs visible -> torchrun --nproc-per-node=$gpu_count"
    echo "[ddp] each train.py run launches through torch.distributed.run; --batch_size is the GLOBAL batch (per-rank = global / $gpu_count)"
    echo "[ddp] override with AD_NO_DDP=1 to force single-process DataParallel"
  fi
}

# Scale --batch-size for DDP global-batch semantics.
#
# In DDP, train.py treats --batch_size as the global batch and divides by
# world_size for the per-rank micro-batch. The runtime profile (server / pc /
# minimal) PROFILE_BATCH_SIZE is the per-GPU intended size. So when we
# auto-enable DDP we also need to multiply the profile by NPROC, unless the
# caller already passed an explicit --batch-size.
auto_scale_batch_for_ddp() {
  local current_batch="${1:-}"
  if [[ -z "${AD_TRAIN_DDP_NPROC:-}" || "${AD_TRAIN_DDP_NPROC:-1}" -lt 2 ]]; then
    printf '%s' "$current_batch"
    return 0
  fi
  if [[ -n "$current_batch" ]]; then
    printf '%s' "$current_batch"
    return 0
  fi
  # Caller did not pass --batch-size; use profile per-GPU x NPROC as global.
  detect_profile >/dev/null
  local global_batch=$(( PROFILE_BATCH_SIZE * AD_TRAIN_DDP_NPROC ))
  echo "[ddp] auto --batch-size $global_batch (= profile per-GPU $PROFILE_BATCH_SIZE x NPROC $AD_TRAIN_DDP_NPROC)" >&2
  printf '%s' "$global_batch"
}

detect_profile() {
  # GPU total memory (MiB), 0 if no GPU.
  local gpu_mem=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    # Avoid `head -1` here. With `set -o pipefail`, nvidia-smi can receive
    # SIGPIPE on multi-GPU hosts when head exits early, aborting the wrapper
    # before run.log is even opened.
    local gpu_query
    gpu_query="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
    gpu_mem=$(printf '%s\n' "$gpu_query" \
      | awk 'NR == 1 { first = $1 } END { gsub(/[ \r]/, "", first); print first }')
    [[ -z "$gpu_mem" ]] && gpu_mem=0
  fi
  # CPU logical core count.
  local cpus
  cpus=$(nproc 2>/dev/null || echo 8)
  # System RAM in MiB. Linux: /proc/meminfo. Windows (Git Bash): WMIC fallback.
  local mem_mb=0
  if [[ -r /proc/meminfo ]]; then
    mem_mb=$(awk '/MemTotal/ {printf "%d\n", $2 / 1024}' /proc/meminfo 2>/dev/null || echo 0)
  fi
  if [[ "${mem_mb:-0}" -le 0 ]] && command -v wmic >/dev/null 2>&1; then
    local wmic_query
    wmic_query="$(wmic computersystem get TotalPhysicalMemory 2>/dev/null || true)"
    mem_mb=$(printf '%s\n' "$wmic_query" \
      | awk 'NR>1 && $1+0>0 && first == "" { first = $1 } END { if (first != "") printf "%d\n", first / 1048576 }')
    [[ -z "$mem_mb" ]] && mem_mb=0
  fi

  # workers default = cpus - 2 (OS 여유 2코어). cap 동일 식이라 사실상 cpus-2로 정렬됨.
  local default_workers=$((cpus - 2))
  [[ "$default_workers" -lt 0 ]] && default_workers=0

  if [[ "$gpu_mem" -ge 40000 && "${mem_mb:-0}" -ge 64000 ]]; then
    PROFILE_NAME="server"
    PROFILE_NUM_WORKERS=$default_workers
    PROFILE_PREFETCH=4
    PROFILE_MAX_LAUNCHED=0
    PROFILE_BATCH_SIZE=256
  elif [[ "$gpu_mem" -ge 12000 && "${mem_mb:-0}" -ge 16000 ]]; then
    PROFILE_NAME="pc"
    PROFILE_NUM_WORKERS=2
    PROFILE_PREFETCH=2
    PROFILE_MAX_LAUNCHED=0
    PROFILE_BATCH_SIZE=32
  else
    PROFILE_NAME="minimal"
    PROFILE_NUM_WORKERS=0
    PROFILE_PREFETCH=1
    PROFILE_MAX_LAUNCHED=0
    PROFILE_BATCH_SIZE=8
  fi

  # 안전망: 어떤 프로파일이든 cpus-2 초과 금지 (OS 응답성).
  local cap=$((cpus - 2))
  [[ "$cap" -lt 0 ]] && cap=0
  if [[ "$PROFILE_NUM_WORKERS" -gt "$cap" ]]; then
    PROFILE_NUM_WORKERS=$cap
  fi

  echo "[profile] $PROFILE_NAME gpu=${gpu_mem}MB ram=${mem_mb}MB cpu=${cpus} workers=$PROFILE_NUM_WORKERS prefetch=$PROFILE_PREFETCH batch=$PROFILE_BATCH_SIZE max_launched=$PROFILE_MAX_LAUNCHED"
}

run_paper_stage() {
  local stage="$1"
  shift
  echo "== paper stage: $stage =="
  local code=0
  bash "$PAPER_RUNNER" "$@" || code=$?
  if [[ "$code" -ne 0 ]]; then
    echo "[stage failed] $stage exit=$code command=bash $PAPER_RUNNER $*" >&2
  fi
  return "$code"
}

run_round1_axes() {
  local stage="$1"
  local axes="$2"
  shift 2
  run_paper_stage "$stage" \
    --skip-weights \
    --skip-dataset \
    --skip-refcheck \
    --round1-skip-completed \
    --round1-include-axes "$axes" \
    --skip-round2 \
    --skip-post \
    "$@"
}
