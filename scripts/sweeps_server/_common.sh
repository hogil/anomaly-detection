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

detect_profile() {
  # GPU total memory (MiB), 0 if no GPU.
  local gpu_mem=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
      | head -1 | tr -d ' \r')
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
    mem_mb=$(wmic computersystem get TotalPhysicalMemory 2>/dev/null \
      | awk 'NR>1 && $1+0>0 {printf "%d\n", $1 / 1048576}' | head -1)
    [[ -z "$mem_mb" ]] && mem_mb=0
  fi

  if [[ "$gpu_mem" -ge 40000 && "${mem_mb:-0}" -ge 64000 ]]; then
    PROFILE_NAME="server"
    PROFILE_NUM_WORKERS=24
    PROFILE_PREFETCH=4
    PROFILE_MAX_LAUNCHED=0
  elif [[ "$gpu_mem" -ge 12000 && "${mem_mb:-0}" -ge 16000 ]]; then
    PROFILE_NAME="pc"
    PROFILE_NUM_WORKERS=2
    PROFILE_PREFETCH=2
    PROFILE_MAX_LAUNCHED=1
  else
    PROFILE_NAME="minimal"
    PROFILE_NUM_WORKERS=0
    PROFILE_PREFETCH=1
    PROFILE_MAX_LAUNCHED=1
  fi

  # cap workers at cpus-2 so OS stays responsive.
  local cap=$((cpus - 2))
  [[ "$cap" -lt 0 ]] && cap=0
  if [[ "$PROFILE_NUM_WORKERS" -gt "$cap" ]]; then
    PROFILE_NUM_WORKERS=$cap
  fi

  echo "[profile] $PROFILE_NAME gpu=${gpu_mem}MB ram=${mem_mb}MB cpu=${cpus} workers=$PROFILE_NUM_WORKERS prefetch=$PROFILE_PREFETCH max_launched=$PROFILE_MAX_LAUNCHED"
}

run_paper_stage() {
  local stage="$1"
  shift
  echo "== paper stage: $stage =="
  bash "$PAPER_RUNNER" "$@"
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
