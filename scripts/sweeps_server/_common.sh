#!/usr/bin/env bash
# Shared helpers for paper server pipeline wrappers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PAPER_RUNNER="$ROOT_DIR/scripts/run_paper_server_all.sh"

cd "$ROOT_DIR"

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
