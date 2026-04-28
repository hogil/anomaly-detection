#!/usr/bin/env bash
# Summarize logs safely from the repository root.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
OUT_PREFIX="${OUT_PREFIX:-validations/log_folder_summary}"

exec "$PYTHON" scripts/summarize_log_folders.py \
  --logs-dir logs \
  --out-prefix "$OUT_PREFIX" \
  "$@"
