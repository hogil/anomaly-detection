#!/usr/bin/env bash
# One-command full server run for the default dataset.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
cd "$D/.."

PYTHON="${PYTHON:-python}"
"$PYTHON" scripts/check_torch_runtime.py

exec bash "$D/all-dataset-backbone.sh" --datasets dataset.yaml --reset-data "$@"
