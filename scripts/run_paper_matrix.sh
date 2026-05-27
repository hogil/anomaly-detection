#!/usr/bin/env bash
# One-command paper matrix run: default datasets x default backbones.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
cd "$D/.."

PYTHON="${PYTHON:-python}"
"$PYTHON" download.py

exec bash "$D/all-dataset-backbone.sh" -x --reset-data "$@"
