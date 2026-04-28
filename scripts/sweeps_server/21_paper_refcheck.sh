#!/usr/bin/env bash
# Run only the refcheck queue, after weights/data preparation if needed.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_paper_common.sh"

run_paper_stage "refcheck" --skip-round1 --skip-round2 --skip-post "$@"
