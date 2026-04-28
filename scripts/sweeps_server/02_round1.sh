#!/usr/bin/env bash
# Run only strict round1, after weights/data preparation if needed.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

run_paper_stage "round1" --skip-refcheck --skip-round2 --skip-post "$@"
