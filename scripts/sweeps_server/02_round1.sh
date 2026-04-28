#!/usr/bin/env bash
# Run only strict round1, skipping the already-completed GC block.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

run_paper_stage "round1_after_gc" --skip-refcheck --round1-after-gc --skip-round2 --skip-post "$@"
