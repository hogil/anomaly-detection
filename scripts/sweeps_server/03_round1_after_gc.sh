#!/usr/bin/env bash
# Resume strict round1 after the GC candidate block.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

run_paper_stage "round1_after_gc" --skip-refcheck --round1-after-gc --skip-round2 --skip-post "$@"
