#!/usr/bin/env bash
# Run the full paper server pipeline, skipping the already-completed GC block.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

run_paper_stage "all_after_gc" --round1-after-gc "$@"
