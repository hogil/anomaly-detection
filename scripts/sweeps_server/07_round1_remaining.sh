#!/usr/bin/env bash
# Resume strict round1 with completed/skipped tags removed from the prepared queue.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

run_paper_stage "round1_remaining" --skip-refcheck --round1-skip-completed --skip-round2 --skip-post "$@"
