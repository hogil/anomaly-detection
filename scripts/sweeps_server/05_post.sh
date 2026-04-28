#!/usr/bin/env bash
# Run only instability/trend/report post-processing.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

run_paper_stage "post" --skip-weights --skip-dataset --skip-refcheck --skip-round1 --skip-round2 "$@"
