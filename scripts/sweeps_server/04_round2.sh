#!/usr/bin/env bash
# Run round2 selection and queue only. Requires a completed server round1 summary.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

run_paper_stage "round2" --skip-weights --skip-dataset --skip-refcheck --skip-round1 --skip-post "$@"
