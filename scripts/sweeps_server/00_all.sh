#!/usr/bin/env bash
# Run the full paper server pipeline.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

run_paper_stage "all" "$@"
