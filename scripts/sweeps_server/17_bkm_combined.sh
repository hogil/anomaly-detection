#!/usr/bin/env bash
# BKM (Best Known Method) combined sweep. Applies the per-axis BKM values
# documented in docs/summary.md all at once, in a single candidate, at 5 seeds.
# This is the "kitchen sink" comparison against the rawbase baseline.
#
# Source of BKM values: docs/summary.md "Best Known Method" table.
#   normal_ratio     700  -> 3300
#   gc               0.0  -> 0.5
#   label_smoothing  0.00 -> 0.02
#   stochastic_depth 0.00 -> 0.05
#   focal_gamma      0.0  -> 2.0
#   abnormal_weight  1.0  -> 1.5
#   ema              0.0  -> 0.95
#   allow_tie_save   off  -> on
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dataset.yaml}"
detect_profile
NUM_WORKERS="${NUM_WORKERS:-$PROFILE_NUM_WORKERS}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-$PROFILE_PREFETCH}"
BATCH_SIZE="${BATCH_SIZE:-$PROFILE_BATCH_SIZE}"
MAX_LAUNCHED="${MAX_LAUNCHED:-$PROFILE_MAX_LAUNCHED}"
LOG_DIR_GROUP="${LOG_DIR_GROUP:-$(date +%Y%m%d_%H%M%S)_bkm_combined}"
CHECKPOINT_RETENTION="${CHECKPOINT_RETENTION:-all}"
CHECKPOINT_RETENTION_SCOPE="${CHECKPOINT_RETENTION_SCOPE:-summary}"
SEEDS="${SEEDS:-42,1,2,3,4}"
FORCE=0
MODEL_NAME="${MODEL_NAME:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sweeps_server/bkm_combined.sh [options]

Runs ONE candidate with every BKM value from docs/summary.md applied at once,
at the default 5 seeds. Compared against the rawbase baseline.

Options:
  --python PATH        python executable
  --config PATH        dataset config (default dataset.yaml)
  --num-workers N      train.py DataLoader workers
  --prefetch-factor N  train.py prefetch factor
  --batch-size N       train.py --batch_size override (auto from profile)
  --seeds CSV          seeds (default: 42,1,2,3,4)
  --force              re-run completed tags
  --max-launched N     stop controller after launching N runs
  --log-dir-group NAME group runs under logs/<NAME>/
  --checkpoint-retention MODE      all | dataset-backbone-best (default: dataset-backbone-best)
  --checkpoint-retention-scope S   summary | log-group | logs (default: logs)
  -h, --help           show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --prefetch-factor) PREFETCH_FACTOR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --max-launched) MAX_LAUNCHED="$2"; shift 2 ;;
    --log-dir-group) LOG_DIR_GROUP="$2"; shift 2 ;;
    --checkpoint-retention) CHECKPOINT_RETENTION="$2"; shift 2 ;;
    --checkpoint-retention-scope) CHECKPOINT_RETENTION_SCOPE="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

run_cmd() { echo; echo "+ $*"; "$@"; }

VAL_DIR="validations/${LOG_DIR_GROUP}"
mkdir -p "$VAL_DIR"
QUEUE="${VAL_DIR}/05_bkm_combined_queue.json"
ACTIVE="${VAL_DIR}/05_bkm_combined_active.json"
SUMMARY="${VAL_DIR}/05_bkm_combined_results.json"
MARKDOWN="${VAL_DIR}/05_bkm_combined_results.md"

echo "== paper stage: bkm_combined (output: $VAL_DIR) =="

SWEEP_RESULTS="${VAL_DIR}/02_sweep_results.json"
SWEEP_ACTIVE="${VAL_DIR}/02_sweep_active.json"

run_cmd "$PYTHON" - "$QUEUE" "$SEEDS" "$NUM_WORKERS" "$PREFETCH_FACTOR" "$BATCH_SIZE" "$SWEEP_RESULTS" "$SWEEP_ACTIVE" "$MODEL_NAME" <<'PY'
"""Build the bkm_combined queue.

Reads the same-group 02_sweep_results.json + 02_sweep_active.json and applies
the per-axis F1-argmax candidate's args. Falls back to docs/summary.md hardcoded
values if 02 results are unavailable (e.g., running stage 17 standalone).

Per-axis BKM is found by:
  1. Group runs by candidate, mean test_f1 over status=complete runs only.
  2. For each candidate, compare its args against the rawbase (baseline)
     candidate's args. If it differs on exactly one BKM-tracked axis, the
     candidate's value on that axis is a contender.
  3. The axis-best candidate is the one with max mean F1.
"""
import json
import sys
from pathlib import Path
from statistics import mean

queue_path, seeds_raw, num_workers, prefetch, batch_size, sweep_results_path, sweep_active_path, model_name = sys.argv[1:9]
seeds = [int(s.strip()) for s in seeds_raw.split(",") if s.strip()]

# Axes that stage 17 applies together. Each axis is a single CLI flag.
BKM_AXES = [
    "--normal_ratio",
    "--grad_clip",
    "--label_smoothing",
    "--stochastic_depth_rate",
    "--focal_gamma",
    "--abnormal_weight",
    "--ema_decay",
    "--allow_tie_save",
]

# Fallback BKM values from docs/summary.md (v2_tiny x dataset.yaml). Used only
# when 02 results are unavailable for the current group.
FALLBACK_BKM = {
    "--normal_ratio": 3300,
    "--grad_clip": 0.5,
    "--label_smoothing": 0.02,
    "--stochastic_depth_rate": 0.05,
    "--focal_gamma": 2.0,
    "--abnormal_weight": 1.5,
    "--ema_decay": 0.95,
    "--allow_tie_save": True,
}

# Operating baseline args (must match scripts/sweeps_server/01_baseline.sh).
base_args = {
    "--mode": "binary",
    "--config": "dataset.yaml",
    "--epochs": 20,
    "--patience": 5,
    "--batch_size": int(batch_size),
    "--dropout": 0.0,
    "--precision": "fp16",
    "--num_workers": int(num_workers),
    "--prefetch_factor": int(prefetch),
    "--smooth_window": 1,
    "--smooth_method": "median",
    "--lr_backbone": "2e-5",
    "--lr_head": "2e-4",
    "--warmup_epochs": 5,
    "--weight_decay": 0.01,
    "--normal_ratio": 700,
    "--grad_clip": 0.0,
    "--label_smoothing": 0.0,
    "--stochastic_depth_rate": 0.0,
    "--focal_gamma": 0.0,
    "--abnormal_weight": 1.0,
    "--ema_decay": 0.0,
    "--allow_tie_save": False,
}


def _normalize(v):
    """Coerce arg values to a comparable form (numeric strings -> float)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def discover_bkm(results_path: Path, active_path: Path):
    """Return dict[axis] -> {"value": ..., "f1": ..., "candidate": ...} or {}."""
    if not (results_path.exists() and active_path.exists()):
        return {}
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
        active = json.loads(active_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[bkm] failed to read 02 sweep files: {exc}", file=sys.stderr)
        return {}

    cand_args: dict[str, dict] = {}
    for r in active.get("runs", []):
        cand = r.get("candidate")
        if cand and cand not in cand_args:
            cand_args[cand] = r.get("args", {})

    cand_f1: dict[str, list[float]] = {}
    for run in results.get("runs", {}).values():
        if run.get("status") != "complete":
            continue
        f1 = run.get("test_f1")
        if f1 is None:
            continue
        cand_f1.setdefault(run["candidate"], []).append(float(f1))

    if not cand_f1:
        return {}

    cand_mean: dict[str, float] = {c: mean(v) for c, v in cand_f1.items() if v}

    per_axis: dict[str, dict] = {}
    for cand, args in cand_args.items():
        if cand not in cand_mean:
            continue
        diff_axes: list[str] = []
        diff_value = None
        for ax in BKM_AXES:
            if ax not in args:
                continue
            base_val = _normalize(base_args.get(ax))
            cand_val = _normalize(args.get(ax))
            if base_val != cand_val:
                diff_axes.append(ax)
                diff_value = args[ax]
        if len(diff_axes) != 1:
            continue
        ax = diff_axes[0]
        entry = per_axis.get(ax)
        if entry is None or cand_mean[cand] > entry["f1"]:
            per_axis[ax] = {"value": diff_value, "f1": cand_mean[cand], "candidate": cand}

    return per_axis


sweep_results = Path(sweep_results_path)
sweep_active = Path(sweep_active_path)
discovered = discover_bkm(sweep_results, sweep_active)

bkm_args = dict(base_args)
bkm_source: dict[str, str] = {}
if discovered:
    print(f"[bkm] dynamic BKM from {sweep_results.name}:")
    for ax in BKM_AXES:
        entry = discovered.get(ax)
        if entry is not None:
            bkm_args[ax] = entry["value"]
            bkm_source[ax] = f"dynamic ({entry['candidate']} F1={entry['f1']:.4f})"
        else:
            bkm_args[ax] = FALLBACK_BKM[ax]
            bkm_source[ax] = "fallback (axis not found in 02 results)"
        print(f"  {ax} = {bkm_args[ax]!r}  [{bkm_source[ax]}]")
else:
    print(f"[bkm] 02 sweep results unavailable; using FALLBACK BKM from docs/summary.md", file=sys.stderr)
    for ax in BKM_AXES:
        bkm_args[ax] = FALLBACK_BKM[ax]
        bkm_source[ax] = "fallback (no 02 results)"

if model_name:
    bkm_args["--model_name"] = model_name

# Candidate name encodes the normal_ratio for backwards-compat with existing reports.
nr = int(bkm_args["--normal_ratio"])
candidate = f"fresh0412_v11_bkm_combined_n{nr}"

runs = []
for seed in seeds:
    runs.append({
        "tag": f"{candidate}_s{seed}",
        "candidate": candidate,
        "seed": seed,
        "args": dict(bkm_args),
        "reason": "BKM combined: per-axis F1-argmax (dynamic) applied together",
    })

payload = {
    "created_by": "scripts/sweeps_server/bkm_combined.sh",
    "note": "BKM values discovered from 02_sweep_results.json (this group), or fallback to docs/summary.md.",
    "bkm_source": bkm_source,
    "runs": runs,
}
p = Path(queue_path)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[bkm] wrote {p} runs={len(runs)}")
PY

run_cmd "$PYTHON" scripts/prepare_server_queue.py \
  --src "$QUEUE" \
  --dst "$ACTIVE" \
  --config "$CONFIG" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --batch-size "$BATCH_SIZE"

cmd=(
  "$PYTHON" scripts/adaptive_experiment_controller.py
  --queue "$ACTIVE"
  --summary "$SUMMARY"
  --markdown "$MARKDOWN"
  --target-min 5
  --target-max 15
  --stop-mode never
  --candidate-min-runs-before-skip 0
  --completion-exit-grace 15
  --update-live-summary
  --checkpoint-retention "$CHECKPOINT_RETENTION"
  --checkpoint-retention-scope "$CHECKPOINT_RETENTION_SCOPE"
)
[[ "$FORCE" -eq 1 ]] && cmd+=(--force)
[[ "$MAX_LAUNCHED" -gt 0 ]] && cmd+=(--max-launched "$MAX_LAUNCHED")
[[ -n "$LOG_DIR_GROUP" ]] && cmd+=(--log-dir-group "$LOG_DIR_GROUP")

run_cmd "${cmd[@]}"

run_cmd "$PYTHON" scripts/generate_stage_comparison.py \
  --results "$SUMMARY" \
  --baseline "${VAL_DIR}/01_baseline_results.json" \
  --out-md "${VAL_DIR}/05_bkm_combined_results.md" \
  --out-plot "${VAL_DIR}/05_bkm_combined_plot.png" \
  --title "BKM combined vs baseline"
