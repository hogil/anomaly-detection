#!/usr/bin/env python
"""Prepare server queues with the current raw baseline policy."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RAW_REFERENCE = "fresh0412_v11_refcheck_raw_n700"
RAW_DUPLICATE_GC00 = "fresh0412_v11_rawbase_gc00_n700"
DEFAULT_GC_CANDIDATES = (
    "fresh0412_v11_gc01_n700",
    "fresh0412_v11_gc025_n700",
    "fresh0412_v11_gc05_n700",
    "fresh0412_v11_gc15_n700",
    "fresh0412_v11_gc50_n700",
)


def candidate_name(run: dict[str, Any]) -> str:
    candidate = run.get("candidate")
    if candidate:
        return str(candidate)
    return re.sub(r"_s\d+$", "", str(run.get("tag", "")))


def normalize_candidate(candidate: str) -> str:
    if candidate.startswith("fresh0412_v11_rawbase_"):
        return "fresh0412_v11_" + candidate.removeprefix("fresh0412_v11_rawbase_")
    return candidate


def infer_axis(candidate: str) -> str:
    candidate = normalize_candidate(candidate)
    if re.search(r"_gc(?:\d|$)", candidate):
        return "gc"
    if "_lrwarm" in candidate:
        return "warmup"
    if re.search(r"_n\d+$", candidate):
        return "normal_ratio"
    if "_regls" in candidate:
        return "label_smoothing"
    if "_regdp" in candidate:
        return "stochastic_depth"
    if "_fg" in candidate:
        return "focal_gamma"
    if "_aw" in candidate:
        return "abnormal_weight"
    if "_ema" in candidate:
        return "ema"
    if "_color_" in candidate:
        return "color"
    if "_tie_" in candidate:
        return "allow_tie_save"
    return "other"


def rewrite_for_raw_server_baseline(run: dict[str, Any]) -> str:
    old_candidate = candidate_name(run)
    if (
        old_candidate.startswith("fresh0412_v11_rawbase_")
        or "_refcheck_raw_" in old_candidate
        or "_lossfilter_raw_" in old_candidate
    ):
        return old_candidate
    if not old_candidate.startswith("fresh0412_v11_"):
        return old_candidate

    new_candidate = "fresh0412_v11_rawbase_" + old_candidate.removeprefix("fresh0412_v11_")
    run["candidate"] = new_candidate
    old_tag = str(run.get("tag", ""))
    if old_tag.startswith(old_candidate):
        run["tag"] = new_candidate + old_tag[len(old_candidate):]
    elif old_tag.startswith("fresh0412_v11_"):
        run["tag"] = "fresh0412_v11_rawbase_" + old_tag.removeprefix("fresh0412_v11_")
    return new_candidate


def is_duplicate_raw_gc00(candidate: str) -> bool:
    return candidate == RAW_DUPLICATE_GC00 or normalize_candidate(candidate) == "fresh0412_v11_gc00_n700"


def parse_gc_candidates(raw: str) -> set[str]:
    out: set[str] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("gc"):
            item = f"fresh0412_v11_{item}_n700"
        out.add(normalize_candidate(item))
    return out


def trim_runs(
    runs: list[dict[str, Any]],
    src: Path,
    start_after_axis: str,
    start_after_candidate: str,
) -> list[dict[str, Any]]:
    if not start_after_axis and not start_after_candidate:
        return runs

    normalized_target = normalize_candidate(start_after_candidate) if start_after_candidate else ""
    last_idx = -1
    for idx, run in enumerate(runs):
        candidate = candidate_name(run)
        if normalized_target and normalize_candidate(candidate) == normalized_target:
            last_idx = idx
        if start_after_axis and infer_axis(candidate) == start_after_axis:
            last_idx = idx

    if last_idx < 0:
        target = start_after_candidate or f"axis:{start_after_axis}"
        raise SystemExit(f"start-after target not found in queue {src}: {target}")
    kept = runs[last_idx + 1 :]
    if not kept:
        target = start_after_candidate or f"axis:{start_after_axis}"
        raise SystemExit(f"queue has no runs after start-after target: {target}")
    print(f"[queue] trimmed {len(runs) - len(kept)} runs before start-after target; kept {len(kept)}")
    return kept


def prepare_queue(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.src.read_text(encoding="utf-8"))
    payload["server_rewritten_from"] = str(args.src)
    payload["selected_reference"] = args.raw_reference
    payload["server_config"] = args.config
    payload["server_num_workers"] = args.num_workers
    payload["server_baseline"] = {
        "candidate": args.raw_reference,
        "grad_clip": 0.0,
        "smooth_window": 1,
        "smooth_method": "median",
    }
    if args.start_after_axis:
        payload["server_start_after_axis"] = args.start_after_axis
    if args.start_after_candidate:
        payload["server_start_after_candidate"] = args.start_after_candidate
    allowed_gc_candidates = parse_gc_candidates(args.gc_candidates)
    payload["server_gc_candidates"] = sorted(allowed_gc_candidates)

    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        if args.start_after_axis or args.start_after_candidate:
            raise SystemExit(f"queue has no explicit runs list to trim: {args.src}")
        runs = []
    else:
        payload["runs"] = trim_runs(
            runs,
            args.src,
            args.start_after_axis,
            args.start_after_candidate,
        )

    prepared_runs: list[dict[str, Any]] = []
    skipped_duplicate_controls: list[str] = []
    skipped_gc_candidates: list[str] = []
    for run in payload.get("runs", []):
        candidate = rewrite_for_raw_server_baseline(run)
        if is_duplicate_raw_gc00(candidate):
            skipped_duplicate_controls.append(candidate)
            continue
        axis = infer_axis(candidate)
        if axis == "gc" and normalize_candidate(candidate) not in allowed_gc_candidates:
            skipped_gc_candidates.append(candidate)
            continue
        run_args = run.setdefault("args", {})
        if not isinstance(run_args, dict):
            raise SystemExit(f"run args must be a JSON object: {run.get('tag')}")
        if axis != "gc":
            run_args["--grad_clip"] = 0.0
        if axis != "smoothing":
            run_args["--smooth_window"] = 1
            run_args["--smooth_method"] = "median"
        run_args["--config"] = args.config
        run_args["--num_workers"] = args.num_workers
        run_args["--prefetch_factor"] = args.prefetch_factor
        prepared_runs.append(run)

    payload["runs"] = prepared_runs
    if skipped_duplicate_controls:
        payload["server_skipped_duplicate_controls"] = sorted(set(skipped_duplicate_controls))
        print(
            "[queue] skipped duplicate raw controls: "
            + ", ".join(payload["server_skipped_duplicate_controls"])
        )
    if skipped_gc_candidates:
        payload["server_skipped_gc_candidates"] = sorted(set(skipped_gc_candidates))
        print(
            "[queue] skipped extra GC candidates: "
            + ", ".join(payload["server_skipped_gc_candidates"])
        )

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--config", default="dataset.yaml")
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--start-after-axis", default="")
    parser.add_argument("--start-after-candidate", default="")
    parser.add_argument("--raw-reference", default=RAW_REFERENCE)
    parser.add_argument(
        "--gc-candidates",
        default=",".join(DEFAULT_GC_CANDIDATES),
        help="Comma-separated GC candidates to keep. Tokens like gc01 are accepted. Default keeps 5 GC conditions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = prepare_queue(args)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[queue] wrote {args.dst} runs={len(payload.get('runs', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
