"""
Select paper-axis candidates that deserve seeds 3 and 4 after the fast 3-seed pass.

The first pass is intentionally broad and quick. This selector keeps the paper
pipeline automatic by expanding only candidates that are useful for a claim:
best per axis, close to the selected reference, or clearly moving FN/FP.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def queue_runs(queue: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(queue, list):
        return queue
    runs: list[dict[str, Any]] = []
    defaults = queue.get("defaults", {})
    default_seeds = defaults.get("seeds", [42, 1, 2, 3, 4])
    default_args = defaults.get("args", {})
    for candidate in queue.get("candidates", []):
        args = dict(default_args)
        args.update(candidate.get("args", {}))
        name = str(candidate["name"])
        tag_template = candidate.get("tag_template", f"{name}_s{{seed}}")
        for seed in candidate.get("seeds", default_seeds):
            runs.append({
                "tag": str(tag_template).format(seed=int(seed), name=name),
                "candidate": name,
                "seed": int(seed),
                "args": args,
                "reason": candidate.get("reason", ""),
            })
    runs.extend(queue.get("runs", []))
    return runs


def candidate_axis(candidate: str) -> str:
    if "_fg" in candidate:
        return "focal_gamma"
    if "_aw" in candidate:
        return "abnormal_weight"
    if "_ema" in candidate:
        return "ema"
    if "_tie_" in candidate:
        return "tie_save"
    if "_color_" in candidate:
        return "color"
    return "other"


def run_exists(tag: str) -> bool:
    logs = ROOT / "logs"
    for path in logs.glob(f"*{tag}*"):
        if path.is_dir() and (path / "best_info.json").exists():
            return True
    direct = logs / tag
    return direct.is_dir() and (direct / "best_info.json").exists()


def metric(row: dict[str, Any], key: str, default: float) -> float:
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def candidate_score(row: dict[str, Any], ref_f1: float, ref_fn: float, ref_fp: float) -> tuple[float, float, float]:
    f1 = metric(row, "f1_mean", 0.0)
    fn = metric(row, "fn_mean", 999.0)
    fp = metric(row, "fp_mean", 999.0)
    balance_gap = abs(fn - ref_fn) + abs(fp - ref_fp)
    return (-f1, balance_gap, fn + fp)


def select_candidates(
    by_candidate: dict[str, Any],
    *,
    ref_f1: float,
    ref_fn: float,
    ref_fp: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate, row in by_candidate.items():
        if int(row.get("complete") or 0) < 3:
            continue
        enriched = dict(row)
        enriched["candidate"] = candidate
        enriched["axis"] = candidate_axis(candidate)
        rows.append(enriched)

    selected: dict[str, dict[str, Any]] = {}

    axes = sorted({row["axis"] for row in rows})
    for axis in axes:
        axis_rows = [row for row in rows if row["axis"] == axis]
        if not axis_rows:
            continue
        axis_rows.sort(key=lambda row: candidate_score(row, ref_f1, ref_fn, ref_fp))
        best = axis_rows[0]
        best["selection_reason"] = f"best candidate in {axis}"
        selected[best["candidate"]] = best

    for row in rows:
        f1 = metric(row, "f1_mean", 0.0)
        fn = metric(row, "fn_mean", 999.0)
        fp = metric(row, "fp_mean", 999.0)
        reasons: list[str] = []
        if f1 >= ref_f1 - 0.002:
            reasons.append(f"F1 within 0.002 of selected ref ({f1:.4f})")
        if fn <= ref_fn - 1.0:
            reasons.append(f"FN lower than selected ref ({fn:.2f} vs {ref_fn:.2f})")
        if fp <= ref_fp - 1.0:
            reasons.append(f"FP lower than selected ref ({fp:.2f} vs {ref_fp:.2f})")
        if reasons:
            copy = dict(row)
            copy["selection_reason"] = "; ".join(reasons)
            selected[row["candidate"]] = copy

    ranked = sorted(
        selected.values(),
        key=lambda row: (
            row["axis"],
            candidate_score(row, ref_f1, ref_fn, ref_fp),
            row["candidate"],
        ),
    )
    return ranked[:max_candidates]


def make_expansion_runs(
    selected: list[dict[str, Any]],
    source_runs: list[dict[str, Any]],
    seeds: list[int],
) -> list[dict[str, Any]]:
    args_by_candidate: dict[str, dict[str, Any]] = {}
    reason_by_candidate: dict[str, str] = {}
    for run in source_runs:
        candidate = str(run.get("candidate") or "")
        if candidate and candidate not in args_by_candidate and isinstance(run.get("args"), dict):
            args_by_candidate[candidate] = dict(run["args"])
            reason_by_candidate[candidate] = str(run.get("reason") or "")

    runs: list[dict[str, Any]] = []
    for row in selected:
        candidate = row["candidate"]
        args = args_by_candidate.get(candidate)
        if not args:
            continue
        for seed in seeds:
            tag = f"{candidate}_s{seed}"
            if run_exists(tag):
                continue
            runs.append({
                "tag": tag,
                "candidate": candidate,
                "seed": int(seed),
                "args": args,
                "reason": (
                    f"Auto-expanded after 3-seed paper-axis pass: "
                    f"{row.get('selection_reason', '')}. "
                    f"Original axis reason: {reason_by_candidate.get(candidate, '')}"
                ).strip(),
            })
    return runs


def write_decision_markdown(path: Path, selected: list[dict[str, Any]], runs: list[dict[str, Any]]) -> None:
    lines = [
        "# Paper Axis Expansion Decision",
        "",
        f"- Updated: `{now_iso()}`",
        f"- Selected candidates: `{len(selected)}`",
        f"- Expansion runs queued: `{len(runs)}`",
        "",
        "| Candidate | Axis | Complete | F1 Mean | FN Mean | FP Mean | Reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in selected:
        lines.append(
            "| {candidate} | {axis} | {complete} | {f1} | {fn} | {fp} | {reason} |".format(
                candidate=row.get("candidate", ""),
                axis=row.get("axis", ""),
                complete=row.get("complete", ""),
                f1=row.get("f1_mean", ""),
                fn=row.get("fn_mean", ""),
                fp=row.get("fp_mean", ""),
                reason=row.get("selection_reason", ""),
            )
        )
    if not selected:
        lines.append("| none | - | - | - | - | - | no 3-seed candidate qualified |")
    write_text(path, "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Select paper-axis candidates for seed expansion")
    parser.add_argument("--source-queue", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-queue", type=Path, required=True)
    parser.add_argument("--decision-md", type=Path, required=True)
    parser.add_argument("--ref-f1", type=float, default=0.9901)
    parser.add_argument("--ref-fn", type=float, default=9.8)
    parser.add_argument("--ref-fp", type=float, default=5.0)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 4])
    args = parser.parse_args()

    source_queue = read_json(args.source_queue, {})
    summary = read_json(args.summary, {})
    by_candidate = summary.get("aggregates", {}).get("by_candidate", {})
    selected = select_candidates(
        by_candidate,
        ref_f1=args.ref_f1,
        ref_fn=args.ref_fn,
        ref_fp=args.ref_fp,
        max_candidates=args.max_candidates,
    )
    runs = make_expansion_runs(selected, queue_runs(source_queue), args.seeds)
    payload = {
        "created_at": now_iso(),
        "source_queue": str(args.source_queue),
        "source_summary": str(args.summary),
        "selected_reference": "fresh0412_v11_n700_existing",
        "selection": selected,
        "runs": runs,
    }
    write_json(args.out_queue, payload)
    write_decision_markdown(args.decision_md, selected, runs)
    print(json.dumps({"selected": len(selected), "runs": len(runs)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
