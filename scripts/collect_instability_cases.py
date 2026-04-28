"""
Collect collapse, oscillation, and gradient-spike cases for paper stability claims.

This does not launch training. It tags completed or in-progress runs so later
grad-clip and smoothing ablations can be compared against concrete failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def parse_scalar(text: str) -> Any:
    text = text.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"null", "none"}:
        return None
    try:
        if re.search(r"[.eE]", text):
            return float(text)
        return int(text)
    except ValueError:
        return text.strip("'\"")


def read_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key and not key.startswith(" "):
            data[key.strip()] = parse_scalar(value)
    return data


def clean_tag(name: str) -> str:
    match = re.match(r"^\d{6}_\d{6}_(.*?)(?:_F[0-9.]+_R[0-9.]+)?$", name)
    return match.group(1) if match else name


def seed_from_tag(tag: str) -> int | None:
    match = re.search(r"_s(\d+)$", tag)
    return int(match.group(1)) if match else None


def candidate_from_tag(tag: str) -> str:
    return re.sub(r"_s\d+$", "", tag)


def finite_values(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out.append(val)
    return out


def post_peak_drop(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    peak = max(values)
    peak_idx = values.index(peak)
    later = values[peak_idx + 1 :]
    if not later:
        return None
    return peak - min(later)


def metric_counts(best_info: dict[str, Any], test_events: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    metrics = best_info.get("test_metrics") or {}
    abnormal = metrics.get("abnormal") or {}
    normal = metrics.get("normal") or {}
    fn = abnormal.get("false_negatives")
    fp = normal.get("false_positives")
    if fn is None or fp is None:
        event = test_events[-1] if test_events else {}
        fn = event.get("fn", fn)
        fp = event.get("fp", fp)
    if fn is None:
        recall = abnormal.get("recall", best_info.get("test_abn_recall"))
        count = abnormal.get("count", abnormal.get("support", 750))
        if recall is not None:
            fn = round((1.0 - float(recall)) * int(count))
    if fp is None:
        recall = normal.get("recall", best_info.get("test_nor_recall"))
        count = normal.get("count", normal.get("support", 750))
        if recall is not None:
            fp = round((1.0 - float(recall)) * int(count))
    return (int(fn) if fn is not None else None, int(fp) if fp is not None else None)


def collect_test_events(best_info: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    events = read_json(run_dir / "test_history.json", None)
    if isinstance(events, list):
        return [e for e in events if isinstance(e, dict)]
    embedded = best_info.get("test_history")
    if isinstance(embedded, list):
        return [e for e in embedded if isinstance(e, dict)]
    return []


def axis_from_candidate(candidate: str) -> str:
    if "_regls" in candidate or "_ls" in candidate:
        return "label_smoothing"
    if "_regdp" in candidate or "_dp" in candidate:
        return "stochastic_depth"
    if "_lr" in candidate or "_lrwarm" in candidate:
        return "lr_warmup"
    if "_gc" in candidate:
        return "grad_clip"
    if "_sw" in candidate or "_smooth" in candidate:
        return "smoothing"
    if "_wd" in candidate:
        return "weight_decay"
    if "_n" in candidate:
        return "normal_ratio"
    return "other"


def summarize_run(run_dir: Path) -> dict[str, Any] | None:
    best_info = read_json(run_dir / "best_info.json", {})
    history = read_json(run_dir / "history.json", [])
    if not best_info and not history:
        return None
    if not isinstance(best_info, dict):
        best_info = {}
    if not isinstance(history, list):
        history = []

    tag = clean_tag(run_dir.name)
    candidate = candidate_from_tag(tag)
    seed = seed_from_tag(tag)
    config = read_simple_yaml(run_dir / "train_config_used.yaml")
    test_events = collect_test_events(best_info, run_dir)
    fn, fp = metric_counts(best_info, test_events)

    best_f1 = best_info.get("test_f1")
    if best_f1 is None and test_events:
        best_f1 = test_events[-1].get("test_f1")
    best_epoch = best_info.get("epoch", best_info.get("best_epoch"))

    val_f1s = finite_values([row.get("val_f1") for row in history if isinstance(row, dict)])
    val_losses = finite_values([row.get("val_loss") for row in history if isinstance(row, dict)])
    hist_test_f1s = finite_values([row.get("test_f1") for row in history if isinstance(row, dict)])
    event_test_f1s = finite_values([row.get("test_f1") for row in test_events])
    event_fns = finite_values([row.get("fn") for row in test_events])
    event_fps = finite_values([row.get("fp") for row in test_events])

    grad_maxs = finite_values([row.get("grad_norm_max") for row in history if isinstance(row, dict)])
    grad_p99s = finite_values([row.get("grad_norm_p99") for row in history if isinstance(row, dict)])
    clipped = finite_values([row.get("grad_n_clipped") for row in history if isinstance(row, dict)])
    steps = finite_values([row.get("grad_n_steps") for row in history if isinstance(row, dict)])
    clipped_ratio = (sum(clipped) / sum(steps)) if clipped and steps and sum(steps) else None

    val_f1_range = max(val_f1s) - min(val_f1s) if val_f1s else None
    peak_val_f1 = max(val_f1s) if val_f1s else None
    last_val_f1 = val_f1s[-1] if val_f1s else None
    post_peak_val_drop = post_peak_drop(val_f1s)
    test_f1_range = max(event_test_f1s) - min(event_test_f1s) if event_test_f1s else (
        max(hist_test_f1s) - min(hist_test_f1s) if hist_test_f1s else None
    )
    val_loss_range = max(val_losses) - min(val_losses) if val_losses else None
    val_loss_ratio = (max(val_losses) / max(min(val_losses), 1e-8)) if val_losses else None
    fn_range = max(event_fns) - min(event_fns) if event_fns else None
    fp_range = max(event_fps) - min(event_fps) if event_fps else None
    last_test_f1 = hist_test_f1s[-1] if hist_test_f1s else (event_test_f1s[-1] if event_test_f1s else None)
    peak_test_f1 = max([*hist_test_f1s, *event_test_f1s]) if (hist_test_f1s or event_test_f1s) else None
    dominant_test_series = hist_test_f1s if hist_test_f1s else event_test_f1s
    post_peak_test_drop = post_peak_drop(dominant_test_series)
    last_drop = peak_test_f1 - last_test_f1 if peak_test_f1 is not None and last_test_f1 is not None else None
    selected_gap = peak_test_f1 - float(best_f1) if peak_test_f1 is not None and best_f1 is not None else None

    reasons: list[str] = []
    f1_num = float(best_f1) if best_f1 is not None else None
    if f1_num is not None and f1_num <= 0.80:
        reasons.append("collapse_best_f1")
    if (fn is not None and fn >= 200) or (fp is not None and fp >= 200):
        reasons.append("collapse_best_counts")
    if any(v >= 200 for v in event_fns) or any(v >= 200 for v in event_fps):
        reasons.append("epoch_level_collapse")
    if val_f1_range is not None and val_f1_range >= 0.05:
        reasons.append("val_f1_oscillation")
    if test_f1_range is not None and test_f1_range >= 0.05:
        reasons.append("test_f1_oscillation")
    if peak_val_f1 is not None and post_peak_val_drop is not None and peak_val_f1 >= 0.999 and post_peak_val_drop >= 0.003:
        reasons.append("optimistic_val_spike")
    if peak_test_f1 is not None and post_peak_test_drop is not None and peak_test_f1 >= 0.998 and post_peak_test_drop >= 0.003:
        reasons.append("optimistic_test_spike")
    if last_drop is not None and last_drop >= 0.02:
        reasons.append("late_epoch_drift")
    if selected_gap is not None and selected_gap >= 0.02:
        reasons.append("selection_missed_peak")
    if val_loss_ratio is not None and val_loss_ratio >= 20:
        reasons.append("val_loss_spike")
    if grad_maxs and max(grad_maxs) >= 500:
        reasons.append("grad_max_spike")
    if grad_p99s and max(grad_p99s) >= 200:
        reasons.append("grad_p99_spike")
    if clipped_ratio is not None and clipped_ratio >= 0.25:
        reasons.append("heavy_grad_clipping")

    priority_reasons = {
        "collapse_best_f1",
        "collapse_best_counts",
        "epoch_level_collapse",
        "selection_missed_peak",
        "late_epoch_drift",
    }
    severe_ranges = (
        (test_f1_range is not None and test_f1_range >= 0.20)
        or (fn_range is not None and fn_range >= 200)
        or (fp_range is not None and fp_range >= 200)
        or (f1_num is not None and f1_num < 0.98)
        or (grad_p99s and max(grad_p99s) >= 800)
    )
    priority = "critical" if (priority_reasons & set(reasons) or severe_ranges) else (
        "stability_signal" if reasons else "stable_or_clean"
    )
    status = "tagged" if reasons else "stable_or_clean"
    if best_info:
        run_status = "complete"
    else:
        run_status = "in_progress"

    return {
        "tag": tag,
        "candidate": candidate,
        "axis": axis_from_candidate(candidate),
        "seed": seed,
        "run_status": run_status,
        "tag_status": status,
        "priority": priority,
        "reasons": ";".join(reasons),
        "best_f1": round(f1_num, 6) if f1_num is not None else None,
        "fn": fn,
        "fp": fp,
        "best_epoch": best_epoch,
        "val_f1_range": round(val_f1_range, 6) if val_f1_range is not None else None,
        "peak_val_f1": round(peak_val_f1, 6) if peak_val_f1 is not None else None,
        "last_val_f1": round(last_val_f1, 6) if last_val_f1 is not None else None,
        "post_peak_val_drop": round(post_peak_val_drop, 6) if post_peak_val_drop is not None else None,
        "test_f1_range": round(test_f1_range, 6) if test_f1_range is not None else None,
        "val_loss_range": round(val_loss_range, 6) if val_loss_range is not None else None,
        "val_loss_ratio": round(val_loss_ratio, 3) if val_loss_ratio is not None else None,
        "fn_range": int(fn_range) if fn_range is not None else None,
        "fp_range": int(fp_range) if fp_range is not None else None,
        "peak_test_f1": round(peak_test_f1, 6) if peak_test_f1 is not None else None,
        "last_test_f1": round(last_test_f1, 6) if last_test_f1 is not None else None,
        "post_peak_test_drop": round(post_peak_test_drop, 6) if post_peak_test_drop is not None else None,
        "last_drop": round(last_drop, 6) if last_drop is not None else None,
        "selected_gap": round(selected_gap, 6) if selected_gap is not None else None,
        "grad_max": round(max(grad_maxs), 4) if grad_maxs else None,
        "grad_p99_max": round(max(grad_p99s), 4) if grad_p99s else None,
        "clip_ratio": round(clipped_ratio, 4) if clipped_ratio is not None else None,
        "lr_backbone": config.get("lr_backbone"),
        "lr_head": config.get("lr_head"),
        "warmup_epochs": config.get("warmup_epochs"),
        "normal_ratio": config.get("normal_ratio"),
        "grad_clip": config.get("grad_clip"),
        "smooth_window": config.get("smooth_window"),
        "smooth_method": config.get("smooth_method"),
        "label_smoothing": config.get("label_smoothing"),
        "stochastic_depth_rate": config.get("stochastic_depth_rate"),
        "run_dir": str(run_dir),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["candidate"], []).append(row)
    out = []
    for candidate, items in sorted(grouped.items()):
        f1s = [r["best_f1"] for r in items if r.get("best_f1") is not None]
        tagged = [r for r in items if r.get("tag_status") == "tagged"]
        critical = [r for r in items if r.get("priority") == "critical"]
        out.append({
            "candidate": candidate,
            "axis": items[0].get("axis"),
            "n": len(items),
            "tagged": len(tagged),
            "critical": len(critical),
            "seeds": ",".join(str(r.get("seed")) for r in sorted(items, key=lambda x: -1 if x.get("seed") == 42 else (x.get("seed") or 999))),
            "f1_mean": round(mean(f1s), 6) if f1s else None,
            "f1_std": round(pstdev(f1s), 6) if len(f1s) > 1 else 0.0,
            "worst_f1": round(min(f1s), 6) if f1s else None,
            "reasons": ";".join(sorted({reason for r in tagged for reason in str(r.get("reasons", "")).split(";") if reason})),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["tag"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], agg_rows: list[dict[str, Any]]) -> None:
    tagged = [r for r in rows if r.get("tag_status") == "tagged"]
    critical = [r for r in rows if r.get("priority") == "critical"]
    lines = [
        "# Instability Case Report",
        "",
        f"- Updated: `{now_iso()}`",
        f"- Runs scanned: `{len(rows)}`",
        f"- Stability-signal runs preserved: `{len(tagged)}`",
        f"- Priority collapse/oscillation runs: `{len(critical)}`",
        "",
        "## Use In Paper",
        "",
        "- Use collapse and oscillation cases as motivation/evidence for the grad-clip, smoothing, and label-smoothing ablations.",
        "- Compare the tagged LR/warmup failures against later `gc`, `smooth`, and `regls` groups with the same seeds where available.",
        "- Do not discard collapsed runs; report them as instability boundary cases.",
        "",
        "## Candidate Summary",
        "",
        "| Candidate | Axis | N | Tagged | Critical | F1 Mean | Worst F1 | Reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in agg_rows:
        if row.get("critical", 0) or row.get("axis") in {"grad_clip", "smoothing", "label_smoothing"}:
            lines.append(
                "| {candidate} | {axis} | {n} | {tagged} | {critical} | {f1} | {worst} | {reasons} |".format(
                    candidate=row.get("candidate"),
                    axis=row.get("axis"),
                    n=row.get("n"),
                    tagged=row.get("tagged"),
                    critical=row.get("critical"),
                    f1=row.get("f1_mean", ""),
                    worst=row.get("worst_f1", ""),
                    reasons=row.get("reasons", ""),
                )
            )
    lines.extend([
        "",
        "## Priority Runs",
        "",
        "| Tag | Seed | F1 | FN | FP | Reasons | Val F1 Range | Test F1 Range | FN Range | FP Range | Last Drop | Grad Max | Grad P99 Max |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in sorted(critical, key=lambda r: (r.get("axis", ""), r.get("candidate", ""), r.get("seed") or 999)):
        lines.append(
            "| {tag} | {seed} | {f1} | {fn} | {fp} | {reasons} | {vfr} | {tfr} | {fnr} | {fpr} | {drop} | {gmax} | {gp99} |".format(
                tag=row.get("tag"),
                seed=row.get("seed", ""),
                f1=row.get("best_f1", ""),
                fn=row.get("fn", ""),
                fp=row.get("fp", ""),
                reasons=row.get("reasons", ""),
                vfr=row.get("val_f1_range", ""),
                tfr=row.get("test_f1_range", ""),
                fnr=row.get("fn_range", ""),
                fpr=row.get("fp_range", ""),
                drop=row.get("last_drop", ""),
                gmax=row.get("grad_max", ""),
                gp99=row.get("grad_p99_max", ""),
            )
        )
    lines.extend([
        "",
        "## All Preserved Stability Signals",
        "",
        "All tagged rows are saved in `validations/instability_cases.csv` and `validations/instability_cases.json`.",
        "",
        "## Next Comparison Hooks",
        "",
        "- Grad clip: compare tagged rows with later `fresh0412_v11_gc05_n700`, `gc20`, and `gc50` runs.",
        "- Smoothing: compare tagged rows with later `fresh0412_v11_sw1raw_n700`, `sw5med`, and `sw3mean` runs.",
        "- Label smoothing: compare tagged rows with later `fresh0412_v11_regls002_n700`, `regls005`, `regls01`, and `regls015` runs.",
        "- Selection stability: prioritize `selection_missed_peak`, `late_epoch_drift`, and `val_loss_spike` tags.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect instability cases from anomaly runs")
    parser.add_argument("--pattern", default="*fresh0412_v11*")
    parser.add_argument("--out-md", type=Path, default=ROOT / "validations" / "instability_cases_report.md")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "validations" / "instability_cases.csv")
    parser.add_argument("--out-json", type=Path, default=ROOT / "validations" / "instability_cases.json")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for run_dir in sorted((ROOT / "logs").glob(args.pattern), key=lambda p: p.stat().st_mtime):
        if not run_dir.is_dir():
            continue
        row = summarize_run(run_dir)
        if row:
            rows.append(row)
    agg_rows = aggregate(rows)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"updated_at": now_iso(), "rows": rows, "aggregates": agg_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if rows:
        write_csv(args.out_csv, rows)
    write_markdown(args.out_md, rows, agg_rows)
    print(json.dumps({
        "scanned": len(rows),
        "tagged": sum(1 for r in rows if r.get("tag_status") == "tagged"),
        "critical": sum(1 for r in rows if r.get("priority") == "critical"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
