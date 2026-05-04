#!/usr/bin/env python
"""Build a comparison markdown + bar plot from a controller results JSON.

Used by stages whose `*_results.json` is a single-axis comparison (backbone
sweep, BKM combined, sample-skip probe). Pulls the rawbase baseline from
`validations/01_baseline_results.json` so every plot/table shares the same
reference point. No hard-coded candidate names.

Output:
  <results_path>.md            -- comparison table (overwrites controller's live markdown)
  <plot_path>                  -- candidate bar chart against baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else {}


def aggregate_candidates(summary: dict[str, Any]) -> list[dict[str, Any]]:
    by_candidate = summary.get("aggregates", {}).get("by_candidate")
    if isinstance(by_candidate, dict) and by_candidate:
        rows = []
        for name, row in by_candidate.items():
            rows.append({
                "candidate": name,
                "complete": int(row.get("complete", 0)),
                "f1": row.get("f1_mean"),
                "fn": row.get("fn_mean"),
                "fp": row.get("fp_mean"),
            })
        return rows
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in summary.get("runs", {}).values():
        if run.get("status") != "complete":
            continue
        cand = run.get("candidate") or run.get("tag", "").rsplit("_s", 1)[0]
        grouped.setdefault(cand, []).append(run)
    rows = []
    for cand, items in grouped.items():
        f1s = [r.get("test_f1") for r in items if r.get("test_f1") is not None]
        fns = [r.get("fn") for r in items if r.get("fn") is not None]
        fps = [r.get("fp") for r in items if r.get("fp") is not None]
        rows.append({
            "candidate": cand,
            "complete": len(items),
            "f1": mean(f1s) if f1s else None,
            "fn": mean(fns) if fns else None,
            "fp": mean(fps) if fps else None,
        })
    return rows


def baseline_metrics(baseline_summary: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    rows = aggregate_candidates(baseline_summary)
    if not rows:
        return (None, None, None)
    row = max(rows, key=lambda r: r.get("complete", 0))
    return (row.get("f1"), row.get("fn"), row.get("fp"))


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_delta(value: float | None, base: float | None, digits: int = 4) -> str:
    if value is None or base is None:
        return "-"
    diff = value - base
    if abs(diff) < 1e-9:
        return "0"
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.{digits}f}".rstrip("0").rstrip(".")


def write_markdown(out: Path, title: str, rows: list[dict[str, Any]],
                   base: tuple[float | None, float | None, float | None], plot_rel: str) -> None:
    base_f1, base_fn, base_fp = base
    lines = [
        f"# {title}",
        "",
        f"![{title}]({plot_rel})",
        "",
        "Baseline = `01_baseline_results.json` 의 평균.",
        "",
        "| candidate | seeds | F1 | ΔF1 | FN | ΔFN | FP | ΔFP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if base_f1 is not None:
        lines.append(
            f"| **baseline** | - | {fmt(base_f1)} | 0 | {fmt(base_fn, 3)} | 0 | {fmt(base_fp, 3)} | 0 |"
        )
    rows_sorted = sorted(rows, key=lambda r: (r.get("f1") is None, -(r.get("f1") or 0.0)))
    for row in rows_sorted:
        lines.append(
            f"| `{row['candidate']}` | {row['complete']}/5 | "
            f"{fmt(row['f1'])} | {fmt_delta(row['f1'], base_f1)} | "
            f"{fmt(row['fn'], 3)} | {fmt_delta(row['fn'], base_fn, 3)} | "
            f"{fmt(row['fp'], 3)} | {fmt_delta(row['fp'], base_fp, 3)} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(plot_path: Path, title: str, rows: list[dict[str, Any]],
               base: tuple[float | None, float | None, float | None]) -> None:
    base_f1, base_fn, base_fp = base
    rows_with_f1 = [r for r in rows if r.get("f1") is not None]
    if not rows_with_f1:
        return
    rows_sorted = sorted(rows_with_f1, key=lambda r: -r["f1"])
    labels = [r["candidate"].replace("fresh0412_v11_rawbase_", "").replace("fresh0412_v11_", "") for r in rows_sorted]
    f1s = [r["f1"] for r in rows_sorted]
    fns = [r["fn"] if r.get("fn") is not None else 0 for r in rows_sorted]
    fps = [r["fp"] if r.get("fp") is not None else 0 for r in rows_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(max(8.0, 0.7 * len(labels) + 6), 5.0))
    ax_f1, ax_err = axes
    bar_color = "#4878CF"
    ax_f1.bar(range(len(labels)), f1s, color=bar_color)
    if base_f1 is not None:
        ax_f1.axhline(base_f1, color="#888888", linestyle="--", linewidth=1, label=f"baseline F1={base_f1:.4f}")
        ax_f1.legend(fontsize=8)
    ax_f1.set_xticks(range(len(labels)))
    ax_f1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_f1.set_ylabel("F1")
    ax_f1.set_title(f"{title} — F1")
    if f1s:
        f1_min = min(f1s + ([base_f1] if base_f1 is not None else []))
        f1_max = max(f1s + ([base_f1] if base_f1 is not None else []))
        margin = max(0.0005, (f1_max - f1_min) * 0.2)
        ax_f1.set_ylim(max(0.0, f1_min - margin), min(1.0, f1_max + margin))

    width = 0.4
    x = list(range(len(labels)))
    ax_err.bar([i - width / 2 for i in x], fns, width=width, label="FN", color="#E43320")
    ax_err.bar([i + width / 2 for i in x], fps, width=width, label="FP", color="#F5B041")
    if base_fn is not None:
        ax_err.axhline(base_fn, color="#E43320", linestyle="--", linewidth=1, alpha=0.5)
    if base_fp is not None:
        ax_err.axhline(base_fp, color="#F5B041", linestyle="--", linewidth=1, alpha=0.5)
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_err.set_ylabel("count (mean over seeds)")
    ax_err.set_title(f"{title} — FN / FP (dashed = baseline)")
    ax_err.legend(fontsize=8)

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, type=Path, help="stage results JSON")
    p.add_argument("--baseline", type=Path, default=Path("validations/01_baseline_results.json"))
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--out-plot", type=Path, required=True)
    p.add_argument("--title", required=True)
    args = p.parse_args()

    summary = load_json(args.results)
    rows = aggregate_candidates(summary)
    base = baseline_metrics(load_json(args.baseline))

    if not rows:
        print(f"[stage_comparison] no completed candidates in {args.results}; skipping")
        return 0

    plot_rel = args.out_plot.name if args.out_plot.parent == args.out_md.parent \
        else str(args.out_plot.relative_to(args.out_md.parent))
    write_markdown(args.out_md, args.title, rows, base, plot_rel)
    write_plot(args.out_plot, args.title, rows, base)
    print(f"[stage_comparison] wrote {args.out_md} and {args.out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
