#!/usr/bin/env python
"""Cross-dataset comparison report.

For each ``--groups`` entry of the form ``<group>=<config.yaml>``, reads
results from ``validations/<group>/``:

- ``01_baseline_results.json``: baseline candidate
- ``04_backbone_results.json``: backbone leaderboard
- ``05_bkm_combined_results.json``: BKM-combined candidate

Each ``runs`` entry has seed-level ``test_f1`` / ``fn`` / ``fp``. We aggregate
by ``candidate`` (mean across complete seeds).

Outputs in ``--out-dir``:

- ``cross_dataset_summary.md``
- ``cross_dataset_summary.csv``
- ``cross_dataset_overall.csv``: mean F1/FN/FP across available datasets
- ``cross_dataset_f1.png``: baseline vs BKM-combined F1 per dataset
- ``cross_dataset_backbone.png``: best backbone F1 per dataset
- ``cross_dataset_overall.png``: overall mean F1 and FN/FP by stage
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_groups(raw: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"--groups entry missing '=': {chunk!r}")
        name, cfg = chunk.split("=", 1)
        items.append((name.strip(), cfg.strip()))
    if not items:
        raise SystemExit("--groups produced no entries")
    return items


def load_runs(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def aggregate_by_candidate(payload: dict) -> dict[str, dict]:
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, dict):
        return {}

    bucket: dict[str, list[dict]] = defaultdict(list)
    for entry in runs.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "complete":
            continue
        cand = entry.get("candidate")
        if not cand:
            continue
        bucket[cand].append(entry)

    out: dict[str, dict] = {}
    for cand, items in bucket.items():
        f1s = [float(x["test_f1"]) for x in items if x.get("test_f1") is not None]
        fns = [float(x["fn"]) for x in items if x.get("fn") is not None]
        fps = [float(x["fp"]) for x in items if x.get("fp") is not None]
        if not f1s:
            continue
        out[cand] = {
            "candidate": cand,
            "seeds": len(items),
            "f1": mean(f1s),
            "fn": mean(fns) if fns else float("nan"),
            "fp": mean(fps) if fps else float("nan"),
        }
    return out


def best_candidate(cands: dict[str, dict]) -> dict | None:
    if not cands:
        return None
    return max(cands.values(), key=lambda d: d["f1"])


def fmt(value, spec=".4f"):
    if value is None:
        return "-"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "-"


def values(rows: list[dict], field: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def build_overall_rows(rows: list[dict]) -> list[dict]:
    stages = [
        ("baseline", "Baseline"),
        ("bkm", "BKM combined"),
        ("backbone", "Best backbone"),
    ]
    out = []
    for prefix, label in stages:
        f1s = values(rows, f"{prefix}_f1")
        fns = values(rows, f"{prefix}_fn")
        fps = values(rows, f"{prefix}_fp")
        seeds = [int(r.get(f"{prefix}_seeds") or 0) for r in rows]
        out.append(
            {
                "stage": label,
                "datasets": len(f1s),
                "f1_mean": mean(f1s) if f1s else None,
                "fn_mean": mean(fns) if fns else None,
                "fp_mean": mean(fps) if fps else None,
                "total_complete_seeds": sum(seeds),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        required=True,
        help="Semicolon-separated <group>=<config.yaml> entries",
    )
    parser.add_argument(
        "--validations-root",
        type=Path,
        default=Path("validations"),
        help="Validations root (default: validations)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for the cross-dataset report",
    )
    args = parser.parse_args()

    groups = parse_groups(args.groups)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for group, cfg in groups:
        gdir = args.validations_root / group
        baseline_path = gdir / "01_baseline_results.json"
        backbone_path = gdir / "04_backbone_results.json"
        bkm_path = gdir / "05_bkm_combined_results.json"

        baseline_cands = aggregate_by_candidate(load_runs(baseline_path))
        backbone_cands = aggregate_by_candidate(load_runs(backbone_path))
        bkm_cands = aggregate_by_candidate(load_runs(bkm_path))

        baseline_best = best_candidate(baseline_cands)
        backbone_best = best_candidate(backbone_cands)
        bkm_best = best_candidate(bkm_cands)

        rows.append(
            {
                "group": group,
                "config": cfg,
                "baseline_candidate": baseline_best["candidate"] if baseline_best else "",
                "baseline_f1": baseline_best["f1"] if baseline_best else None,
                "baseline_fn": baseline_best["fn"] if baseline_best else None,
                "baseline_fp": baseline_best["fp"] if baseline_best else None,
                "baseline_seeds": baseline_best["seeds"] if baseline_best else 0,
                "bkm_candidate": bkm_best["candidate"] if bkm_best else "",
                "bkm_f1": bkm_best["f1"] if bkm_best else None,
                "bkm_fn": bkm_best["fn"] if bkm_best else None,
                "bkm_fp": bkm_best["fp"] if bkm_best else None,
                "bkm_seeds": bkm_best["seeds"] if bkm_best else 0,
                "backbone_candidate": backbone_best["candidate"] if backbone_best else "",
                "backbone_f1": backbone_best["f1"] if backbone_best else None,
                "backbone_fn": backbone_best["fn"] if backbone_best else None,
                "backbone_fp": backbone_best["fp"] if backbone_best else None,
                "backbone_seeds": backbone_best["seeds"] if backbone_best else 0,
            }
        )

    overall_rows = build_overall_rows(rows)

    csv_path = args.out_dir / "cross_dataset_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    overall_csv_path = args.out_dir / "cross_dataset_overall.csv"
    with overall_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage", "datasets", "f1_mean", "fn_mean", "fp_mean", "total_complete_seeds"],
        )
        writer.writeheader()
        for row in overall_rows:
            writer.writerow(row)

    md_lines = [
        "# Cross-dataset comparison",
        "",
        "Per-dataset best candidate within each stage. Seed-level F1/FN/FP averaged across complete seeds.",
        "",
        "| dataset (config) | group | baseline F1 | baseline FN | baseline FP | BKM F1 | BKM FN | BKM FP | best backbone F1 | best backbone candidate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    r["config"],
                    r["group"],
                    fmt(r["baseline_f1"]),
                    fmt(r["baseline_fn"], ".2f"),
                    fmt(r["baseline_fp"], ".2f"),
                    fmt(r["bkm_f1"]),
                    fmt(r["bkm_fn"], ".2f"),
                    fmt(r["bkm_fp"], ".2f"),
                    fmt(r["backbone_f1"]),
                    r["backbone_candidate"] or "-",
                ]
            )
            + " |"
        )

    md_lines.extend(
        [
            "",
            "## Overall summary",
            "",
            "Mean across datasets with available complete candidate results.",
            "",
            "| stage | datasets | complete seeds | F1 mean | FN mean | FP mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in overall_rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    row["stage"],
                    str(row["datasets"]),
                    str(row["total_complete_seeds"]),
                    fmt(row["f1_mean"]),
                    fmt(row["fn_mean"], ".2f"),
                    fmt(row["fp_mean"], ".2f"),
                ]
            )
            + " |"
        )

    md_path = args.out_dir / "cross_dataset_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Plot 1: baseline vs BKM combined F1 per dataset.
    labels = [Path(r["config"]).stem for r in rows]
    base_vals = [r["baseline_f1"] if r["baseline_f1"] is not None else 0.0 for r in rows]
    bkm_vals = [r["bkm_f1"] if r["bkm_f1"] is not None else 0.0 for r in rows]

    if labels:
        x = range(len(labels))
        width = 0.4
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4.2))
        ax.bar([i - width / 2 for i in x], base_vals, width=width, label="baseline")
        ax.bar([i + width / 2 for i in x], bkm_vals, width=width, label="BKM combined")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("F1")
        ax.set_title("Cross-dataset: baseline vs BKM combined F1")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for i, (b, k) in enumerate(zip(base_vals, bkm_vals)):
            if b:
                ax.text(i - width / 2, b, f"{b:.4f}", ha="center", va="bottom", fontsize=8)
            if k:
                ax.text(i + width / 2, k, f"{k:.4f}", ha="center", va="bottom", fontsize=8)
        ymin = min([v for v in (base_vals + bkm_vals) if v] or [0.99]) - 0.005
        ax.set_ylim(max(0, ymin), 1.0)
        fig.tight_layout()
        fig.savefig(args.out_dir / "cross_dataset_f1.png", dpi=150)
        plt.close(fig)

        # Plot 2: best backbone F1 per dataset.
        bb_vals = [r["backbone_f1"] if r["backbone_f1"] is not None else 0.0 for r in rows]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4.2))
        ax.bar(list(x), bb_vals, color="#5b8def")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("F1")
        ax.set_title("Cross-dataset: best backbone F1")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for i, v in enumerate(bb_vals):
            if v:
                ax.text(i, v, f"{v:.4f}\n{rows[i]['backbone_candidate']}", ha="center", va="bottom", fontsize=7)
        ymin = min([v for v in bb_vals if v] or [0.99]) - 0.005
        ax.set_ylim(max(0, ymin), 1.0)
        fig.tight_layout()
        fig.savefig(args.out_dir / "cross_dataset_backbone.png", dpi=150)
        plt.close(fig)

        overall_with_f1 = [row for row in overall_rows if row["f1_mean"] is not None]
        if overall_with_f1:
            stage_labels = [row["stage"] for row in overall_with_f1]
            f1_vals = [row["f1_mean"] for row in overall_with_f1]
            fn_vals = [row["fn_mean"] if row["fn_mean"] is not None else 0.0 for row in overall_with_f1]
            fp_vals = [row["fp_mean"] if row["fp_mean"] is not None else 0.0 for row in overall_with_f1]
            x = list(range(len(stage_labels)))
            fig, axes = plt.subplots(1, 2, figsize=(max(7, len(stage_labels) * 1.8), 4.2))
            ax_f1, ax_err = axes
            ax_f1.bar(x, f1_vals, color="#4878CF")
            ax_f1.set_xticks(x)
            ax_f1.set_xticklabels(stage_labels, rotation=20, ha="right")
            ax_f1.set_ylabel("F1 mean")
            ax_f1.set_title("Overall F1")
            ax_f1.grid(axis="y", linestyle="--", alpha=0.4)
            for i, value in enumerate(f1_vals):
                ax_f1.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
            f1_min = min(f1_vals)
            f1_max = max(f1_vals)
            margin = max(0.0005, (f1_max - f1_min) * 0.2)
            ax_f1.set_ylim(max(0.0, f1_min - margin), min(1.0, f1_max + margin))

            width = 0.38
            ax_err.bar([i - width / 2 for i in x], fn_vals, width=width, label="FN", color="#E43320")
            ax_err.bar([i + width / 2 for i in x], fp_vals, width=width, label="FP", color="#F5B041")
            ax_err.set_xticks(x)
            ax_err.set_xticklabels(stage_labels, rotation=20, ha="right")
            ax_err.set_ylabel("mean count")
            ax_err.set_title("Overall FN / FP")
            ax_err.grid(axis="y", linestyle="--", alpha=0.4)
            ax_err.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(args.out_dir / "cross_dataset_overall.png", dpi=150)
            plt.close(fig)

    print(f"[cross_dataset_report] datasets={len(rows)}")
    print(f"  md  : {md_path}")
    print(f"  csv : {csv_path}")
    print(f"        {overall_csv_path}")
    if labels:
        print(f"  png : {args.out_dir / 'cross_dataset_f1.png'}")
        print(f"        {args.out_dir / 'cross_dataset_backbone.png'}")
        if any(row["f1_mean"] is not None for row in overall_rows):
            print(f"        {args.out_dir / 'cross_dataset_overall.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
