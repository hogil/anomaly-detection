#!/usr/bin/env python
"""Build markdown tables and plots from flat and grouped logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def iter_run_dirs(logs_dir: Path):
    if not logs_dir.exists():
        return
    for history_path in sorted(logs_dir.glob("**/history.json")):
        run_dir = history_path.parent
        if (run_dir / "best_info.json").exists():
            yield run_dir


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_run_name(path: Path) -> tuple[str, int | None]:
    name = re.sub(r"^\d{6}_\d{6}_", "", path.name)
    name = re.sub(r"_F\d+(?:\.\d+)?_R\d+(?:\.\d+)?$", "", name)
    match = re.search(r"_s(\d+)(?:_|$)", name)
    if not match:
        return name, None
    seed = int(match.group(1))
    return name[: match.start()], seed


def binary_counts(metrics: dict[str, Any]) -> tuple[int | None, int | None]:
    normal = metrics.get("normal") or {}
    abnormal = metrics.get("abnormal") or {}
    try:
        if "fn" in normal and "fn" in abnormal:
            # Report anomaly-view counts: FN = missed abnormal, FP = normal flagged as abnormal.
            return int(round(float(abnormal["fn"]))), int(round(float(normal["fn"])))
    except (TypeError, ValueError):
        pass
    try:
        fp = int(round(float(normal.get("support", normal.get("count"))) * (1.0 - float(normal["recall"]))))
        fn = int(round(float(abnormal.get("support", abnormal.get("count"))) * (1.0 - float(abnormal["recall"]))))
        return fn, fp
    except (KeyError, TypeError, ValueError):
        return None, None


def report_metrics(best_info: dict[str, Any]) -> tuple[float | None, int | None, int | None, str]:
    selected_nt = best_info.get("selected_normal_threshold_result")
    if isinstance(selected_nt, dict):
        f1 = finite_float(selected_nt.get("f1"))
        metrics = selected_nt.get("metrics") or {}
        if isinstance(metrics, dict):
            fn, fp = binary_counts(metrics)
            return f1, fn, fp, "selected_nt"

    metrics = best_info.get("test_metrics") or {}
    fn, fp = binary_counts(metrics) if isinstance(metrics, dict) else (None, None)
    return finite_float(best_info.get("test_f1")), fn, fp, "best_info"


def collect_run(run_dir: Path) -> dict[str, Any] | None:
    history = read_json(run_dir / "history.json", [])
    if not isinstance(history, list) or not history:
        return None
    best_info = read_json(run_dir / "best_info.json", {})
    if not isinstance(best_info, dict):
        best_info = {}

    candidate, seed = parse_run_name(run_dir)
    val_f1 = [finite_float(row.get("val_f1")) for row in history]
    val_f1 = [v for v in val_f1 if v is not None]
    test_f1 = [finite_float(row.get("test_f1")) for row in history]
    test_f1 = [v for v in test_f1 if v is not None]
    val_loss = [finite_float(row.get("val_loss")) for row in history]
    val_loss = [v for v in val_loss if v is not None]
    grad_p99 = [finite_float(row.get("grad_norm_p99")) for row in history]
    grad_p99 = [v for v in grad_p99 if v is not None]
    grad_max = [finite_float(row.get("grad_norm_max")) for row in history]
    grad_max = [v for v in grad_max if v is not None]

    report_f1, fn, fp, metric_source = report_metrics(best_info)
    return {
        "candidate": candidate,
        "seed": seed,
        "run_dir": str(run_dir),
        "epochs": len(history),
        "best_epoch": best_info.get("epoch", best_info.get("best_epoch")),
        "best_info_f1": finite_float(best_info.get("test_f1")),
        "report_f1": report_f1,
        "best_val_f1": max(val_f1) if val_f1 else None,
        "best_test_f1": max(test_f1) if test_f1 else finite_float(best_info.get("test_f1")),
        "final_val_f1": val_f1[-1] if val_f1 else None,
        "final_test_f1": test_f1[-1] if test_f1 else None,
        "max_val_loss": max(val_loss) if val_loss else None,
        "max_grad_p99": max(grad_p99) if grad_p99 else None,
        "max_grad_norm": max(grad_max) if grad_max else None,
        "loss_nonfinite_samples": sum(int(row.get("loss_nonfinite_samples", 0) or 0) for row in history),
        "loss_skipped_batches": sum(int(row.get("loss_skipped_batches", 0) or 0) for row in history),
        "fn": fn,
        "fp": fp,
        "metric_source": metric_source,
        "history": history,
    }


def fmt(value: Any, digits: int = 4) -> str:
    value = finite_float(value)
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["candidate"]].append(row)

    out = []
    for candidate, items in groups.items():
        values = [x["report_f1"] for x in items if x["report_f1"] is not None]
        fns = [x["fn"] for x in items if x["fn"] is not None]
        fps = [x["fp"] for x in items if x["fp"] is not None]
        grad = [x["max_grad_p99"] for x in items if x["max_grad_p99"] is not None]
        out.append({
            "candidate": candidate,
            "seeds": len({x["seed"] for x in items if x["seed"] is not None}),
            "runs": len(items),
            "f1_mean": mean(values) if values else None,
            "f1_std": pstdev(values) if len(values) > 1 else 0.0 if values else None,
            "fn_mean": mean(fns) if fns else None,
            "fp_mean": mean(fps) if fps else None,
            "max_grad_p99_mean": mean(grad) if grad else None,
            "loss_nonfinite_samples": sum(x["loss_nonfinite_samples"] for x in items),
            "loss_skipped_batches": sum(x["loss_skipped_batches"] for x in items),
        })
    return sorted(out, key=lambda x: (x["f1_mean"] is None, -(x["f1_mean"] or 0), x["candidate"]))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def plot_path(prefix: Path, suffix: str) -> str:
    return prefix.with_name(prefix.name + suffix).name


def write_markdown(path: Path, prefix: Path, run_rows: list[dict[str, Any]], agg_rows: list[dict[str, Any]], top_k: int, no_plots: bool) -> None:
    lines = [
        "# Server Log Summary",
        "",
        f"- Runs parsed: `{len(run_rows)}`",
        f"- Candidates: `{len(agg_rows)}`",
        "- Source: `logs/**/history.json` plus `best_info.json`.",
        "- Metric: `selected_normal_threshold_result` when present, otherwise `best_info.test_f1`.",
        "",
    ]
    if not no_plots:
        lines.extend([
            "## Live Plots",
            "",
            f"![candidate F1]({plot_path(prefix, '_candidate_f1.png')})",
            "",
            f"![FN FP]({plot_path(prefix, '_fn_fp.png')})",
            "",
            f"![val F1 curves]({plot_path(prefix, '_val_f1_curves.png')})",
            "",
            f"![grad p99 curves]({plot_path(prefix, '_grad_p99_curves.png')})",
            "",
        ])
    lines.extend([
        "## Candidate Summary",
        "",
        "| candidate | runs | seeds | F1 mean | std | FN mean | FP mean | max grad p99 mean | nonfinite loss samples | skipped batches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in agg_rows[:top_k]:
        lines.append(
            "| {candidate} | {runs} | {seeds} | {f1_mean} | {f1_std} | {fn_mean} | {fp_mean} | {grad} | {bad} | {skip} |".format(
                candidate=row["candidate"],
                runs=row["runs"],
                seeds=row["seeds"],
                f1_mean=fmt(row["f1_mean"]),
                f1_std=fmt(row["f1_std"]),
                fn_mean=fmt(row["fn_mean"], 2),
                fp_mean=fmt(row["fp_mean"], 2),
                grad=fmt(row["max_grad_p99_mean"], 2),
                bad=row["loss_nonfinite_samples"],
                skip=row["loss_skipped_batches"],
            )
        )

    lines.extend([
        "",
        "## Per-Run Summary",
        "",
        "| candidate | seed | epochs | F1 | FN | FP | metric source | final val F1 | max val loss | max grad p99 | run dir |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ])
    for row in sorted(run_rows, key=lambda x: (x["candidate"], x["seed"] if x["seed"] is not None else 9999))[:top_k * 5]:
        lines.append(
            "| {candidate} | {seed} | {epochs} | {f1} | {fn} | {fp} | {metric_source} | {final_val} | {val_loss} | {grad} | `{run_dir}` |".format(
                candidate=row["candidate"],
                seed="" if row["seed"] is None else row["seed"],
                epochs=row["epochs"],
                f1=fmt(row["report_f1"]),
                fn="" if row["fn"] is None else row["fn"],
                fp="" if row["fp"] is None else row["fp"],
                metric_source=row["metric_source"],
                final_val=fmt(row["final_val_f1"]),
                val_loss=fmt(row["max_val_loss"]),
                grad=fmt(row["max_grad_p99"], 2),
                run_dir=row["run_dir"],
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_reports(prefix: Path, run_rows: list[dict[str, Any]], agg_rows: list[dict[str, Any]], top_k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = agg_rows[:top_k]
    if top:
        labels = [row["candidate"] for row in top]
        values = [row["f1_mean"] or 0.0 for row in top]
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 5))
        ax.bar(range(len(labels)), values)
        ax.set_ylabel("mean best test F1")
        ax.set_ylim(max(0.0, min(values) - 0.01), min(1.0, max(values) + 0.005))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
        fig.tight_layout()
        fig.savefig(prefix.with_name(prefix.name + "_candidate_f1.png"), dpi=160)
        plt.close(fig)

        fn_values = [row["fn_mean"] or 0.0 for row in top]
        fp_values = [row["fp_mean"] or 0.0 for row in top]
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 5))
        xs = list(range(len(labels)))
        ax.bar([x - 0.2 for x in xs], fn_values, width=0.4, label="FN mean")
        ax.bar([x + 0.2 for x in xs], fp_values, width=0.4, label="FP mean")
        ax.set_ylabel("mean count")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
        ax.legend()
        fig.tight_layout()
        fig.savefig(prefix.with_name(prefix.name + "_fn_fp.png"), dpi=160)
        plt.close(fig)

    selected = sorted(run_rows, key=lambda x: (x["report_f1"] is None, -(x["report_f1"] or 0)))[:top_k]
    if selected:
        fig, ax = plt.subplots(figsize=(10, 5))
        for row in selected:
            xs = [item.get("epoch") for item in row["history"] if item.get("val_f1") is not None]
            ys = [item.get("val_f1") for item in row["history"] if item.get("val_f1") is not None]
            ax.plot(xs, ys, linewidth=1.4, alpha=0.8, label=f"{row['candidate']} s{row['seed']}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val F1")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(prefix.with_name(prefix.name + "_val_f1_curves.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for row in selected:
            xs = [item.get("epoch") for item in row["history"] if item.get("grad_norm_p99") is not None]
            ys = [item.get("grad_norm_p99") for item in row["history"] if item.get("grad_norm_p99") is not None]
            if xs:
                ax.plot(xs, ys, linewidth=1.2, alpha=0.8, label=f"{row['candidate']} s{row['seed']}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("grad norm p99")
        ax.set_yscale("symlog")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(prefix.with_name(prefix.name + "_grad_p99_curves.png"), dpi=160)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--out-prefix", type=Path, default=ROOT / "validations/log_history_report")
    parser.add_argument("--contains", default="")
    parser.add_argument("--candidate-prefix", default="")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for run_dir in iter_run_dirs(args.logs_dir):
        rel_name = str(run_dir.relative_to(args.logs_dir)).replace("\\", "/")
        if args.contains and args.contains not in rel_name:
            continue
        row = collect_run(run_dir)
        if row is None:
            continue
        if args.candidate_prefix and not row["candidate"].startswith(args.candidate_prefix):
            continue
        rows.append(row)

    agg_rows = aggregate(rows)
    run_fields = [
        "candidate", "seed", "epochs", "best_epoch", "best_info_f1", "report_f1", "best_val_f1", "best_test_f1",
        "final_val_f1", "final_test_f1", "max_val_loss", "max_grad_p99", "max_grad_norm",
        "loss_nonfinite_samples", "loss_skipped_batches", "fn", "fp", "metric_source", "run_dir",
    ]
    agg_fields = [
        "candidate", "runs", "seeds", "f1_mean", "f1_std", "fn_mean", "fp_mean",
        "max_grad_p99_mean", "loss_nonfinite_samples", "loss_skipped_batches",
    ]
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_runs.csv"), rows, run_fields)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_candidates.csv"), agg_rows, agg_fields)
    write_markdown(args.out_prefix.with_suffix(".md"), args.out_prefix, rows, agg_rows, args.top_k, args.no_plots)
    if not args.no_plots:
        plot_reports(args.out_prefix, rows, agg_rows, args.top_k)
    print(f"[history-report] runs={len(rows)} candidates={len(agg_rows)} out={args.out_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
