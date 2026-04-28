from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR_RE = re.compile(
    r"^(?P<date>\d{6})_(?P<time>\d{6})_(?P<candidate>.+)_s(?P<seed>\d+)"
    r"(?:_F(?P<f1>\d+(?:\.\d+)?)_R(?P<recall>\d+(?:\.\d+)?))?$"
)


@dataclass(frozen=True)
class RunRecord:
    name: str
    path: Path
    started_at: datetime
    candidate: str
    seed: int
    complete: bool
    f1: float | None
    recall: float | None
    fn: int | None
    fp: int | None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize train.py log folders into a text table and plots. "
            "F1/recall are parsed from folder names; FN/FP are added from best_info.json when available."
        )
    )
    ap.add_argument("--logs-dir", default=str(ROOT / "logs"), help="Directory containing run folders")
    ap.add_argument("--out-prefix", default=str(ROOT / "validations" / "log_folder_summary"))
    ap.add_argument("--candidate-prefix", default="", help="Only include candidates starting with this prefix")
    ap.add_argument("--contains", default="", help="Only include candidates containing this substring")
    ap.add_argument("--expected-seeds", type=int, default=5, help="Seed count used for completion display")
    ap.add_argument("--top-k", type=int, default=80, help="Maximum candidate rows in plots")
    ap.add_argument("--no-best-info", action="store_true", help="Do not read best_info.json for FN/FP")
    ap.add_argument("--no-plots", action="store_true", help="Write tables only")
    return ap.parse_args()


def parse_started(date_part: str, time_part: str) -> datetime:
    return datetime.strptime(date_part + time_part, "%y%m%d%H%M%S")


def metric_from_best_info(path: Path) -> tuple[int | None, int | None]:
    best_path = path / "best_info.json"
    if not best_path.exists():
        return None, None
    try:
        best = json.loads(best_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    tm = best.get("test_metrics") or {}
    normal = tm.get("normal") or {}
    abnormal = tm.get("abnormal") or {}
    nc = normal.get("count")
    nr = normal.get("recall")
    ac = abnormal.get("count")
    ar = abnormal.get("recall")
    if None in (nc, nr, ac, ar):
        return None, None
    fn = int(round(float(ac) * (1.0 - float(ar))))
    fp = int(round(float(nc) * (1.0 - float(nr))))
    return fn, fp


def parse_run_dir(path: Path, *, read_best_info: bool) -> RunRecord | None:
    m = RUN_DIR_RE.match(path.name)
    if not m:
        return None
    f1 = float(m.group("f1")) if m.group("f1") is not None else None
    recall = float(m.group("recall")) if m.group("recall") is not None else None
    fn, fp = metric_from_best_info(path) if read_best_info else (None, None)
    return RunRecord(
        name=path.name,
        path=path,
        started_at=parse_started(m.group("date"), m.group("time")),
        candidate=m.group("candidate"),
        seed=int(m.group("seed")),
        complete=f1 is not None and recall is not None,
        f1=f1,
        recall=recall,
        fn=fn,
        fp=fp,
    )


def collect_runs(
    logs_dir: Path,
    *,
    candidate_prefix: str,
    contains: str,
    read_best_info: bool,
) -> list[RunRecord]:
    rows: list[RunRecord] = []
    for path in sorted(logs_dir.iterdir()):
        if not path.is_dir():
            continue
        rec = parse_run_dir(path, read_best_info=read_best_info)
        if rec is None:
            continue
        if candidate_prefix and not rec.candidate.startswith(candidate_prefix):
            continue
        if contains and contains not in rec.candidate:
            continue
        rows.append(rec)
    rows.sort(key=lambda r: (r.started_at, r.candidate, r.seed))
    return rows


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_mean(values: list[float], digits: int = 4) -> str:
    if not values:
        return ""
    return f"{mean(values):.{digits}f}"


def fmt_std(values: list[float], digits: int = 4) -> str:
    if len(values) < 2:
        return ""
    return f"{pstdev(values):.{digits}f}"


def summarize_candidates(rows: list[RunRecord], expected_seeds: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for row in rows:
        grouped[row.candidate].append(row)

    summary: list[dict[str, Any]] = []
    for candidate, cand_rows in grouped.items():
        complete_rows = [r for r in cand_rows if r.complete]
        f1s = [r.f1 for r in complete_rows if r.f1 is not None]
        recalls = [r.recall for r in complete_rows if r.recall is not None]
        fns = [r.fn for r in complete_rows if r.fn is not None]
        fps = [r.fp for r in complete_rows if r.fp is not None]
        seeds = sorted({r.seed for r in cand_rows})
        latest = max(r.started_at for r in cand_rows)
        best_run = max(complete_rows, key=lambda r: r.f1 or -1.0) if complete_rows else None
        summary.append(
            {
                "candidate": candidate,
                "runs": len(cand_rows),
                "complete": len(complete_rows),
                "expected": expected_seeds,
                "seeds": ",".join(str(s) for s in seeds),
                "f1_mean": mean(f1s) if f1s else None,
                "f1_std": pstdev(f1s) if len(f1s) >= 2 else None,
                "f1_min": min(f1s) if f1s else None,
                "f1_max": max(f1s) if f1s else None,
                "recall_mean": mean(recalls) if recalls else None,
                "fn_mean": mean(fns) if fns else None,
                "fp_mean": mean(fps) if fps else None,
                "best_seed": best_run.seed if best_run else None,
                "best_f1": best_run.f1 if best_run else None,
                "latest": latest.isoformat(sep=" ", timespec="seconds"),
            }
        )
    summary.sort(key=lambda r: (-(r["f1_mean"] or -1.0), r["candidate"]))
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(headers: list[str], body: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return lines


def write_text_report(
    path: Path,
    *,
    rows: list[RunRecord],
    summary: list[dict[str, Any]],
    logs_dir: Path,
    candidate_prefix: str,
    contains: str,
    expected_seeds: int,
) -> None:
    complete = [r for r in rows if r.complete]
    incomplete = [r for r in rows if not r.complete]
    lines: list[str] = [
        "# Log Folder Experiment Summary",
        "",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Source: `{logs_dir}`",
        f"- Candidate prefix: `{candidate_prefix or '(none)'}`",
        f"- Contains filter: `{contains or '(none)'}`",
        f"- Parsed runs: `{len(rows)}` (`{len(complete)}` complete, `{len(incomplete)}` incomplete)",
        f"- Expected seeds per candidate: `{expected_seeds}`",
        "- F1/Recall are parsed from folder names. FN/FP are filled from `best_info.json` when present.",
        "",
        "## Candidate Summary",
        "",
    ]
    lines.extend(
        markdown_table(
            ["candidate", "complete", "seeds", "F1 mean", "F1 std", "F1 min", "F1 max", "R mean", "FN mean", "FP mean", "best seed", "latest"],
            [
                [
                    str(row["candidate"]),
                    f"{row['complete']}/{row['expected']}",
                    str(row["seeds"]),
                    fmt_float(row["f1_mean"]),
                    fmt_float(row["f1_std"]),
                    fmt_float(row["f1_min"]),
                    fmt_float(row["f1_max"]),
                    fmt_float(row["recall_mean"]),
                    fmt_float(row["fn_mean"], 1),
                    fmt_float(row["fp_mean"], 1),
                    "" if row["best_seed"] is None else str(row["best_seed"]),
                    str(row["latest"]),
                ]
                for row in summary
            ],
        )
    )
    lines.extend(["", "## Completed Runs", ""])
    complete_sorted = sorted(complete, key=lambda r: (-(r.f1 or -1.0), r.candidate, r.seed))
    lines.extend(
        markdown_table(
            ["candidate", "seed", "F1", "Recall", "FN", "FP", "folder"],
            [
                [
                    r.candidate,
                    str(r.seed),
                    fmt_float(r.f1),
                    fmt_float(r.recall),
                    "" if r.fn is None else str(r.fn),
                    "" if r.fp is None else str(r.fp),
                    r.name,
                ]
                for r in complete_sorted
            ],
        )
    )
    if incomplete:
        lines.extend(["", "## Incomplete Runs", ""])
        lines.extend(
            markdown_table(
                ["candidate", "seed", "started", "folder"],
                [[r.candidate, str(r.seed), r.started_at.isoformat(sep=" ", timespec="seconds"), r.name] for r in incomplete],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_f1(summary: list[dict[str, Any]], out_path: Path, top_k: int) -> None:
    rows = [r for r in summary if r["f1_mean"] is not None][:top_k]
    if not rows:
        return
    rows = list(reversed(rows))
    labels = [r["candidate"] for r in rows]
    values = [r["f1_mean"] for r in rows]
    fig_h = max(5.0, min(28.0, 0.32 * len(rows) + 1.5))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh(range(len(rows)), values, color="#4C78A8")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("mean test F1 from folder name")
    ax.set_title("Candidate Mean F1")
    low = min(values)
    ax.set_xlim(max(0.0, low - 0.01), min(1.0005, max(values) + 0.002))
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_timeline(rows: list[RunRecord], out_path: Path) -> None:
    complete = [r for r in rows if r.complete and r.f1 is not None]
    if not complete:
        return
    complete.sort(key=lambda r: r.started_at)
    xs = [r.started_at for r in complete]
    ys = [r.f1 for r in complete]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs, ys, color="#4C78A8", marker="o", ms=3, lw=1.0)
    ax.set_xlabel("run start time")
    ax.set_ylabel("test F1 from folder name")
    ax.set_title("Completed Run Timeline")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_errors(summary: list[dict[str, Any]], out_path: Path, top_k: int) -> None:
    rows = [r for r in summary if r["fn_mean"] is not None and r["fp_mean"] is not None][:top_k]
    if not rows:
        return
    rows = list(reversed(rows))
    labels = [r["candidate"] for r in rows]
    y = list(range(len(rows)))
    fn_vals = [r["fn_mean"] for r in rows]
    fp_vals = [r["fp_mean"] for r in rows]
    fig_h = max(5.0, min(28.0, 0.32 * len(rows) + 1.5))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh([i + 0.18 for i in y], fn_vals, height=0.34, color="#E15759", label="FN mean")
    ax.barh([i - 0.18 for i in y], fp_vals, height=0.34, color="#59A14F", label="FP mean")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("mean count")
    ax.set_title("Candidate FN/FP Means")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    out_prefix = Path(args.out_prefix)
    rows = collect_runs(
        logs_dir,
        candidate_prefix=args.candidate_prefix,
        contains=args.contains,
        read_best_info=not args.no_best_info,
    )
    summary = summarize_candidates(rows, args.expected_seeds)

    summary_fields = [
        "candidate", "runs", "complete", "expected", "seeds", "f1_mean", "f1_std",
        "f1_min", "f1_max", "recall_mean", "fn_mean", "fp_mean", "best_seed", "best_f1", "latest",
    ]
    run_fields = ["name", "candidate", "seed", "complete", "f1", "recall", "fn", "fp", "started_at", "path"]
    run_dicts = [
        {
            "name": r.name,
            "candidate": r.candidate,
            "seed": r.seed,
            "complete": r.complete,
            "f1": r.f1,
            "recall": r.recall,
            "fn": r.fn,
            "fp": r.fp,
            "started_at": r.started_at.isoformat(sep=" ", timespec="seconds"),
            "path": str(r.path),
        }
        for r in rows
    ]

    write_text_report(
        out_prefix.with_suffix(".txt"),
        rows=rows,
        summary=summary,
        logs_dir=logs_dir,
        candidate_prefix=args.candidate_prefix,
        contains=args.contains,
        expected_seeds=args.expected_seeds,
    )
    write_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), summary, summary_fields)
    write_csv(out_prefix.with_name(out_prefix.name + "_runs.csv"), run_dicts, run_fields)

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment-dependent fallback
            print(f"[log-summary] plot skipped: matplotlib.pyplot import failed: {exc}")
        else:
            globals()["plt"] = plt
            plot_f1(summary, out_prefix.with_name(out_prefix.name + "_f1.png"), args.top_k)
            plot_timeline(rows, out_prefix.with_name(out_prefix.name + "_timeline.png"))
            plot_errors(summary, out_prefix.with_name(out_prefix.name + "_errors.png"), args.top_k)

    print(f"[log-summary] parsed {len(rows)} runs, {len(summary)} candidates")
    print(f"[log-summary] wrote {out_prefix.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
