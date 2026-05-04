#!/usr/bin/env python
"""Build a performance markdown + plots from a logs/<group>/ folder.

CLI:
  python scripts/generate_group_report.py --group-dir logs/run_20260430_120000

Walks every <group>/<run>/ that contains best_info.json, builds:
  - candidate aggregate table (mean F1, FN, FP, completed seeds)
  - per-run table (tag, seed, F1, FN, FP, best epoch)
  - F1 bar plot per candidate
  - val_f1 curves per candidate

Outputs (default, beside the group folder):
  <group_dir>/group_report.md
  <group_dir>/group_report_f1.png
  <group_dir>/group_report_val_f1_curves.png
Override paths with --out-md / --out-f1-plot / --out-val-curves.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_DIR_RE = re.compile(r"^\d{6}_\d{6}_(?P<tag>.+?)(?:_F[\d.]+_R[\d.]+)?$")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_run(run_dir: Path) -> dict[str, Any] | None:
    best = read_json(run_dir / "best_info.json")
    if not isinstance(best, dict):
        return None
    m = RUN_DIR_RE.match(run_dir.name)
    raw_tag = m.group("tag") if m else run_dir.name
    candidate = re.sub(r"_s\d+$", "", raw_tag)
    seed_match = re.search(r"_s(\d+)$", raw_tag)
    seed = int(seed_match.group(1)) if seed_match else None

    metrics = best.get("test_metrics") or {}
    abn = metrics.get("abnormal", {}) or {}
    nor = metrics.get("normal", {}) or {}
    fn_count = abn.get("count") or abn.get("support")
    nor_count = nor.get("count") or nor.get("support")
    abn_recall = best.get("test_abn_recall", abn.get("recall"))
    nor_recall = best.get("test_nor_recall", nor.get("recall"))
    fn = best.get("fn")
    if fn is None and isinstance(fn_count, (int, float)) and isinstance(abn_recall, (int, float)):
        fn = round(fn_count * (1.0 - abn_recall))
    fp = best.get("fp")
    if fp is None and isinstance(nor_count, (int, float)) and isinstance(nor_recall, (int, float)):
        fp = round(nor_count * (1.0 - nor_recall))

    history = read_json(run_dir / "history.json") or []
    val_curve = []
    for ev in history if isinstance(history, list) else []:
        if not isinstance(ev, dict):
            continue
        epoch = ev.get("epoch")
        v = ev.get("val_f1")
        if epoch is None or v is None:
            continue
        try:
            val_curve.append((int(epoch), float(v)))
        except Exception:
            continue
    val_curve.sort()

    return {
        "run_dir": str(run_dir),
        "tag": raw_tag,
        "candidate": candidate,
        "seed": seed,
        "f1": best.get("test_f1"),
        "fn": fn,
        "fp": fp,
        "best_epoch": best.get("best_epoch", best.get("epoch")),
        "val_f1_curve": val_curve,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["candidate"]].append(r)
    out = []
    for cand, items in grouped.items():
        f1s = [r["f1"] for r in items if isinstance(r.get("f1"), (int, float))]
        fns = [r["fn"] for r in items if isinstance(r.get("fn"), (int, float))]
        fps = [r["fp"] for r in items if isinstance(r.get("fp"), (int, float))]
        seeds = sorted(r["seed"] for r in items if r.get("seed") is not None)
        out.append({
            "candidate": cand,
            "n_runs": len(items),
            "seeds": seeds,
            "f1_mean": mean(f1s) if f1s else None,
            "fn_mean": mean(fns) if fns else None,
            "fp_mean": mean(fps) if fps else None,
        })
    out.sort(key=lambda r: (r.get("f1_mean") is None, -(r.get("f1_mean") or 0.0)))
    return out


def fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_md(out: Path, group_dir: Path, agg: list[dict[str, Any]], rows: list[dict[str, Any]],
             f1_plot: Path, val_plot: Path) -> None:
    lines = [
        f"# Group report — `{group_dir}`",
        "",
        f"- runs scanned: {len(rows)}",
        f"- candidates: {len(agg)}",
        "",
        f"![candidate F1]({f1_plot.name})",
        "",
        f"![val F1 curves]({val_plot.name})",
        "",
        "## Candidate aggregate",
        "",
        "| candidate | runs | seeds | F1 mean | FN mean | FP mean |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for r in agg:
        lines.append(
            f"| `{r['candidate']}` | {r['n_runs']} | "
            f"{','.join(str(s) for s in r['seeds'])} | "
            f"{fmt(r['f1_mean'])} | {fmt(r['fn_mean'], 3)} | {fmt(r['fp_mean'], 3)} |"
        )

    lines.extend([
        "",
        "## Runs",
        "",
        "| tag | seed | F1 | FN | FP | best epoch | run dir |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    rows_sorted = sorted(rows, key=lambda r: (r["candidate"], r.get("seed") or -1))
    for r in rows_sorted:
        lines.append(
            f"| `{r['tag']}` | {fmt(r.get('seed'))} | {fmt(r.get('f1'))} | "
            f"{fmt(r.get('fn'), 3)} | {fmt(r.get('fp'), 3)} | "
            f"{fmt(r.get('best_epoch'))} | `{r['run_dir']}` |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_candidate_f1(plot_path: Path, agg: list[dict[str, Any]]) -> None:
    rows = [r for r in agg if r.get("f1_mean") is not None]
    if not rows:
        return
    labels = [r["candidate"].replace("fresh0412_v11_rawbase_", "").replace("fresh0412_v11_", "") for r in rows]
    f1s = [r["f1_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.5 * len(labels) + 4), 4.5))
    ax.bar(range(len(labels)), f1s, color="#4878CF")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("F1 (mean over seeds)")
    ax.set_title("Candidate F1 (group)")
    if f1s:
        f1_min, f1_max = min(f1s), max(f1s)
        margin = max(0.0005, (f1_max - f1_min) * 0.2)
        ax.set_ylim(max(0.0, f1_min - margin), min(1.0, f1_max + margin))
    fig.tight_layout()
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)


def plot_val_curves(plot_path: Path, rows: list[dict[str, Any]]) -> None:
    by_cand: dict[str, list[list[tuple[int, float]]]] = defaultdict(list)
    for r in rows:
        if r.get("val_f1_curve"):
            by_cand[r["candidate"]].append(r["val_f1_curve"])
    if not by_cand:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for cand, curves in by_cand.items():
        label = cand.replace("fresh0412_v11_rawbase_", "").replace("fresh0412_v11_", "")
        for curve in curves:
            xs = [x for x, _ in curve]
            ys = [y for _, y in curve]
            ax.plot(xs, ys, alpha=0.5, linewidth=1, label=label)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    deduped = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        deduped.append((h, l))
    if deduped:
        ax.legend([h for h, _ in deduped], [l for _, l in deduped], fontsize=7, loc="lower right")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val F1")
    ax.set_title("Per-run val F1 curves (group)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)


def iter_run_dirs(group_dir: Path):
    for path in sorted(group_dir.iterdir()):
        if path.is_dir() and (path / "best_info.json").exists():
            yield path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group-dir", required=True, type=Path,
                    help="logs/<group>/ to scan")
    ap.add_argument("--out-md", type=Path, default=None,
                    help="default: <group-dir>/group_report.md")
    ap.add_argument("--out-f1-plot", type=Path, default=None,
                    help="default: <group-dir>/group_report_f1.png")
    ap.add_argument("--out-val-curves", type=Path, default=None,
                    help="default: <group-dir>/group_report_val_f1_curves.png")
    args = ap.parse_args()

    group_dir = args.group_dir.resolve()
    if not group_dir.is_dir():
        raise SystemExit(f"group dir not found: {group_dir}")

    out_md = args.out_md or (group_dir / "group_report.md")
    f1_plot = args.out_f1_plot or (group_dir / "group_report_f1.png")
    val_plot = args.out_val_curves or (group_dir / "group_report_val_f1_curves.png")

    rows: list[dict[str, Any]] = []
    for run_dir in iter_run_dirs(group_dir):
        parsed = parse_run(run_dir)
        if parsed is not None:
            rows.append(parsed)
    if not rows:
        raise SystemExit(f"no runs with best_info.json under {group_dir}")

    agg = aggregate(rows)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    plot_candidate_f1(f1_plot, agg)
    plot_val_curves(val_plot, rows)
    write_md(out_md, group_dir, agg, rows, f1_plot, val_plot)
    print(f"[group_report] runs={len(rows)} candidates={len(agg)}")
    print(f"  md   : {out_md}")
    print(f"  plot : {f1_plot}")
    print(f"  curves: {val_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
