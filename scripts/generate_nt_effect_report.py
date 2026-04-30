#!/usr/bin/env python
"""Build a NT (normal_threshold) effect markdown + plot.

Walks logs/**/best_info.json and pulls each run's `normal_threshold_results`
dict (thresholds 0.5 / 0.9 / 0.99 / 0.999 / 0.9999). For every threshold
level it averages FN, FP, and F1 across rawbase baseline runs and reports a
table + bar plot.

Typical use:
  python scripts/generate_nt_effect_report.py
    --candidate-prefix fresh0412_v11_refcheck_raw_n700
    --out-md  validations/nt_effect.md
    --out-csv validations/nt_effect.csv
    --out-plot validations/nt_effect.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def derive_fn_fp(metrics: dict[str, Any]) -> tuple[int, int] | None:
    if not isinstance(metrics, dict):
        return None
    abn = metrics.get("abnormal") or {}
    nor = metrics.get("normal") or {}
    abn_n = abn.get("count") or abn.get("support")
    nor_n = nor.get("count") or nor.get("support")
    abn_R = abn.get("recall")
    nor_R = nor.get("recall")
    if None in (abn_n, nor_n, abn_R, nor_R):
        return None
    return int(round(abn_n * (1.0 - abn_R))), int(round(nor_n * (1.0 - nor_R)))


def candidate_from_dir(folder: str) -> tuple[str, str]:
    name = re.sub(r"^\d+_\d+_", "", folder)
    name = re.sub(r"_F[\d.]+_R[\d.]+$", "", name)
    cand = re.sub(r"_s\d+$", "", name)
    return cand, name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-dir", default="logs", type=Path)
    ap.add_argument("--candidate-contains", default="fresh0412_v11",
                    help="Only include runs whose candidate name contains this string")
    ap.add_argument("--candidate-prefix", default="",
                    help="Only include runs whose candidate exactly matches this prefix (most precise filter)")
    ap.add_argument("--out-md", type=Path, default=Path("validations/nt_effect.md"))
    ap.add_argument("--out-csv", type=Path, default=Path("validations/nt_effect.csv"))
    ap.add_argument("--out-plot", type=Path, default=Path("validations/nt_effect.png"))
    args = ap.parse_args()

    rows_per_nt: dict[float, list[dict[str, Any]]] = defaultdict(list)
    matched_candidates: set[str] = set()
    runs_seen = 0

    for fp in sorted(glob.glob(str(args.logs_dir / "**" / "best_info.json"), recursive=True)):
        parts = os.path.normpath(fp).split(os.sep)
        if len(parts) < 2:
            continue
        cand, tag = candidate_from_dir(parts[-2])
        if args.candidate_prefix and not cand.startswith(args.candidate_prefix):
            continue
        if not args.candidate_prefix and args.candidate_contains and args.candidate_contains not in cand:
            continue
        try:
            d = json.loads(Path(fp).read_text(encoding="utf-8"))
        except Exception:
            continue
        ntr = d.get("normal_threshold_results")
        if not isinstance(ntr, dict):
            continue
        runs_seen += 1
        matched_candidates.add(cand)
        for nt_str, payload in ntr.items():
            try:
                nt_val = float(nt_str)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("metrics")
            f1 = payload.get("f1")
            fn_fp = derive_fn_fp(metrics)
            if fn_fp is None:
                continue
            rows_per_nt[nt_val].append({
                "tag": tag,
                "candidate": cand,
                "f1": f1,
                "fn": fn_fp[0],
                "fp": fn_fp[1],
            })

    if not rows_per_nt:
        raise SystemExit(
            f"no normal_threshold_results found under {args.logs_dir} with filter "
            f"prefix='{args.candidate_prefix}' contains='{args.candidate_contains}'"
        )

    # Aggregate
    agg = []
    for nt_val in sorted(rows_per_nt.keys()):
        rs = rows_per_nt[nt_val]
        agg.append({
            "nt": nt_val,
            "n_runs": len(rs),
            "f1_mean": mean(r["f1"] for r in rs if isinstance(r["f1"], (int, float))) if any(isinstance(r["f1"], (int, float)) for r in rs) else None,
            "fn_mean": mean(r["fn"] for r in rs),
            "fp_mean": mean(r["fp"] for r in rs),
            "fn_max": max(r["fn"] for r in rs),
            "fp_max": max(r["fp"] for r in rs),
        })

    # Markdown
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# NT (normal_threshold) effect",
        "",
        f"- runs scanned: {runs_seen}",
        f"- candidates matched: {len(matched_candidates)} ({', '.join(sorted(matched_candidates)[:5])}{'...' if len(matched_candidates) > 5 else ''})",
        "",
        f"![NT effect]({args.out_plot.name})",
        "",
        "| NT | runs | F1 mean | FN mean | FP mean | FN max | FP max |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in agg:
        f1 = f"{row['f1_mean']:.4f}" if row["f1_mean"] is not None else "-"
        md_lines.append(
            f"| {row['nt']} | {row['n_runs']} | {f1} | {row['fn_mean']:.2f} | {row['fp_mean']:.2f} | {row['fn_max']} | {row['fp_max']} |"
        )
    md_lines.extend([
        "",
        "**해석**: NT(`normal_threshold`)를 올리면 \"normal 로 판정되려면 더 높은 confidence 필요\" 기준이 되므로 더 많은 케이스를 abnormal 로 보내게 됩니다. 결과: **NT 올라가면 FN 줄고 (불량 더 잘 잡음) FP 늘어남 (정상도 abnormal 로 오인)**. 0.9999 같은 극단값에서는 모든 케이스를 abnormal 로 분류해서 FP=전체 normal 수가 되는 degenerate. test-peeking 위험으로 권장하지 않습니다 (memory `feedback_normal_threshold_099`).",
    ])
    args.out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["nt", "n_runs", "f1_mean", "fn_mean", "fp_mean", "fn_max", "fp_max"])
        w.writeheader()
        for row in agg:
            w.writerow(row)

    # Plot — drop degenerate NT rows where FP (or FN) saturates the test set,
    # otherwise the y-axis is dominated by collapsed runs.
    plot_rows = [r for r in agg if r["fp_mean"] < 100 and r["fn_mean"] < 100]
    if not plot_rows:
        plot_rows = agg
    nts = [str(r["nt"]) for r in plot_rows]
    fns = [r["fn_mean"] for r in plot_rows]
    fps = [r["fp_mean"] for r in plot_rows]

    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    width = 0.4
    x = list(range(len(nts)))
    ax.bar([i - width / 2 for i in x], fns, width=width, label="FN (mean)", color="#E43320")
    ax.bar([i + width / 2 for i in x], fps, width=width, label="FP (mean)", color="#F5B041")
    ax.set_xticks(x)
    ax.set_xticklabels(nts)
    ax.set_xlabel("normal_threshold")
    ax.set_ylabel("count (mean over runs)")
    ax.set_title(f"NT effect on FN / FP (rawbase, {runs_seen} runs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=130)
    plt.close(fig)

    print(f"[nt_effect] runs={runs_seen} candidates={len(matched_candidates)}")
    print(f"  md  : {args.out_md}")
    print(f"  csv : {args.out_csv}")
    print(f"  plot: {args.out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
