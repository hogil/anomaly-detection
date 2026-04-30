#!/usr/bin/env python
"""Build a NT (normal_threshold) effect markdown + plot.

Walks logs/**/best_info.json and pulls each run's `normal_threshold_results`
for two cases:
- "그냥" (no NT applied, argmax-only) -> NT key 0.5
- "NT applied" -> NT key 0.9 (the reporting default)

Aggregates mean FN / FP / F1 across runs of the matched candidate so the
output is a 2-row table + a side-by-side bar plot.

Typical use:
  python scripts/generate_nt_effect_report.py
    --candidate-prefix fresh0412_v11_n700
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
    ap.add_argument("--collapse-threshold", type=int, default=50,
                    help="drop runs where any FN/FP exceeds this (collapsed/degenerate). default 50.")
    args = ap.parse_args()

    # Per-run paired records: (no-NT, with-NT) on the SAME run only.
    # That way both columns share the same denominator and the mean FN/FP move
    # is genuinely the threshold effect, not a different run set.
    paired_runs: list[dict[str, Any]] = []
    matched_candidates: set[str] = set()
    skipped_collapsed = 0

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

        def at(nt_key: str) -> tuple[int, int, float | None] | None:
            payload = ntr.get(nt_key)
            if not isinstance(payload, dict):
                return None
            fnfp = derive_fn_fp(payload.get("metrics"))
            if fnfp is None:
                return None
            return fnfp[0], fnfp[1], payload.get("f1")

        no_nt = at("0.5")
        nt = at("0.9")
        if no_nt is None or nt is None:
            continue
        # Drop collapsed runs (any side > collapse threshold).
        if no_nt[0] >= args.collapse_threshold or no_nt[1] >= args.collapse_threshold or \
           nt[0] >= args.collapse_threshold or nt[1] >= args.collapse_threshold:
            skipped_collapsed += 1
            continue

        matched_candidates.add(cand)
        paired_runs.append({
            "tag": tag, "candidate": cand,
            "no_nt_fn": no_nt[0], "no_nt_fp": no_nt[1], "no_nt_f1": no_nt[2],
            "nt_fn": nt[0], "nt_fp": nt[1], "nt_f1": nt[2],
        })

    if not paired_runs:
        raise SystemExit(
            f"no paired NT results found under {args.logs_dir} (filter "
            f"prefix='{args.candidate_prefix}' contains='{args.candidate_contains}')"
        )

    runs_seen = len(paired_runs)
    f1_no = [r["no_nt_f1"] for r in paired_runs if isinstance(r["no_nt_f1"], (int, float))]
    f1_nt = [r["nt_f1"] for r in paired_runs if isinstance(r["nt_f1"], (int, float))]
    agg = [
        {
            "label": "그냥 (no NT)",
            "nt": 0.5,
            "n_runs": runs_seen,
            "f1_mean": mean(f1_no) if f1_no else None,
            "fn_mean": mean(r["no_nt_fn"] for r in paired_runs),
            "fp_mean": mean(r["no_nt_fp"] for r in paired_runs),
        },
        {
            "label": "NT=0.9",
            "nt": 0.9,
            "n_runs": runs_seen,
            "f1_mean": mean(f1_nt) if f1_nt else None,
            "fn_mean": mean(r["nt_fn"] for r in paired_runs),
            "fp_mean": mean(r["nt_fp"] for r in paired_runs),
        },
    ]

    # Markdown
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# NT (normal_threshold) effect",
        "",
        f"- paired runs (no-NT and NT both available): **{runs_seen}**",
        f"- collapsed runs dropped (any FN/FP ≥ {args.collapse_threshold}): {skipped_collapsed}",
        f"- candidates matched: {len(matched_candidates)}",
        "",
        f"![NT effect]({args.out_plot.name})",
        "",
        "| condition | runs | F1 mean | FN mean | FP mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in agg:
        f1 = f"{row['f1_mean']:.4f}" if row["f1_mean"] is not None else "-"
        md_lines.append(
            f"| {row['label']} | {row['n_runs']} | {f1} | {row['fn_mean']:.2f} | {row['fp_mean']:.2f} |"
        )
    md_lines.extend([
        "",
        "**해석**: NT 적용 시 \"p_normal ≥ 0.9 이어야 normal 로 판정\" 기준으로 더 많은 케이스를 abnormal 쪽으로 보냅니다. 결과: **FN 감소 (불량 더 잡음) / FP 약간 증가 (정상도 abnormal 로 오인 가능)**. 기본 NT=0.9 가 sweet spot. 0.99 이상 극단값은 권장 안 함 (memory `feedback_normal_threshold_099`).",
    ])
    args.out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["label", "nt", "n_runs", "f1_mean", "fn_mean", "fp_mean"])
        w.writeheader()
        for row in agg:
            w.writerow(row)

    # Plot — 2 conditions side by side, compact + pastel
    # English x-labels avoid matplotlib Korean-font box-glyph issues.
    en_labels = ["no NT (argmax)", "NT = 0.9"]

    fns = [r["fn_mean"] for r in agg]
    fps = [r["fp_mean"] for r in agg]

    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    width = 0.32
    x = list(range(len(en_labels)))
    ax.bar([i - width / 2 for i in x], fns, width=width, label="FN", color="#F4B6C2")  # pastel pink
    ax.bar([i + width / 2 for i in x], fps, width=width, label="FP", color="#A8D8EA")  # pastel blue
    for i, (fn, fp) in enumerate(zip(fns, fps)):
        ax.text(i - width / 2, fn + 0.05, f"{fn:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, fp + 0.05, f"{fp:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(en_labels, fontsize=9)
    ax.set_ylabel("count (mean)", fontsize=9)
    ax.set_title(f"NT effect ({runs_seen} runs)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=140)
    plt.close(fig)

    print(f"[nt_effect] runs={runs_seen} candidates={len(matched_candidates)}")
    print(f"  md  : {args.out_md}")
    print(f"  csv : {args.out_csv}")
    print(f"  plot: {args.out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
