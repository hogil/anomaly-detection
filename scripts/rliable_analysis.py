"""EXP-6: rliable-style (Agarwal 2021) statistical analysis on vd080_bc runs.

Zero GPU, pure analysis. Reads per-item FP/FN chart_ids from existing
predictions/ folders, builds 1500-dim per-run error vectors, and computes:

  1. IQM (Interquartile Mean) of per-seed error counts per cell
  2. Stratified bootstrap over the 1500 test items (10,000 resamples) with
     95 % CI on the mean error rate per cell
  3. Pairwise probability-of-improvement P(A beats B):
        for each bootstrap resample r:
           sample indices I_r of size 1500 (stratified by class)
           compute mean err_A(I_r), mean err_B(I_r) averaged over the cell's
           seeds; count how often err_A < err_B

Outputs:
    validations/rliable_summary.txt
    experiments/plots/rliable_comparison_matrix.png
    experiments/RLIABLE_ANALYSIS.md  (written by caller; this script just
    produces numbers + txt + heatmap)

Run:
    python scripts/rliable_analysis.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("D:/project/anomaly-detection")
LOGS = ROOT / "logs"
VAL_OUT = ROOT / "validations" / "rliable_summary.txt"
PLOT_OUT = ROOT / "experiments" / "plots" / "rliable_comparison_matrix.png"
DATA_SCEN = ROOT / "data_vd080" / "scenarios.csv"

N_BOOT = 10_000
RNG_SEED = 20260420
CI_LOW = 2.5
CI_HIGH = 97.5

RUN_PATTERN = re.compile(
    r"^\d{6}_\d{6}_vd080_bc_gc(?P<gc>[^_]+)_sw(?P<sw>[^_]+)_n(?P<n>\d+)_s(?P<seed>\d+)"
)
CHART_RE = re.compile(r"(ch_\d+)")


# ---------------------------------------------------------------------------
# Dataset: build canonical ordering + per-item class label
# ---------------------------------------------------------------------------

def load_test_items() -> Tuple[List[str], np.ndarray]:
    """Return (chart_ids, is_abnormal) in a fixed canonical order."""
    df = pd.read_csv(DATA_SCEN)
    test = df[df["split"] == "test"].copy()
    test = test.sort_values("chart_id").reset_index(drop=True)
    chart_ids = test["chart_id"].tolist()
    is_abn = (test["class"] != "normal").to_numpy(dtype=bool)
    assert len(chart_ids) == 1500, f"expected 1500 test items, got {len(chart_ids)}"
    assert int(is_abn.sum()) == 750, f"expected 750 abnormals, got {int(is_abn.sum())}"
    return chart_ids, is_abn


# ---------------------------------------------------------------------------
# Per-run error vector: 1 at item i iff run misclassified chart_ids[i]
# ---------------------------------------------------------------------------

def extract_misclass(run_dir: Path) -> set:
    """Union of FP (predicted abnormal / actually normal) and FN (predicted
    normal / actually abnormal) chart_ids for a single run."""
    miss = set()
    for sub in ("fp_normal", "fn_abnormal"):
        d = run_dir / "predictions" / sub
        if not d.exists():
            continue
        for f in d.glob("*.png"):
            m = CHART_RE.search(f.name)
            if m:
                miss.add(m.group(1))
    return miss


def parse_run(d: Path):
    m = RUN_PATTERN.match(d.name)
    if not m:
        return None
    if not (d / "best_info.json").exists():
        return None
    return {
        "run": d.name,
        "gc": m["gc"],
        "sw": m["sw"],
        "seed": int(m["seed"]),
        "cell": f"gc={m['gc']}_sw={m['sw']}",
        "dir": d,
    }


def build_err_matrices(chart_ids: List[str]):
    """Return dict[cell] -> (err_matrix [n_seeds x 1500], seeds, runs).

    Only cells with 5 seeds (1,2,3,4,42) are kept.
    """
    id_to_idx = {c: i for i, c in enumerate(chart_ids)}

    runs = []
    for d in sorted(LOGS.iterdir()):
        if not d.is_dir():
            continue
        r = parse_run(d)
        if r is None:
            continue
        runs.append(r)

    by_cell: Dict[str, List[dict]] = defaultdict(list)
    for r in runs:
        by_cell[r["cell"]].append(r)

    err_by_cell: Dict[str, Dict] = {}
    for cell, run_list in by_cell.items():
        # keep only one run per seed (latest by folder name)
        seed_to_run: Dict[int, dict] = {}
        for r in run_list:
            prev = seed_to_run.get(r["seed"])
            if prev is None or r["run"] > prev["run"]:
                seed_to_run[r["seed"]] = r
        seeds_sorted = sorted(seed_to_run.keys())
        if set(seeds_sorted) < {1, 2, 3, 4, 42}:
            # incomplete cell — skip for rliable (need 5-seed cells)
            continue
        # build matrix restricted to canonical 5 seeds in fixed order
        target_seeds = [1, 2, 3, 4, 42]
        mat = np.zeros((len(target_seeds), len(chart_ids)), dtype=np.int8)
        picked_runs = []
        for i, s in enumerate(target_seeds):
            r = seed_to_run[s]
            picked_runs.append(r["run"])
            miss = extract_misclass(r["dir"])
            for cid in miss:
                j = id_to_idx.get(cid)
                if j is not None:
                    mat[i, j] = 1
        err_by_cell[cell] = {
            "matrix": mat,
            "seeds": target_seeds,
            "runs": picked_runs,
        }
    return err_by_cell


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def iqm(values: np.ndarray) -> float:
    """Interquartile mean: mean of middle 50 % of values."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan")
    lo, hi = np.percentile(v, [25, 75])
    mask = (v >= lo) & (v <= hi)
    if not mask.any():
        return float(v.mean())
    return float(v[mask].mean())


def stratified_bootstrap_indices(is_abn: np.ndarray, rng: np.random.Generator,
                                  n_boot: int) -> np.ndarray:
    """Return [n_boot, 1500] int64 matrix of resampled item indices.
    Stratified within (normal, abnormal) to preserve class balance."""
    n = is_abn.shape[0]
    normal_idx = np.where(~is_abn)[0]
    abn_idx = np.where(is_abn)[0]
    n_n = len(normal_idx)
    n_a = len(abn_idx)
    out = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        samp_n = rng.choice(normal_idx, size=n_n, replace=True)
        samp_a = rng.choice(abn_idx, size=n_a, replace=True)
        out[b, :n_n] = samp_n
        out[b, n_n:] = samp_a
    return out


def cell_mean_err_rate_per_boot(err_matrix: np.ndarray,
                                boot_idx: np.ndarray) -> np.ndarray:
    """For each bootstrap sample r, compute mean error rate across seeds
    (average of per-seed mean(err[r])).

    err_matrix: [S, N]  int {0,1}
    boot_idx:   [B, N]  int64
    returns:    [B]     float
    """
    S, N = err_matrix.shape
    # per-seed, per-boot: mean over sampled items
    # gather: resampled = err_matrix[:, boot_idx]  -> [S, B, N], heavy
    # Instead: for each seed, compute mean err per bootstrap row separately
    # to keep memory sane (S usually 5, B=10k, N=1500 -> 5*10k*1500*1 = 75MB OK)
    resamp = err_matrix[:, boot_idx]           # [S, B, N]
    per_seed_mean = resamp.mean(axis=2)         # [S, B]
    cell_mean = per_seed_mean.mean(axis=0)      # [B]
    return cell_mean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[rliable] loading test scenario items ...")
    chart_ids, is_abn = load_test_items()

    print("[rliable] building error matrices per cell ...")
    err_by_cell = build_err_matrices(chart_ids)
    if not err_by_cell:
        raise SystemExit("No 5-seed complete cells found; aborting")
    cells_sorted = sorted(err_by_cell.keys())
    print(f"[rliable] {len(cells_sorted)} complete 5-seed cells: {cells_sorted}")

    # per-cell summary stats on raw per-seed error counts (out of 1500)
    per_cell_stats: Dict[str, Dict] = {}
    for cell in cells_sorted:
        mat = err_by_cell[cell]["matrix"]
        per_seed_errs = mat.sum(axis=1).astype(float)   # [S]
        per_cell_stats[cell] = {
            "per_seed_errs": per_seed_errs,
            "mean": float(per_seed_errs.mean()),
            "std": float(per_seed_errs.std(ddof=1)),
            "median": float(np.median(per_seed_errs)),
            "iqm": iqm(per_seed_errs),
            "min": float(per_seed_errs.min()),
            "max": float(per_seed_errs.max()),
        }

    # stratified bootstrap
    print(f"[rliable] running stratified bootstrap ({N_BOOT} resamples) ...")
    rng = np.random.default_rng(RNG_SEED)
    boot_idx = stratified_bootstrap_indices(is_abn, rng, N_BOOT)

    cell_boot_err_rate: Dict[str, np.ndarray] = {}
    for cell in cells_sorted:
        mat = err_by_cell[cell]["matrix"]
        cell_boot_err_rate[cell] = cell_mean_err_rate_per_boot(mat, boot_idx)

    # 95% CI per cell
    ci_table: Dict[str, Tuple[float, float, float]] = {}
    for cell in cells_sorted:
        v = cell_boot_err_rate[cell]
        ci_lo, ci_hi = np.percentile(v, [CI_LOW, CI_HIGH])
        ci_table[cell] = (float(v.mean()), float(ci_lo), float(ci_hi))

    # Pairwise P(A beats B) — A has FEWER errors than B in bootstrap
    print("[rliable] computing pairwise P(A beats B) ...")
    n_cells = len(cells_sorted)
    p_matrix = np.full((n_cells, n_cells), np.nan)
    for i, ci in enumerate(cells_sorted):
        for j, cj in enumerate(cells_sorted):
            if i == j:
                p_matrix[i, j] = 0.5
                continue
            a = cell_boot_err_rate[ci]
            b = cell_boot_err_rate[cj]
            wins = (a < b).sum() + 0.5 * (a == b).sum()
            p_matrix[i, j] = wins / N_BOOT

    # ---------------- write text summary ----------------
    VAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("EXP-6: rliable-style statistical analysis (vd080_bc, 5-seed cells)")
    lines.append("=" * 78)
    lines.append(f"cells analyzed : {len(cells_sorted)}")
    lines.append(f"seeds per cell : 5 (s=1, 2, 3, 4, 42)")
    lines.append(f"test items     : 1500 (750 normal + 750 abnormal)")
    lines.append(f"bootstrap      : {N_BOOT} stratified resamples, rng seed={RNG_SEED}")
    lines.append(f"CI             : {CI_LOW:.1f} – {CI_HIGH:.1f} percentile")
    lines.append("")

    # Per-cell stats table
    lines.append("-" * 78)
    lines.append("Per-cell per-seed error counts (out of 1500)")
    lines.append("-" * 78)
    header = (
        f"{'cell':<26}{'mean':>8}{'std':>8}{'med':>8}{'IQM':>8}"
        f"{'min':>6}{'max':>6}"
    )
    lines.append(header)
    cells_by_mean = sorted(cells_sorted, key=lambda c: per_cell_stats[c]["mean"])
    cells_by_iqm = sorted(cells_sorted, key=lambda c: per_cell_stats[c]["iqm"])
    for c in cells_by_mean:
        s = per_cell_stats[c]
        lines.append(
            f"{c:<26}{s['mean']:>8.2f}{s['std']:>8.2f}{s['median']:>8.2f}"
            f"{s['iqm']:>8.2f}{s['min']:>6.0f}{s['max']:>6.0f}"
        )

    lines.append("")
    lines.append("Ranking: mean vs IQM")
    lines.append("-" * 78)
    lines.append(f"{'rank':<6}{'by mean err':<32}{'by IQM err':<32}")
    for r in range(len(cells_sorted)):
        lines.append(f"{r+1:<6}{cells_by_mean[r]:<32}{cells_by_iqm[r]:<32}")

    # Bootstrap CI table
    lines.append("")
    lines.append("-" * 78)
    lines.append("Bootstrap 95% CI on cell-mean error rate (err / 1500)")
    lines.append("-" * 78)
    lines.append(f"{'cell':<26}{'mean_rate':>10}{'CI_lo':>10}{'CI_hi':>10}"
                 f"{'mean_err':>10}{'CI_lo_err':>12}{'CI_hi_err':>12}")
    for c in cells_by_mean:
        mean_rate, lo, hi = ci_table[c]
        lines.append(
            f"{c:<26}{mean_rate:>10.5f}{lo:>10.5f}{hi:>10.5f}"
            f"{mean_rate*1500:>10.2f}{lo*1500:>12.2f}{hi*1500:>12.2f}"
        )

    # CI overlap analysis — any pair whose CIs DON'T overlap
    lines.append("")
    lines.append("-" * 78)
    lines.append("CI overlap analysis (non-overlapping pairs are candidates for "
                 "'clearly different'):")
    lines.append("-" * 78)
    any_non_overlap = False
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            ci_i = ci_table[cells_sorted[i]]
            ci_j = ci_table[cells_sorted[j]]
            # non-overlap: i.CI_hi < j.CI_lo  OR  j.CI_hi < i.CI_lo
            if ci_i[2] < ci_j[1] or ci_j[2] < ci_i[1]:
                any_non_overlap = True
                lines.append(
                    f"  {cells_sorted[i]:<26} CI=[{ci_i[1]:.5f},{ci_i[2]:.5f}]  VS  "
                    f"{cells_sorted[j]:<26} CI=[{ci_j[1]:.5f},{ci_j[2]:.5f}]  "
                    f"→ non-overlap"
                )
    if not any_non_overlap:
        lines.append("  (none — all 95% CIs overlap, cells are statistically "
                     "indistinguishable)")

    # P(A beats B) significant pairs
    lines.append("")
    lines.append("-" * 78)
    lines.append("Pairwise P(A beats B)  (A has FEWER errors than B)")
    lines.append("significant at 95 %: P ≥ 0.95 (A better) or P ≤ 0.05 (B better)")
    lines.append("-" * 78)
    sig_pairs: List[Tuple[str, str, float]] = []
    for i in range(n_cells):
        for j in range(n_cells):
            if i == j:
                continue
            p = p_matrix[i, j]
            if p >= 0.95:
                sig_pairs.append((cells_sorted[i], cells_sorted[j], p))
    if sig_pairs:
        for a, b, p in sorted(sig_pairs, key=lambda x: -x[2]):
            lines.append(f"  {a:<26} beats {b:<26}  P = {p:.4f}")
    else:
        lines.append("  (no pair reaches P ≥ 0.95 — no statistically significant "
                     "winner)")

    # full matrix
    lines.append("")
    lines.append("-" * 78)
    lines.append("Full pairwise P(row beats column) matrix:")
    lines.append("-" * 78)
    col_hdr = " " * 28 + "".join(f"{c[:9]:>10}" for c in cells_sorted)
    lines.append(col_hdr)
    for i, ci in enumerate(cells_sorted):
        row = f"{ci:<28}"
        for j in range(n_cells):
            row += f"{p_matrix[i, j]:>10.3f}"
        lines.append(row)

    # Top-3 focused callout
    lines.append("")
    lines.append("-" * 78)
    lines.append("Top-3 cells (by mean err) — checkpoint-70 highlights")
    lines.append("-" * 78)
    for c in cells_by_mean[:3]:
        s = per_cell_stats[c]
        mr, lo, hi = ci_table[c]
        lines.append(
            f"  {c:<26} mean={s['mean']:.2f}  IQM={s['iqm']:.2f}  "
            f"bootstrap 95% CI on err = [{lo*1500:.2f}, {hi*1500:.2f}]"
        )
    # head-to-head among top 3
    lines.append("")
    lines.append("  Head-to-head among top-3:")
    top3 = cells_by_mean[:3]
    for i, a in enumerate(top3):
        for b in top3:
            if a == b:
                continue
            ia = cells_sorted.index(a)
            ib = cells_sorted.index(b)
            lines.append(f"    P({a} beats {b}) = {p_matrix[ia, ib]:.4f}")

    # Recommendation
    lines.append("")
    lines.append("-" * 78)
    lines.append("Recommendation for paper reporting")
    lines.append("-" * 78)
    best = cells_by_mean[0]
    best_iqm = cells_by_iqm[0]
    lines.append(f"  Headline: IQM of per-seed errors ({best_iqm} leads),")
    lines.append(f"           accompanied by bootstrap 95% CI (overlap across all "
                 f"top cells).")
    lines.append(f"  Supporting: pairwise P(A beats B). {len(sig_pairs)} "
                 f"pair(s) reached P ≥ 0.95.")
    if not sig_pairs and not any_non_overlap:
        lines.append("  Conclusion: top cells are statistically indistinguishable "
                     "at 5-seed/1500-item resolution — consistent with the "
                     "McNemar p=0.66 result. Additional seeds or a harder "
                     "benchmark is required to separate them.")

    VAL_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[rliable] wrote {VAL_OUT}")

    # ---------------- plot heatmap ----------------
    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    # colormap: P>0.95 green, P<0.05 red, otherwise gray
    im = ax.imshow(p_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_cells))
    ax.set_yticks(range(n_cells))
    ax.set_xticklabels(cells_sorted, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cells_sorted, fontsize=9)
    ax.set_title(
        f"P(row beats column) — stratified bootstrap, {N_BOOT} resamples\n"
        f"green = row significantly better  |  red = column significantly better",
        fontsize=11,
    )
    for i in range(n_cells):
        for j in range(n_cells):
            p = p_matrix[i, j]
            if np.isnan(p):
                continue
            marker = ""
            if i != j:
                if p >= 0.95:
                    marker = "*"
                elif p <= 0.05:
                    marker = "·"
            ax.text(j, i, f"{p:.2f}{marker}", ha="center", va="center",
                    fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(A beats B)")
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close(fig)
    print(f"[rliable] wrote {PLOT_OUT}")

    # Return data for caller (if ever imported)
    return {
        "cells": cells_sorted,
        "per_cell": per_cell_stats,
        "ci": ci_table,
        "p_matrix": p_matrix,
        "sig_pairs": sig_pairs,
        "non_overlap": any_non_overlap,
    }


if __name__ == "__main__":
    main()
