"""Regenerate paper plots at N=122 (109 vd080_bc + 13 vd080_ax runs).

Matplotlib only — no seaborn dependency.

Outputs 8 PNGs into experiments/plots/:
  1. vd080_bc_20cells_heatmap.png       — 4x5 mu-err heatmap with sigma annotations
  2. vd080_bc_per_cell_dotplot.png      — 20 cells x up to 8 seeds
  3. vd080_ax_wd_boxplot.png            — 3 wd levels, individual seed dots
  4. vd080_bc_gc_pool.png               — pooled by gc (violin + mean)
  5. chronic_charts_bar.png             — top-10 chart_id failure counts (FN vs FP)
  6. fn_fp_composition.png              — 72/28 FP vs FN stacked bar
  7. n5_vs_n8_flip.png                  — gc=0.5 sw3mean n=5 vs n=8 reproducibility failure
  8. best_epoch_dist.png                — histogram of best-epoch over 122 runs
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path("D:/project/anomaly-detection")
LOGS = ROOT / "logs"
OUT = ROOT / "experiments" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": False,
})

# ------------------------------------------------------------
# Regexes
# ------------------------------------------------------------
BC_PAT = re.compile(r"vd080_bc_gc([0-9p]+)_sw([0-9a-z]+)_n(\d+)_s(\d+)")
AX_PAT = re.compile(r"vd080_ax_([a-z]+[0-9]+)_n(\d+)_s(\d+)")
CHART_RE = re.compile(r"pred_(normal|abnormal)_(ch_\d+)\.png$")

GC_ORDER = ["0p5", "1p0", "2p0", "5p0"]
GC_DISP = {"0p5": "0.5", "1p0": "1.0", "2p0": "2.0", "5p0": "5.0"}
SW_ORDER = ["1raw", "3med", "3mean", "5med", "5mean"]
SW_DISP = {"1raw": "sw1 raw", "3med": "sw3 med", "3mean": "sw3 mean",
           "5med": "sw5 med", "5mean": "sw5 mean"}
GC_COLOR = {"0p5": "#4C72B0", "1p0": "#DD8452", "2p0": "#55A467", "5p0": "#C44E52"}

STAGE9_SEEDS = {5, 6, 7}


def load_best(run_dir: Path):
    p = run_dir / "best_info.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    tm = data.get("test_metrics") or {}
    n = tm.get("normal") or {}
    a = tm.get("abnormal") or {}
    nR = n.get("recall")
    aR = a.get("recall")
    nc = n.get("count") or 0
    ac = a.get("count") or 0
    fp = round(nc * (1.0 - nR)) if nR is not None else None
    fn = round(ac * (1.0 - aR)) if aR is not None else None
    return {
        "epoch": data.get("epoch"),
        "val_f1": data.get("val_f1"),
        "test_f1": data.get("test_f1"),
        "fp": fp,
        "fn": fn,
    }


def collect_vd080_bc():
    rows = []
    for run_dir in LOGS.iterdir():
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if "vd080_bc" not in name:
            continue
        m = BC_PAT.search(name)
        if not m:
            continue
        gc, sw, _n, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        best = load_best(run_dir)
        if best is None or best["test_f1"] is None:
            continue
        rows.append({
            "run_dir": run_dir,
            "run": name,
            "gc": gc,
            "sw": sw,
            "seed": seed,
            **best,
        })
    return rows


def collect_vd080_ax():
    rows = []
    for run_dir in LOGS.iterdir():
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if "vd080_ax" not in name:
            continue
        m = AX_PAT.search(name)
        if not m:
            continue
        level, _n, seed = m.group(1), int(m.group(2)), int(m.group(3))
        best = load_best(run_dir)
        if best is None or best["test_f1"] is None:
            continue
        rows.append({
            "run_dir": run_dir,
            "run": name,
            "level": level,
            "seed": seed,
            **best,
        })
    return rows


# ------------------------------------------------------------
# 1. vd080_bc_20cells_heatmap.png
# ------------------------------------------------------------
def plot_20cells_heatmap(bc_rows):
    mat_mu = np.full((len(GC_ORDER), len(SW_ORDER)), np.nan)
    mat_sd = np.full((len(GC_ORDER), len(SW_ORDER)), np.nan)
    n_seeds = np.zeros((len(GC_ORDER), len(SW_ORDER)), dtype=int)

    by_cell = defaultdict(list)
    for r in bc_rows:
        by_cell[(r["gc"], r["sw"])].append(r["test_f1"])

    for i, gc in enumerate(GC_ORDER):
        for j, sw in enumerate(SW_ORDER):
            vals = by_cell.get((gc, sw), [])
            if not vals:
                continue
            err = [(1.0 - v) * 100 for v in vals]
            mat_mu[i, j] = float(np.mean(err))
            mat_sd[i, j] = float(np.std(err, ddof=0)) if len(err) > 1 else 0.0
            n_seeds[i, j] = len(err)

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    cmap = plt.get_cmap("YlOrRd")

    finite = mat_mu[np.isfinite(mat_mu)]
    vmin = float(np.nanmin(finite)) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if vmax - vmin < 1e-3:
        vmax = vmin + 0.5

    im = ax.imshow(mat_mu, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("mean test error mu (1-F1) x 100 [%]")

    ax.set_xticks(range(len(SW_ORDER)))
    ax.set_xticklabels([SW_DISP[s] for s in SW_ORDER])
    ax.set_yticks(range(len(GC_ORDER)))
    ax.set_yticklabels([f"gc={GC_DISP[g]}" for g in GC_ORDER])

    for i in range(len(GC_ORDER)):
        for j in range(len(SW_ORDER)):
            mu = mat_mu[i, j]
            sd = mat_sd[i, j]
            n = n_seeds[i, j]
            if not np.isfinite(mu):
                ax.text(j, i, "n/a", ha="center", va="center", color="#888",
                        fontsize=9)
                continue
            pos = (mu - vmin) / max(vmax - vmin, 1e-9)
            color = "white" if pos > 0.55 else "black"
            ax.text(j, i,
                    f"mu={mu:.2f}%\nsigma={sd:.2f}\nn={n}",
                    ha="center", va="center", color=color,
                    fontsize=8.5, fontweight="bold")

    ax.set_title(f"vd080_bc 20 cells (gc x sw): mean test error mu, sigma annotation  (N={sum(n_seeds.flatten())} runs)",
                 fontsize=11.5)
    ax.set_xlabel("Validation smoothing")
    ax.set_ylabel("Gradient clip norm")
    fig.tight_layout()
    out = OUT / "vd080_bc_20cells_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------
# 2. vd080_bc_per_cell_dotplot.png
# ------------------------------------------------------------
def plot_per_cell_dotplot(bc_rows):
    cells = []
    for gc in GC_ORDER:
        for sw in SW_ORDER:
            cells.append((gc, sw))

    by_cell = defaultdict(list)
    for r in bc_rows:
        by_cell[(r["gc"], r["sw"])].append(r)

    fig, ax = plt.subplots(figsize=(16, 6.5))

    # gc group separators
    for gi in range(1, len(GC_ORDER)):
        ax.axvline(gi * len(SW_ORDER) + 0.5, color="#aaa", lw=0.8, ls="--", alpha=0.6)

    rng = np.random.RandomState(0)
    used_stage9_label = False
    used_main_label = False
    used_perfect_label = False

    for idx, (gc, sw) in enumerate(cells, start=1):
        records = by_cell.get((gc, sw), [])
        if not records:
            continue
        for rec in records:
            err = (1.0 - rec["test_f1"]) * 100
            xj = idx + rng.normal(0, 0.09)
            is_stage9 = rec["seed"] in STAGE9_SEEDS
            is_perfect = abs(rec["test_f1"] - 1.0) < 1e-9
            if is_perfect:
                ax.scatter(xj, err, s=320, marker="*", color="#FFD700",
                           edgecolor="black", linewidth=1.2, zorder=5,
                           label=None if used_perfect_label else "Perfect (F1=1.0)")
                used_perfect_label = True
            elif is_stage9:
                ax.scatter(xj, err, s=90, marker="D", color="#E07A5F",
                           edgecolor="black", linewidth=0.7, alpha=0.9, zorder=4,
                           label=None if used_stage9_label else "Stage 9 seeds (5,6,7)")
                used_stage9_label = True
            else:
                ax.scatter(xj, err, s=55, color=GC_COLOR[gc],
                           edgecolor="black", linewidth=0.5, alpha=0.85, zorder=3,
                           label=None if used_main_label else "Main seeds (42,1,2,3,4)")
                used_main_label = True
        # mean bar
        errs_all = [(1.0 - rec["test_f1"]) * 100 for rec in records]
        m = float(np.mean(errs_all))
        ax.plot([idx - 0.32, idx + 0.32], [m, m], color="black", lw=2.2, zorder=6)

    x_labels = [f"gc={GC_DISP[gc]}\n{SW_DISP[sw]}" for gc, sw in cells]
    ax.set_xticks(range(1, len(cells) + 1))
    ax.set_xticklabels(x_labels, rotation=0, fontsize=7.5)
    ax.set_xlim(0.5, len(cells) + 0.5)
    ax.set_ylabel("Test error (1 - F1) x 100  [%]")
    ax.set_xlabel("Cell (gradient clip x val-smoothing)")
    n_total = sum(len(v) for v in by_cell.values())
    ax.set_title(f"vd080_bc per-cell test-error distribution — 20 cells, {n_total} runs total  (black bar = cell mean)",
                 fontsize=11.5)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    out = OUT / "vd080_bc_per_cell_dotplot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------
# 3. vd080_ax_wd_boxplot.png
# ------------------------------------------------------------
def plot_wd_boxplot(ax_rows):
    wd_keys = ["wd000", "wd001", "wd005"]
    wd_disp = {"wd000": "wd=0.0", "wd001": "wd=0.01", "wd005": "wd=0.05"}
    by_level = defaultdict(list)
    for r in ax_rows:
        if r["level"] in wd_keys:
            by_level[r["level"]].append(r)

    data = [[(1.0 - rec["test_f1"]) * 100 for rec in by_level[k]] for k in wd_keys]
    counts = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(9, 6))
    positions = list(range(1, len(wd_keys) + 1))
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                    showmeans=True,
                    medianprops=dict(color="black", lw=1.8),
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=7))
    palette = ["#4C72B0", "#DD8452", "#55A467"]
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")

    rng = np.random.RandomState(1)
    for i, (k, recs) in enumerate(zip(wd_keys, [by_level[k] for k in wd_keys]), start=1):
        for rec in recs:
            err = (1.0 - rec["test_f1"]) * 100
            jitter = rng.normal(0, 0.06)
            ax.scatter(i + jitter, err, s=70, color="black", alpha=0.7, zorder=3,
                       edgecolor="white", linewidth=0.6)
            if abs(rec["test_f1"] - 1.0) < 1e-9:
                ax.scatter(i + jitter, err, s=240, marker="*", color="#FFD700",
                           edgecolor="black", linewidth=1.1, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{wd_disp[k]}\n(n={counts[i]})" for i, k in enumerate(wd_keys)])
    ax.set_ylabel("Test error (1 - F1) x 100  [%]")
    ax.set_title(f"vd080_ax weight-decay sweep — {sum(counts)} runs  (gold star = F1=1.0)",
                 fontsize=11.5)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = OUT / "vd080_ax_wd_boxplot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------
# 4. vd080_bc_gc_pool.png
# ------------------------------------------------------------
def plot_gc_pool(bc_rows):
    by_gc = defaultdict(list)
    for r in bc_rows:
        by_gc[r["gc"]].append((1.0 - r["test_f1"]) * 100)

    data = [by_gc[g] for g in GC_ORDER]
    counts = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(9, 6))
    positions = list(range(1, len(GC_ORDER) + 1))
    parts = ax.violinplot(data, positions=positions, widths=0.75,
                          showmeans=False, showmedians=False, showextrema=False)
    for pc, g in zip(parts["bodies"], GC_ORDER):
        pc.set_facecolor(GC_COLOR[g])
        pc.set_alpha(0.55)
        pc.set_edgecolor("black")

    # per-group mean line + scatter dots
    rng = np.random.RandomState(2)
    for i, (g, vals) in enumerate(zip(GC_ORDER, data), start=1):
        if not vals:
            continue
        m = float(np.mean(vals))
        ax.plot([i - 0.28, i + 0.28], [m, m], color="black", lw=2.5, zorder=5)
        for v in vals:
            ax.scatter(i + rng.normal(0, 0.06), v, s=22, color="black",
                       alpha=0.5, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"gc={GC_DISP[g]}\n(n={counts[i]})" for i, g in enumerate(GC_ORDER)])
    ax.set_ylabel("Test error (1 - F1) x 100  [%]")
    ax.set_title(f"vd080_bc pooled by gradient clip — {sum(counts)} runs  (black bar = mean, violins = density)",
                 fontsize=11.5)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = OUT / "vd080_bc_gc_pool.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------
# 5. chronic_charts_bar.png
# ------------------------------------------------------------
def gather_chart_failures(rows):
    fn_counter = Counter()
    fp_counter = Counter()
    for r in rows:
        rd = r["run_dir"]
        fn_dir = rd / "predictions" / "fn_abnormal"
        fp_dir = rd / "predictions" / "fp_normal"
        if fn_dir.exists():
            for f in fn_dir.iterdir():
                m = CHART_RE.match(f.name)
                if m:
                    fn_counter[m.group(2)] += 1
        if fp_dir.exists():
            for f in fp_dir.iterdir():
                m = CHART_RE.match(f.name)
                if m:
                    fp_counter[m.group(2)] += 1
    return fn_counter, fp_counter


def plot_chronic_bar(fn_counter, fp_counter):
    combined = Counter()
    for k, v in fn_counter.items():
        combined[k] += v
    for k, v in fp_counter.items():
        combined[k] += v
    top = combined.most_common(10)
    if not top:
        print("chronic: no failures found, skipping")
        return
    chart_ids = [t[0] for t in top]
    fn_vals = [fn_counter[c] for c in chart_ids]
    fp_vals = [fp_counter[c] for c in chart_ids]
    totals = [fn + fp for fn, fp in zip(fn_vals, fp_vals)]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    x = np.arange(len(chart_ids))
    ax.bar(x, fn_vals, color="#C44E52", edgecolor="black", linewidth=0.6,
           label=f"FN (missed anomaly)  total={sum(fn_counter.values())}")
    ax.bar(x, fp_vals, bottom=fn_vals, color="#4C72B0", edgecolor="black",
           linewidth=0.6, label=f"FP (false alarm)  total={sum(fp_counter.values())}")

    for xi, tot, fn_v, fp_v in zip(x, totals, fn_vals, fp_vals):
        ax.text(xi, tot + 0.3, f"{tot}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(chart_ids, rotation=25, ha="right")
    ax.set_ylabel("Failure count across 122 runs")
    n_total_charts = len(combined)
    ax.set_title(f"Chronic charts — top-10 chart_ids by cross-run failure count  "
                 f"({n_total_charts} unique failing charts total)",
                 fontsize=11.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = OUT / "chronic_charts_bar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------
# 6. fn_fp_composition.png
# ------------------------------------------------------------
def plot_fn_fp_composition(rows):
    total_fn = 0
    total_fp = 0
    for r in rows:
        if r.get("fn") is not None:
            total_fn += r["fn"]
        if r.get("fp") is not None:
            total_fp += r["fp"]
    total = total_fn + total_fp
    if total == 0:
        print("fn_fp: no errors, skipping")
        return
    fn_pct = 100 * total_fn / total
    fp_pct = 100 * total_fp / total

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh([0], [fp_pct], color="#4C72B0", edgecolor="black", linewidth=0.6,
            label=f"FP (false alarm): {total_fp} errors ({fp_pct:.1f}%)")
    ax.barh([0], [fn_pct], left=[fp_pct], color="#C44E52", edgecolor="black",
            linewidth=0.6, label=f"FN (missed anomaly): {total_fn} errors ({fn_pct:.1f}%)")

    ax.text(fp_pct / 2, 0, f"FP\n{fp_pct:.1f}%\n({total_fp})",
            ha="center", va="center", color="white", fontweight="bold", fontsize=13)
    ax.text(fp_pct + fn_pct / 2, 0, f"FN\n{fn_pct:.1f}%\n({total_fn})",
            ha="center", va="center", color="white", fontweight="bold", fontsize=13)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("Share of total errors [%]")
    ax.set_title(f"Error composition across {len(rows)} runs  —  total errors: {total}",
                 fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=2, frameon=False, fontsize=10)
    fig.tight_layout()
    out = OUT / "fn_fp_composition.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (FP={fp_pct:.1f}%  FN={fn_pct:.1f}%)")


# ------------------------------------------------------------
# 7. n5_vs_n8_flip.png
# ------------------------------------------------------------
def plot_n5_vs_n8_flip(bc_rows):
    target_gc = "0p5"
    target_sw = "3mean"
    cell_rows = [r for r in bc_rows if r["gc"] == target_gc and r["sw"] == target_sw]
    main_seeds = [42, 1, 2, 3, 4]
    stage9 = [5, 6, 7]

    # Use FN+FP raw counts (matches "n=5 mu=2.60 -> n=8 mu=3.25" spec)
    n5 = []
    n8 = []
    for r in cell_rows:
        fn = r.get("fn") or 0
        fp = r.get("fp") or 0
        total = fn + fp
        if r["seed"] in main_seeds:
            n5.append((r["seed"], total))
        if r["seed"] in main_seeds or r["seed"] in stage9:
            n8.append((r["seed"], total))

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = [1, 2]
    groups = [("n=5 (main seeds 42,1,2,3,4)", n5, "#4C72B0"),
              ("n=8 (main + Stage-9 seeds 5,6,7)", n8, "#DD8452")]

    rng = np.random.RandomState(3)
    for xpos, (lbl, recs, color) in zip(positions, groups):
        vals = [e for _, e in recs]
        if not vals:
            continue
        m = float(np.mean(vals))
        sd = float(np.std(vals, ddof=0))
        ax.bar([xpos], [m], width=0.5, color=color, alpha=0.55,
               edgecolor="black", linewidth=0.8,
               yerr=[sd], capsize=7, error_kw={"lw": 1.6, "capthick": 1.6})
        for s, v in recs:
            xj = xpos + rng.normal(0, 0.05)
            marker = "D" if s in stage9 else "o"
            sz = 100 if s in stage9 else 70
            edge = "#8B0000" if s in stage9 else "black"
            ax.scatter(xj, v, s=sz, marker=marker, color=color, edgecolor=edge,
                       linewidth=0.9, alpha=0.95, zorder=4)
            ax.annotate(f"s{s}", (xj, v), xytext=(6, 4), textcoords="offset points",
                        fontsize=8.5, color="#333")
        ax.text(xpos, m + sd + 0.25, f"mu={m:.2f}\nsigma={sd:.2f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    if n5 and n8:
        m5 = float(np.mean([e for _, e in n5]))
        m8 = float(np.mean([e for _, e in n8]))
        ax.annotate("", xy=(2, m8), xytext=(1, m5),
                    arrowprops=dict(arrowstyle="->", color="#8B0000", lw=2.2,
                                    connectionstyle="arc3,rad=0.25"))
        delta = m8 - m5
        ax.text(1.5, (m5 + m8) / 2 + 0.5, f"delta mu = {delta:+.2f}",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="#8B0000",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8DC",
                          edgecolor="#8B0000", lw=1.2))

    ax.set_xticks(positions)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Total errors per run  (FN + FP out of 1500 test samples)")
    ax.set_title(f"Reproducibility failure at gc=0.5, sw3 mean — adding Stage-9 seeds flips the cell  "
                 f"(n=5 mu=2.60 -> n=8 mu=3.25, delta=+0.65)",
                 fontsize=11)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = OUT / "n5_vs_n8_flip.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (mu5={float(np.mean([e for _,e in n5])):.2f}  mu8={float(np.mean([e for _,e in n8])):.2f})")


# ------------------------------------------------------------
# 8. best_epoch_dist.png
# ------------------------------------------------------------
def plot_best_epoch_dist(rows):
    epochs = [r["epoch"] for r in rows if r.get("epoch") is not None]
    if not epochs:
        print("best_epoch: empty, skipping")
        return
    epochs_arr = np.array(epochs)
    mu = float(np.mean(epochs_arr))
    md = float(np.median(epochs_arr))

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(min(epochs) - 0.5, max(epochs) + 1.5, 1)
    n, _, _ = ax.hist(epochs_arr, bins=bins, color="#55A467", edgecolor="black",
                      linewidth=0.7, alpha=0.85)

    ax.axvline(mu, color="#C44E52", lw=2, ls="--", label=f"mean = {mu:.1f}")
    ax.axvline(md, color="#4C72B0", lw=2, ls=":", label=f"median = {md:.0f}")

    # count on each bar
    for i, count in enumerate(n):
        if count > 0:
            ax.text(bins[i] + 0.5, count + 0.5, f"{int(count)}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Best epoch (NEW_BEST saved)")
    ax.set_ylabel("Number of runs")
    ax.set_title(f"Best-epoch distribution across N={len(epochs)} runs  "
                 f"(109 vd080_bc + 13 vd080_ax)",
                 fontsize=11.5)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = OUT / "best_epoch_dist.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (mean={mu:.2f}, median={md:.1f}, n={len(epochs)})")


# ------------------------------------------------------------
# Δ F1 report helper
# ------------------------------------------------------------
def report_deltas(bc_rows):
    """Report per-cell mu F1 (current N=122 data) vs prior-N=95 checkpoint if available.

    We approximate by taking per-cell mean F1 at 5-seed boundary vs current seed count.
    """
    by_cell = defaultdict(list)
    for r in bc_rows:
        by_cell[(r["gc"], r["sw"])].append((r["seed"], r["test_f1"]))
    # Compare "n=5 main only" vs "all seeds" per cell
    print("\nPer-cell F1 deltas (main 5 seeds vs. all seeds):")
    print(f"{'cell':<22}  {'n_all':>5}  {'mu5':>8}  {'muall':>8}  {'delta':>8}")
    for (gc, sw), recs in sorted(by_cell.items()):
        main = [f1 for s, f1 in recs if s in (42, 1, 2, 3, 4)]
        allv = [f1 for _, f1 in recs]
        if len(main) < 3 or len(allv) == len(main):
            continue
        m5 = float(np.mean(main))
        mall = float(np.mean(allv))
        delta = mall - m5
        if abs(delta) >= 0.002:
            cell = f"gc={GC_DISP[gc]} {SW_DISP[sw]}"
            print(f"{cell:<22}  {len(allv):>5}  {m5:>8.4f}  {mall:>8.4f}  {delta:>+8.4f}  *")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    bc_rows = collect_vd080_bc()
    ax_rows = collect_vd080_ax()
    print(f"collected: vd080_bc={len(bc_rows)}, vd080_ax={len(ax_rows)}, total={len(bc_rows)+len(ax_rows)}")

    if not bc_rows:
        print("no vd080_bc data — aborting")
        return

    plot_20cells_heatmap(bc_rows)
    plot_per_cell_dotplot(bc_rows)
    plot_wd_boxplot(ax_rows)
    plot_gc_pool(bc_rows)

    all_rows = bc_rows + ax_rows
    fn_counter, fp_counter = gather_chart_failures(all_rows)
    plot_chronic_bar(fn_counter, fp_counter)
    plot_fn_fp_composition(all_rows)

    plot_n5_vs_n8_flip(bc_rows)
    plot_best_epoch_dist(all_rows)

    report_deltas(bc_rows)
    print("done.")


if __name__ == "__main__":
    main()
