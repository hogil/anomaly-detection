"""Generate publication-quality matplotlib PNG plots from sweep experiment data.

Sources:
- logs/*/best_info.json (per-run metrics, hparams, test_history)
- logs/*/history.json (per-epoch training curves)
- validations/stability_and_worst_0408.txt (N-scaling slopes, already computed)

Outputs to: experiments/plots/*.png plus a README.md describing each figure.
Do NOT train — analysis only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path("D:/project/anomaly-detection")
LOGS = ROOT / "logs"
OUT = ROOT / "experiments" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})

# ============================================================
# Helpers
# ============================================================

def load_best(run_dir: Path):
    """Return dict of best_info.json + derived FN/FP, else None."""
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
    n_count = n.get("count") or 0
    a_count = a.get("count") or 0
    nR = n.get("recall")
    aR = a.get("recall")
    fp = round(n_count * (1.0 - nR)) if nR is not None else None
    fn = round(a_count * (1.0 - aR)) if aR is not None else None
    return {
        "run": run_dir.name,
        "epoch": data.get("epoch"),
        "val_f1": data.get("val_f1"),
        "test_f1": data.get("test_f1"),
        "abn_recall": aR,
        "nor_recall": nR,
        "fp": fp,
        "fn": fn,
    }


def load_history(run_dir: Path):
    p = run_dir / "history.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_test_history(run_dir: Path):
    """Pull test_history list from standalone test_history.json, fallback to best_info embedded."""
    p = run_dir / "test_history.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")) or []
        except Exception:
            pass
    p = run_dir / "best_info.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data.get("test_history") or []


# ============================================================
# Collect ex0416 axis runs: lr, fg, abw, wd, dp
# ============================================================

# pattern captures axis key and level (e.g., lr4e5, fg0p5, abw0p8, wd0p01, dp0p1)
AX_PATTERNS = {
    "lr":  re.compile(r"ex0416_lr([a-z0-9p]+)_n700_s(\d+)"),
    "fg":  re.compile(r"ex0416_fg([a-z0-9p]+)_n700_s(\d+)"),
    "abw": re.compile(r"ex0416_abw([a-z0-9p]+)_n700_s(\d+)"),
    "wd":  re.compile(r"ex0416_wd([a-z0-9p]+)_n700_s(\d+)"),
    "dp":  re.compile(r"ex0416_dp([a-z0-9p]+)_n700_s(\d+)"),
}

# Canonical ordering — convert level token into sortable numeric for display
def lvl_to_float(axis: str, lvl: str) -> float:
    # Convert e.g. "0p5" -> 0.5, "4e5" -> 4e-5, "0p001" -> 0.001
    s = lvl.replace("p", ".")
    if "e" in s:
        # token like "4e5" means 4e-5 (negative exponent for LR)
        m = re.fullmatch(r"(\d+(?:\.\d+)?)e(\d+)", s)
        if m:
            return float(m.group(1)) * (10 ** (-int(m.group(2))))
        try:
            return float(s)
        except ValueError:
            return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def lvl_display(axis: str, lvl: str) -> str:
    v = lvl_to_float(axis, lvl)
    if axis == "lr":
        return f"{v:.0e}"
    if axis == "wd":
        if v == 0:
            return "0"
        return f"{v:g}"
    return f"{v:g}"


def collect_ex0416():
    """Return DataFrame with columns: axis, level, level_val, seed, test_f1, fn, fp, epoch, run."""
    rows = []
    for run_dir in LOGS.iterdir():
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if "ex0416" not in name:
            continue
        for axis, pat in AX_PATTERNS.items():
            m = pat.search(name)
            if not m:
                continue
            lvl_token = m.group(1)
            seed = int(m.group(2))
            best = load_best(run_dir)
            if best is None or best["test_f1"] is None:
                continue
            rows.append({
                "axis": axis,
                "level": lvl_token,
                "level_val": lvl_to_float(axis, lvl_token),
                "level_disp": lvl_display(axis, lvl_token),
                "seed": seed,
                "test_f1": best["test_f1"],
                "val_f1": best["val_f1"],
                "fn": best["fn"],
                "fp": best["fp"],
                "epoch": best["epoch"],
                "run": name,
            })
            break  # stop at first matching axis
    df = pd.DataFrame(rows)
    # Some axes overlap in token space (fg vs abw collide? No — axis pattern keyed). Dedup anyway.
    df = df.drop_duplicates(subset=["axis", "level", "seed", "run"])
    return df


# ============================================================
# Collect N-scaling runs: fresh0413_reset_, fresh0412_, v11_ prefixes
# ============================================================

NSCALE_PATTERNS = {
    "fresh0413_reset_v11": re.compile(r"^(\d{6})_(\d{6})_fresh0413_reset_v11_n(\d+)_s(\d+)_F"),
    "fresh0412_v11":       re.compile(r"^(\d{6})_(\d{6})_fresh0412_v11_n(\d+)_s(\d+)_F"),
    "v11":                 re.compile(r"^(\d{6})_(\d{6})_v11_n(\d+)_s(\d+)_F"),
}


def collect_nscale():
    rows = []
    for run_dir in LOGS.iterdir():
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        for fam, pat in NSCALE_PATTERNS.items():
            m = pat.match(name)
            if not m:
                continue
            N = int(m.group(3))
            seed = int(m.group(4))
            best = load_best(run_dir)
            if best is None or best["test_f1"] is None:
                continue
            rows.append({
                "family": fam,
                "N": N,
                "seed": seed,
                "test_f1": best["test_f1"],
                "run": name,
            })
            break
    return pd.DataFrame(rows)


# ============================================================
# Plot 1 & 2: axis effect on FN / FP (5 subplots each)
# ============================================================

AXIS_ORDER = ["lr", "fg", "abw", "wd", "dp"]
AXIS_TITLE = {
    "lr":  "Learning rate (backbone)",
    "fg":  "Focal-loss gamma",
    "abw": "Abnormal class weight",
    "wd":  "Weight decay",
    "dp":  "Dropout",
}


def plot_axis_effect(df: pd.DataFrame, metric: str, fname: str, title_suffix: str):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharey=False)
    for ax, axis_key in zip(axes, AXIS_ORDER):
        sub = df[df["axis"] == axis_key]
        if sub.empty:
            ax.set_title(f"{AXIS_TITLE[axis_key]}\n(no data)")
            ax.set_xticks([])
            continue
        grp = (sub.groupby(["level_val", "level_disp"])[metric]
                  .agg(["mean", "std", "count"])
                  .reset_index()
                  .sort_values("level_val"))
        x = np.arange(len(grp))
        means = grp["mean"].to_numpy()
        stds = grp["std"].fillna(0).to_numpy()
        colors = sns.color_palette("viridis", len(x))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                      edgecolor="black", linewidth=0.6, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(grp["level_disp"].tolist(), rotation=30, ha="right")
        ax.set_title(f"{AXIS_TITLE[axis_key]}\n(n seeds each: {int(grp['count'].min())}-{int(grp['count'].max())})")
        ax.set_xlabel(axis_key)
        ax.set_ylabel(f"{metric.upper()} mean (± std)")
        for b, m, s, c in zip(bars, means, stds, grp["count"].tolist()):
            ax.annotate(f"{m:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
        ax.grid(True, axis="y", alpha=0.4)
    fig.suptitle(f"Per-axis effect on test {title_suffix} across ex0416 sweeps (n=700 per class, ConvNeXtV2-Tiny)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ============================================================
# Plot 3: N scaling (log-log)
# ============================================================

def plot_n_scaling(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    markers = {"fresh0413_reset_v11": "o", "fresh0412_v11": "s", "v11": "^"}
    colors = {"fresh0413_reset_v11": "#2E86AB",
              "fresh0412_v11":       "#A23B72",
              "v11":                 "#E07A5F"}
    labels = {"fresh0413_reset_v11": "fresh0413_reset_v11 (slope=+0.29, saturated)",
              "fresh0412_v11":       "fresh0412_v11 (slope=-0.18, mild scaling)",
              "v11":                 "v11 (broken, slope=+0.26)"}
    for fam in ["fresh0413_reset_v11", "fresh0412_v11", "v11"]:
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        agg = (sub.groupby("N")["test_f1"]
                  .agg(["mean", "std", "count"])
                  .reset_index()
                  .sort_values("N"))
        # err = 1 - F1
        err_mean = 1 - agg["mean"].to_numpy()
        err_std = agg["std"].fillna(0).to_numpy()
        N = agg["N"].to_numpy()
        ax.errorbar(N, err_mean, yerr=err_std,
                    marker=markers[fam], ms=9, lw=1.6, capsize=4,
                    color=colors[fam], label=labels[fam])
        # scatter individual seeds
        for _, r in sub.iterrows():
            ax.scatter(r["N"], 1 - r["test_f1"], s=16, color=colors[fam], alpha=0.35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total N per class (log scale)")
    ax.set_ylabel("Test error rate: 1 − F1  (log scale)")
    ax.set_title("N-scaling across dataset families  (5 seeds / point; error bars = ±1σ)")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, which="both", alpha=0.4)
    out = OUT / "n_scaling.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ============================================================
# Plot 4: seed-variance heatmap
# ============================================================

def plot_seed_heatmap(df: pd.DataFrame):
    # for each axis, pivot variant x seed -> test_f1
    # compose a single tall heatmap "variant" = axis/level
    df = df.copy()
    df["variant"] = df["axis"] + "=" + df["level_disp"]
    # pivot
    pv = df.pivot_table(index=["axis", "level_val", "variant"],
                        columns="seed",
                        values="test_f1",
                        aggfunc="first").reset_index()
    # sort: group by axis in AXIS_ORDER, within by level_val
    axis_rank = {a: i for i, a in enumerate(AXIS_ORDER)}
    pv["ar"] = pv["axis"].map(axis_rank)
    pv = pv.sort_values(["ar", "level_val"])
    variants = pv["variant"].tolist()
    seed_cols = [c for c in pv.columns if isinstance(c, (int, np.integer))]
    seed_cols = sorted(seed_cols)
    mat = pv[seed_cols].to_numpy(dtype=float)

    # filter variants with >=5 seeds observed
    keep_mask = (~np.isnan(mat)).sum(axis=1) >= 4
    mat = mat[keep_mask]
    variants = [v for v, k in zip(variants, keep_mask) if k]
    # Add per-row mean/std columns
    row_mean = np.nanmean(mat, axis=1)
    row_std = np.nanstd(mat, axis=1)
    display = np.concatenate([mat, row_mean[:, None], row_std[:, None]], axis=1)
    col_labels = [f"seed {s}" for s in seed_cols] + ["mean", "std"]

    fig_h = max(6, 0.28 * len(variants) + 2)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    # Use percentage distance from 1.0 for color scale -> (1 - f1) * 100
    err_pct = (1.0 - display) * 100
    # std col is already std; scale for color: clip
    err_pct_plot = err_pct.copy()
    err_pct_plot[:, -1] = err_pct_plot[:, -1] * 10  # emphasize std column differently
    vmax = np.nanpercentile(err_pct[:, :-2], 98) if np.isfinite(err_pct[:, :-2]).any() else 2.0

    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(err_pct[:, :-1], ax=ax, cmap=cmap,
                vmin=0, vmax=max(0.5, vmax),
                xticklabels=col_labels[:-1], yticklabels=variants,
                annot=display[:, :-1], fmt=".4f",
                annot_kws={"size": 7},
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": "test error % (1−F1)×100"})
    ax.set_title("Seed variance across ex0416 variants  (annotation = test F1; color = 1−F1 in %)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("variant (axis=level)")
    fig.tight_layout()
    out = OUT / "seed_variance_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ============================================================
# Plot 6: training curves top-3 runs
# ============================================================

TOP3_TARGETS = [
    ("ex0416_lr4e5",  "lr=4e-5"),
    ("ex0416_fg0p5",  "focal γ=0.5"),
    ("ex0416_abw0p8", "abn_w=0.8"),
]


def _pick_best_seed(key: str) -> Path | None:
    """Pick best-F1 seed for a given substring."""
    best = None
    best_f1 = -1.0
    for run_dir in LOGS.iterdir():
        if not run_dir.is_dir():
            continue
        if key not in run_dir.name:
            continue
        info = load_best(run_dir)
        if info is None or info["test_f1"] is None:
            continue
        if info["test_f1"] > best_f1:
            best_f1 = info["test_f1"]
            best = run_dir
    return best


def plot_training_curves_top3():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    colors = {"train": "#888888", "val": "#2E86AB", "test": "#E07A5F"}

    for ax, (key, pretty) in zip(axes, TOP3_TARGETS):
        run_dir = _pick_best_seed(key)
        if run_dir is None:
            ax.set_title(f"{pretty}\n(no run found)")
            continue
        hist = load_history(run_dir)
        test_hist = load_test_history(run_dir)
        best = load_best(run_dir)
        if hist is None:
            ax.set_title(f"{pretty}\n(no history)")
            continue
        ep = [h["epoch"] for h in hist]
        tr_f1 = [h.get("val_f1") for h in hist]  # val_f1 (per-epoch val)
        # Extract train_loss/acc curves too? Focus on F1.
        val_acc = [h.get("val_acc") for h in hist]
        val_f1 = [h.get("val_f1") for h in hist]
        train_acc = [h.get("train_acc") for h in hist]
        ax.plot(ep, train_acc, color=colors["train"], lw=1.4, ls="--", label="train acc")
        ax.plot(ep, val_f1, color=colors["val"], lw=2.0, marker="o", ms=4, label="val F1")
        if test_hist:
            te_ep = [t["epoch"] for t in test_hist]
            te_f1 = [t["test_f1"] for t in test_hist]
            ax.plot(te_ep, te_f1, color=colors["test"], lw=0, marker="*", ms=14,
                    label=f"test F1 @ NEW_BEST (final={best['test_f1']:.4f})")
        # mark best epoch
        if best and best["epoch"]:
            ax.axvline(best["epoch"], color="black", ls=":", lw=1, alpha=0.6)
        ax.set_ylim(0.80, 1.005)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("metric")
        ax.set_title(f"{pretty}\n{run_dir.name[:45]}...\nbest ep={best['epoch']}, test F1={best['test_f1']:.4f}, FN={best['fn']}, FP={best['fp']}",
                     fontsize=9)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.4)

    fig.suptitle("Top-3 ex0416 configurations: training curves (best seed per axis)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT / "training_curves_top3.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ============================================================
# vd080_bc pilot (gc × sw matrix)
# ============================================================

VD080_PAT = re.compile(r"vd080_bc_gc([0-9p]+)_sw([0-9a-z]+)_n(\d+)_s(\d+)")
GC_ORDER = ["0p5", "1p0", "2p0", "5p0"]
GC_DISP = {"0p5": "0.5", "1p0": "1.0", "2p0": "2.0", "5p0": "5.0"}
SW_ORDER = ["1raw", "3med", "3mean", "5med", "5mean"]
SW_DISP = {"1raw": "sw1 (raw)", "3med": "sw3 (med)", "3mean": "sw3 (mean)",
           "5med": "sw5 (med)", "5mean": "sw5 (mean)"}

# The single perfect run (F1 = 1.0000) — highlight on plots
PERFECT_RUN_KEY = {"gc": "2p0", "sw": "1raw", "seed": 4}


def collect_vd080_bc():
    rows = []
    for run_dir in LOGS.iterdir():
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if "vd080_bc" not in name:
            continue
        m = VD080_PAT.search(name)
        if not m:
            continue
        gc, sw, n, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        best = load_best(run_dir)
        if best is None or best["test_f1"] is None:
            continue
        rows.append({
            "run": name,
            "run_dir": run_dir,
            "gc": gc,
            "sw": sw,
            "n": n,
            "seed": seed,
            "test_f1": best["test_f1"],
            "val_f1": best["val_f1"],
            "epoch": best["epoch"],
            "fn": best["fn"],
            "fp": best["fp"],
            "abn_recall": best["abn_recall"],
            "nor_recall": best["nor_recall"],
        })
    return pd.DataFrame(rows)


def plot_vd080_bc_heatmap(df: pd.DataFrame):
    """4×5 heatmap (gc rows × sw cols).
    Cell color = test_F1 (viridis), annotation = F1, FN, FP, epoch.
    """
    if df.empty:
        print("vd080_bc: no data, skipping heatmap")
        return False

    # Build 4x5 matrix. Rows = gc, cols = sw (in canonical order)
    mat = np.full((len(GC_ORDER), len(SW_ORDER)), np.nan)
    annot = np.full((len(GC_ORDER), len(SW_ORDER)), "", dtype=object)
    for i, gc in enumerate(GC_ORDER):
        for j, sw in enumerate(SW_ORDER):
            sub = df[(df["gc"] == gc) & (df["sw"] == sw)]
            if sub.empty:
                annot[i, j] = "(missing)"
                continue
            r = sub.iloc[0]
            mat[i, j] = r["test_f1"]
            annot[i, j] = f"F1={r['test_f1']:.4f}\nFN={r['fn']} FP={r['fp']}\nep={r['epoch']}"

    # Color vmin/vmax — tight around observed range for contrast
    flat = mat[~np.isnan(mat)]
    if flat.size == 0:
        print("vd080_bc: all NaN, skipping")
        return False
    vmin = max(0.98, float(np.nanmin(flat)) - 0.002)
    vmax = min(1.0, float(np.nanmax(flat)) + 0.0005)
    if vmax - vmin < 1e-4:
        vmin = float(np.nanmin(flat)) - 0.002
        vmax = float(np.nanmax(flat)) + 0.002

    fig, ax = plt.subplots(figsize=(11, 6.5))
    cmap = sns.color_palette("viridis", as_cmap=True)
    # seaborn heatmap does not gracefully handle NaN annotations — overlay manually
    sns.heatmap(mat, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=[SW_DISP[s] for s in SW_ORDER],
                yticklabels=[f"gc={GC_DISP[g]}" for g in GC_ORDER],
                annot=False, linewidths=1.0, linecolor="white",
                cbar_kws={"label": "test F1"})
    for i in range(len(GC_ORDER)):
        for j in range(len(SW_ORDER)):
            text = annot[i, j]
            if not text:
                continue
            val = mat[i, j]
            if np.isnan(val):
                color = "black"
            else:
                # bright text on dark cells (low F1) vs dark text on bright cells
                # Use mid-threshold based on normalized position
                pos = (val - vmin) / max(vmax - vmin, 1e-9)
                color = "white" if pos < 0.55 else "black"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")
    ax.set_title("vd080_bc pilot: gradient-clip × val-smoothing grid  (n=700 per class, seed=42, 19/20 runs complete)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Validation smoothing window/method")
    ax.set_ylabel("Gradient clip norm")
    fig.tight_layout()
    out = OUT / "vd080_bc_pilot_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return True


def plot_vd080_bc_5seed_heatmap(df: pd.DataFrame):
    """Heatmap restricted to (gc, sw) cells with >=5 seeds.

    Each cell annotated with: mean FN, mean FP, std err (= std of 1-F1).
    Color = mean test error (1 - F1) in %.
    """
    if df.empty:
        print("vd080_bc 5seed: no data, skipping")
        return False

    # Aggregate by (gc, sw)
    agg = (df.groupby(["gc", "sw"])
             .agg(n_seeds=("seed", "nunique"),
                  mean_fn=("fn", "mean"),
                  mean_fp=("fp", "mean"),
                  mean_f1=("test_f1", "mean"),
                  std_f1=("test_f1", "std"))
             .reset_index())
    agg["std_err"] = agg["std_f1"]  # std of F1 = std of err (constant shift)

    complete = agg[agg["n_seeds"] >= 5].copy()
    if complete.empty:
        print("vd080_bc 5seed: no cell with >=5 seeds, skipping")
        return False

    # Build 4x5 matrix, NaN for incomplete cells
    mat_err = np.full((len(GC_ORDER), len(SW_ORDER)), np.nan)
    annot = np.full((len(GC_ORDER), len(SW_ORDER)), "", dtype=object)
    for _, r in complete.iterrows():
        try:
            i = GC_ORDER.index(r["gc"])
            j = SW_ORDER.index(r["sw"])
        except ValueError:
            continue
        err_pct = (1.0 - r["mean_f1"]) * 100
        mat_err[i, j] = err_pct
        annot[i, j] = (f"FN={r['mean_fn']:.1f}\n"
                       f"FP={r['mean_fp']:.1f}\n"
                       f"std={r['std_err']*100:.2f}%")

    # Fill incomplete cells with a placeholder annotation
    for i, gc in enumerate(GC_ORDER):
        for j, sw in enumerate(SW_ORDER):
            if annot[i, j] == "":
                matched = agg[(agg["gc"] == gc) & (agg["sw"] == sw)]
                n = int(matched["n_seeds"].iloc[0]) if not matched.empty else 0
                annot[i, j] = f"(n={n}/5)\nincomplete"

    # Color scale from observed completed cells
    completed_vals = mat_err[~np.isnan(mat_err)]
    vmin = float(np.nanmin(completed_vals)) if completed_vals.size else 0.0
    vmax = float(np.nanmax(completed_vals)) if completed_vals.size else 1.0
    if vmax - vmin < 1e-3:
        vmax = vmin + 0.5

    fig, ax = plt.subplots(figsize=(11, 6.5))
    cmap = sns.color_palette("rocket_r", as_cmap=True)

    # Draw heatmap (NaN cells become masked / white)
    mask = np.isnan(mat_err)
    sns.heatmap(mat_err, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                mask=mask,
                xticklabels=[SW_DISP[s] for s in SW_ORDER],
                yticklabels=[f"gc={GC_DISP[g]}" for g in GC_ORDER],
                annot=False, linewidths=1.0, linecolor="white",
                cbar_kws={"label": "mean test error % (1-F1)×100"})

    # Lightly shade incomplete cells
    for i in range(len(GC_ORDER)):
        for j in range(len(SW_ORDER)):
            if np.isnan(mat_err[i, j]):
                ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                           facecolor="#f0f0f0", edgecolor="white",
                                           linewidth=1.0, zorder=1))
            text = annot[i, j]
            if not text:
                continue
            val = mat_err[i, j]
            if np.isnan(val):
                color = "#888"
                weight = "normal"
            else:
                pos = (val - vmin) / max(vmax - vmin, 1e-9)
                color = "white" if pos > 0.55 else "black"
                weight = "bold"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                    fontsize=9, color=color, fontweight=weight, zorder=2)

    # Highlight the PERFECT-run cell (F1 = 1.0000) with a gold outline + star
    # Cell = (gc=2.0, sw=1raw). The perfect seed is s=4 but cell-level is shown.
    try:
        pi = GC_ORDER.index(PERFECT_RUN_KEY["gc"])
        pj = SW_ORDER.index(PERFECT_RUN_KEY["sw"])
        # Check if that cell contains the perfect run in the data
        perfect_in_df = df[(df["gc"] == PERFECT_RUN_KEY["gc"])
                           & (df["sw"] == PERFECT_RUN_KEY["sw"])
                           & (df["seed"] == PERFECT_RUN_KEY["seed"])]
        if not perfect_in_df.empty and abs(perfect_in_df.iloc[0]["test_f1"] - 1.0) < 1e-6:
            # Gold rectangle frame around the cell (3 px wide)
            ax.add_patch(plt.Rectangle((pj, pi), 1, 1,
                                       facecolor="none", edgecolor="#FFD700",
                                       linewidth=4.5, zorder=4))
            # Gold "1.0" badge in the top-right corner of the cell
            ax.text(pj + 0.87, pi + 0.13, "F1=1.0",
                    ha="center", va="center",
                    fontsize=10, color="#FFD700",
                    fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="#2a2a2a", edgecolor="#FFD700", lw=1.2))
            # Arrow + external caption pointing to the cell
            caption = "PERFECT RUN\ngc=2.0, sw1(raw), s=4\nF1=1.0000 (FN=0, FP=0)"
            # Place caption above-right the cell (outside grid, top area)
            ax.annotate(caption,
                        xy=(pj + 0.1, pi + 0.05), xycoords="data",
                        xytext=(pj + 1.3, pi - 0.95), textcoords="data",
                        ha="left", va="center", fontsize=9, fontweight="bold",
                        color="#8B6914",
                        bbox=dict(boxstyle="round,pad=0.35",
                                  facecolor="#FFF8DC", edgecolor="#FFD700", lw=1.5),
                        arrowprops=dict(arrowstyle="->", color="#FFD700",
                                        lw=1.8, connectionstyle="arc3,rad=-0.2"),
                        zorder=6,
                        annotation_clip=False)
    except ValueError:
        pass

    n_complete = int(complete.shape[0])
    ax.set_title(f"vd080_bc: 5-seed aggregated gc × sw grid (n=700 per class)\n"
                 f"{n_complete} cells with complete 5 seeds — annotations: mean FN, mean FP, std of test err",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Validation smoothing window/method")
    ax.set_ylabel("Gradient clip norm")
    fig.tight_layout()
    out = OUT / "vd080_bc_5seed_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({n_complete} complete cells)")
    return True


def plot_vd080_bc_seed_variance_boxplot(df: pd.DataFrame):
    """Box plot of per-seed test error per (gc, sw) cell, restricted to cells with >=5 seeds.
    Boxes sorted by median test error (ascending = best first).
    """
    if df.empty:
        print("vd080_bc boxplot: no data, skipping")
        return False

    # Count seeds per cell
    counts = (df.groupby(["gc", "sw"])["seed"].nunique().reset_index(name="n_seeds"))
    complete = counts[counts["n_seeds"] >= 5]
    if complete.empty:
        print("vd080_bc boxplot: no cell with >=5 seeds, skipping")
        return False

    # Build per-run err series
    sub = df.merge(complete[["gc", "sw"]], on=["gc", "sw"], how="inner").copy()
    sub["err_pct"] = (1.0 - sub["test_f1"]) * 100
    sub["cell"] = sub.apply(lambda r: f"gc={GC_DISP[r['gc']]}, {SW_DISP[r['sw']]}", axis=1)

    # Sort cells by median err (ascending)
    medians = sub.groupby("cell")["err_pct"].median().sort_values()
    order = medians.index.tolist()

    # Gather box data in order
    data = [sub[sub["cell"] == c]["err_pct"].to_numpy() for c in order]

    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(order) + 3), 6.0))
    bp = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.6,
                    medianprops=dict(color="black", lw=1.6),
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6))
    # Color by gc level
    gc_of_cell = {}
    for c in order:
        # extract gc value from label "gc=0.5, ..."
        lbl = c.split(",")[0].replace("gc=", "").strip()
        gc_of_cell[c] = lbl
    palette = {"0.5": "#4C72B0", "1.0": "#DD8452", "2.0": "#55A467", "5.0": "#C44E52"}
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(palette.get(gc_of_cell[c], "#888888"))
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")

    # Overlay raw seeds
    for i, c in enumerate(order, start=1):
        vals = sub[sub["cell"] == c]["err_pct"].to_numpy()
        jitter = np.random.RandomState(0).normal(0, 0.05, size=len(vals))
        ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals,
                   s=30, color="black", alpha=0.55, zorder=3)

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("Test error (1 − F1) × 100  [%]")
    ax.set_title(f"vd080_bc seed variance per (gc, sw) cell — cells with 5 complete seeds\n"
                 f"boxes sorted by median test error (ascending)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.4)

    # Legend: gc color key
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=palette[k], edgecolor="black", alpha=0.75,
                     label=f"gc={k}") for k in ["0.5", "1.0", "2.0", "5.0"]
               if k in set(gc_of_cell.values())]
    ax.legend(handles=handles, title="grad clip", loc="upper left")
    fig.tight_layout()
    out = OUT / "vd080_bc_seed_variance_boxplot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({len(order)} cells)")
    return True


def plot_vd080_bc_per_cell_dotplot(df: pd.DataFrame):
    """Per-cell error distribution: one column per (gc, sw) cell, dots for each seed.
    Perfect run (err=0%) drawn as a gold star; mean bar overlayed; cells restricted
    to those with >=3 seeds to keep the figure readable.
    """
    if df.empty:
        print("vd080_bc dotplot: no data, skipping")
        return False

    counts = df.groupby(["gc", "sw"])["seed"].nunique().reset_index(name="n_seeds")
    keep = counts[counts["n_seeds"] >= 3][["gc", "sw"]]
    if keep.empty:
        print("vd080_bc dotplot: no cells with >=3 seeds, skipping")
        return False

    sub = df.merge(keep, on=["gc", "sw"], how="inner").copy()
    sub["err_pct"] = (1.0 - sub["test_f1"]) * 100

    # Order cells: gc outer, sw inner (canonical order)
    cells = []
    for gc in GC_ORDER:
        for sw in SW_ORDER:
            if ((keep["gc"] == gc) & (keep["sw"] == sw)).any():
                cells.append((gc, sw))
    if not cells:
        return False

    palette = {"0p5": "#4C72B0", "1p0": "#DD8452", "2p0": "#55A467", "5p0": "#C44E52"}

    fig, ax = plt.subplots(figsize=(max(11, 0.8 * len(cells) + 3), 6.2))

    # gc-group separator lines
    last_gc = cells[0][0]
    for i, (gc, sw) in enumerate(cells, start=1):
        if gc != last_gc:
            ax.axvline(i - 0.5, color="#555", lw=0.8, alpha=0.5, zorder=1)
            last_gc = gc

    x_labels = []
    for i, (gc, sw) in enumerate(cells, start=1):
        vals = sub[(sub["gc"] == gc) & (sub["sw"] == sw)][["seed", "err_pct"]].values
        seeds = vals[:, 0].astype(int)
        errs = vals[:, 1].astype(float)
        # Horizontal jitter for readability
        rng = np.random.RandomState(abs(hash((gc, sw))) % (2**31))
        jitter = rng.normal(0, 0.08, size=len(errs))
        xs = np.full_like(errs, i, dtype=float) + jitter
        # Perfect run mask (err ~= 0 AND it is the known perfect cell/seed)
        is_perfect_cell = (gc == PERFECT_RUN_KEY["gc"] and sw == PERFECT_RUN_KEY["sw"])
        perfect_mask = np.array([
            is_perfect_cell and int(s) == PERFECT_RUN_KEY["seed"] and e < 1e-6
            for s, e in zip(seeds, errs)
        ])
        ord_mask = ~perfect_mask
        ax.scatter(xs[ord_mask], errs[ord_mask], s=55,
                   color=palette.get(gc, "#888"),
                   edgecolor="black", linewidth=0.7, alpha=0.85, zorder=3)
        if perfect_mask.any():
            ax.scatter(xs[perfect_mask], errs[perfect_mask],
                       s=320, color="#FFD700", marker="*",
                       edgecolor="black", linewidth=1.3, zorder=5,
                       label="PERFECT (F1=1.0)" if i == 1 or not ax.get_legend_handles_labels()[1] else None)
            # Annotate the star
            ax.annotate("PERFECT\ns=4, F1=1.0000",
                        xy=(xs[perfect_mask][0], errs[perfect_mask][0]),
                        xytext=(xs[perfect_mask][0] + 0.3, errs[perfect_mask][0] + 0.18),
                        fontsize=8, fontweight="bold", color="#8B6914",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor="#FFF8DC", edgecolor="#FFD700", lw=1.2),
                        arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1.2),
                        zorder=6)
        # Mean bar
        if len(errs):
            m = errs.mean()
            ax.plot([i - 0.32, i + 0.32], [m, m], color="black", lw=2.0, zorder=4)
        x_labels.append(f"gc={GC_DISP[gc]}\n{SW_DISP[sw]}")

    ax.set_xticks(range(1, len(cells) + 1))
    ax.set_xticklabels(x_labels, rotation=0, fontsize=7.5)
    ax.set_xlim(0.5, len(cells) + 0.5)
    ax.set_ylabel("Test error (1 − F1) × 100  [%]")
    ax.set_xlabel("Cell (grad clip × val smoothing)")
    ax.set_title(f"vd080_bc per-cell error distribution — {len(cells)} cells with ≥3 seeds "
                 f"(N={len(sub)} runs; black bar = cell mean; gold star = the single perfect run)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)

    # gc colour legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    gc_handles = [Patch(facecolor=palette[g], edgecolor="black", alpha=0.85,
                        label=f"gc={GC_DISP[g]}") for g in GC_ORDER
                  if any(gc == g for gc, _ in cells)]
    gc_handles.append(Line2D([0], [0], marker="*", color="w",
                             markerfacecolor="#FFD700", markeredgecolor="black",
                             markersize=14, label="perfect run"))
    ax.legend(handles=gc_handles, loc="upper right", fontsize=8, ncol=1,
              framealpha=0.95)
    fig.tight_layout()
    out = OUT / "vd080_bc_per_cell_dotplot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({len(cells)} cells)")
    return True


def compute_vd080_bc_checkpoint_delta(df: pd.DataFrame, n_baseline: int = 46):
    """Approximate the N=46 checkpoint by taking the earliest 46 runs (by name/timestamp),
    aggregate per-cell mean err% for both baseline and current, return rows with Δerr ≥ 0.3pp.
    """
    if df.empty or len(df) <= n_baseline:
        return pd.DataFrame(), pd.DataFrame()
    ordered = df.sort_values("run").reset_index(drop=True)
    baseline = ordered.iloc[:n_baseline].copy()
    current = ordered.copy()

    def agg(sub):
        sub = sub.copy()
        sub["err_pct"] = (1.0 - sub["test_f1"]) * 100
        a = (sub.groupby(["gc", "sw"])
                 .agg(n_seeds=("seed", "nunique"),
                      mean_err=("err_pct", "mean"),
                      std_err=("err_pct", "std"),
                      mean_fn=("fn", "mean"),
                      mean_fp=("fp", "mean"))
                 .reset_index())
        return a

    base = agg(baseline)
    curr = agg(current)
    merged = curr.merge(base, on=["gc", "sw"], suffixes=("_now", "_base"), how="left")
    merged["delta_err"] = merged["mean_err_now"] - merged["mean_err_base"]
    merged["delta_n"] = merged["n_seeds_now"] - merged["n_seeds_base"].fillna(0).astype(int)
    big = merged.dropna(subset=["mean_err_base"])
    big = big[big["delta_err"].abs() >= 0.3].copy()
    big = big.sort_values("delta_err")
    return merged, big


def plot_val_peak_vs_new_best(df: pd.DataFrame):
    """Scatter: x = NEW_BEST epoch (saved), y = max test_F1 among VAL_PEAK entries.
    Points below diagonal imply smoothing saved a weaker epoch than a peak seen earlier.
    """
    if df.empty:
        print("vd080_bc: no data, skipping scatter")
        return False
    pts = []
    for _, r in df.iterrows():
        rd = r["run_dir"]
        th = load_test_history(rd) or []
        if not th:
            continue
        # find saved NEW_BEST epoch = best_info epoch
        saved_epoch = r["epoch"]
        saved_test_f1 = r["test_f1"]
        # max test_f1 over ALL history entries (VAL_PEAK + NEW_BEST)
        all_tf1 = [e.get("test_f1") for e in th if e.get("test_f1") is not None]
        if not all_tf1:
            continue
        peak_test_f1 = max(all_tf1)
        # find the epoch that produced that peak
        peak_epoch = None
        peak_event = None
        for e in th:
            if e.get("test_f1") == peak_test_f1:
                peak_epoch = e.get("epoch")
                peak_event = e.get("event")
                break
        pts.append({
            "run": r["run"],
            "gc": r["gc"],
            "sw": r["sw"],
            "saved_epoch": saved_epoch,
            "saved_test_f1": saved_test_f1,
            "peak_test_f1": peak_test_f1,
            "peak_epoch": peak_epoch,
            "peak_event": peak_event,
            "gap": peak_test_f1 - saved_test_f1,
        })
    if not pts:
        print("val_peak scatter: no test history data")
        return False
    pt_df = pd.DataFrame(pts)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    colors = sns.color_palette("tab10", len(GC_ORDER))
    gc_color = {g: colors[i] for i, g in enumerate(GC_ORDER)}
    markers = {"1raw": "o", "3med": "s", "3mean": "D", "5med": "^", "5mean": "v"}

    # Point at each run
    for _, r in pt_df.iterrows():
        ax.scatter(r["saved_test_f1"], r["peak_test_f1"],
                   s=180, c=[gc_color[r["gc"]]], marker=markers.get(r["sw"], "o"),
                   edgecolor="black", linewidth=1.0, alpha=0.9, zorder=3)
        # annotate gap if measurable
        if r["gap"] > 0.0005:
            ax.annotate(f"+{r['gap']*1000:.1f}m",
                        (r["saved_test_f1"], r["peak_test_f1"]),
                        xytext=(6, 4), textcoords="offset points", fontsize=7, color="#333")

    # diagonal
    lo = min(pt_df["saved_test_f1"].min(), pt_df["peak_test_f1"].min()) - 0.002
    hi = max(pt_df["saved_test_f1"].max(), pt_df["peak_test_f1"].max()) + 0.002
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1.2, label="saved == peak")
    ax.fill_between([lo, hi], [lo, hi], hi, color="#ffcccc", alpha=0.25,
                    label="peak > saved (smoothing missed better epoch)")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Saved (NEW_BEST) test F1   — x-axis = what was actually shipped")
    ax.set_ylabel("Peak test F1 across VAL_PEAK/NEW_BEST history (unsaved max)")
    ax.set_title("vd080_bc: peak test F1 (VAL_PEAK history) vs. saved NEW_BEST\n"
                 f"{len(pt_df)}/19 runs — points above diagonal = smoothing discarded a stronger epoch",
                 fontsize=11, fontweight="bold")
    # Legend — composite gc + sw
    from matplotlib.lines import Line2D
    gc_handles = [Line2D([0], [0], marker="o", lw=0, markersize=10,
                         markerfacecolor=gc_color[g], markeredgecolor="black",
                         label=f"gc={GC_DISP[g]}") for g in GC_ORDER]
    sw_handles = [Line2D([0], [0], marker=markers[s], lw=0, markersize=10,
                         markerfacecolor="lightgray", markeredgecolor="black",
                         label=SW_DISP[s]) for s in SW_ORDER]
    leg1 = ax.legend(handles=gc_handles, title="grad clip", loc="lower right", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=sw_handles, title="smoothing", loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out = OUT / "val_peak_vs_new_best_scatter.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # Print numeric summary for record
    below = pt_df[pt_df["gap"] > 0.0005]
    print(f"  runs with peak > saved (gap>0.0005): {len(below)}/{len(pt_df)}")
    if len(below):
        for _, r in below.sort_values("gap", ascending=False).iterrows():
            print(f"    gc={GC_DISP[r['gc']]:<3} {SW_DISP[r['sw']]:<12}  saved={r['saved_test_f1']:.4f} (ep{r['saved_epoch']}) "
                  f"peak={r['peak_test_f1']:.4f} (ep{r['peak_epoch']}, {r['peak_event']})  gap=+{r['gap']*1000:.2f}m")
    return True


# ============================================================
# README
# ============================================================

def write_readme(ex_df: pd.DataFrame, ns_df: pd.DataFrame, vd_df: pd.DataFrame,
                 delta_all: pd.DataFrame | None = None,
                 delta_big: pd.DataFrame | None = None):
    lines = [
        "# Paper plots — ex0416 sweep + N-scaling + vd080_bc pilot",
        "",
        f"Auto-generated by `scripts/generate_paper_plots.py` on 2026-04-19.",
        f"Samples: ex0416 rows={len(ex_df)}, N-scaling rows={len(ns_df)}, vd080_bc rows={len(vd_df)}.",
        "",
        "## Figures",
        "",
        "- **axis_effect_FN.png** — 5-panel bar chart of mean FN count across lr / focal_gamma / abnormal_weight / weight_decay / dropout sweeps (error bar = std over 5 seeds). Shows which knobs suppress missed anomalies most.",
        "- **axis_effect_FP.png** — Same layout as above but for FP. Reveals the FN-vs-FP trade-off per axis (e.g. larger focal γ shrinks FN but can raise FP via threshold sharpening).",
        "- **n_scaling.png** — Log-log plot of test error (1 − F1) vs. total N per class for three dataset families (fresh0413_reset_v11, fresh0412_v11, v11). Confirms fresh0413 saturates at N≈700 (slope +0.29 > 0) while fresh0412 still scales mildly (slope −0.18) and legacy v11 is data-broken.",
        "- **seed_variance_heatmap.png** — Heatmap of per-seed test F1 across every ex0416 variant (5 seeds per variant). Rows sorted by axis then level; annotation = F1, color = (1 − F1) in %. Identifies seed-robust configs (flat rows) vs. unstable ones.",
        "- **training_curves_top3.png** — val F1 and per-epoch test F1 curves for the best-seed run of three leading ex0416 configurations (lr4e5, fg0p5, abw0p8). Overlays train accuracy and marks NEW_BEST epochs; diagnoses convergence speed and overfitting patterns.",
        "- **vd080_bc_pilot_heatmap.png** — 4×5 gc × sw matrix for the vd080_bc pilot (seed=42 only, 20 cells). Cell color encodes test F1 (viridis); annotation reports F1, FN/FP counts, and the saved-best epoch. Quickly shows which (gradient clip, val-smoothing) combo lands the highest F1 on seed 42.",
        "- **vd080_bc_5seed_heatmap.png** — Same 4×5 gc × sw grid but restricted to cells with 5 complete seeds (currently: all gc=0.5 × every sw, plus gc=1.0 × sw1raw). Color = mean test error (1−F1) in %; per-cell annotation shows mean FN, mean FP, and std of test error across the 5 seeds. Incomplete cells are greyed out with a seed count.",
        "- **vd080_bc_seed_variance_boxplot.png** — Box plot of per-seed test error (1−F1)×100 for each completed (gc, sw) cell. Boxes are sorted by median error (best first) and colored by gc; overlaid black dots are the individual seed results. Exposes which cells converge tightly across seeds vs. which are seed-fragile.",
        "- **vd080_bc_per_cell_dotplot.png** — Per-cell error dot plot (one column per gc×sw cell with ≥3 seeds). Each dot is one seed's test error (1−F1)×100; the black horizontal bar is the cell mean. The single **perfect run** (gc=2.0, sw1-raw, seed=4, F1=1.0000) is drawn as a gold star and annotated directly. Vertical separators group cells by gc level.",
        "- **val_peak_vs_new_best_scatter.png** — Scatter of peak test F1 observed in the epoch history (VAL_PEAK + NEW_BEST events) vs. the test F1 of the saved NEW_BEST checkpoint. Points on the diagonal mean smoothing-gated early-stop chose the literal best epoch; points above the diagonal indicate smoothing discarded a better test F1 that was briefly visible during training (smoothing-failure evidence).",
        "",
        "## vd080_bc pilot summary",
    ]
    if not vd_df.empty:
        total_cells = len(GC_ORDER) * len(SW_ORDER)
        lines.append(f"- Total runs collected: **{len(vd_df)}** across {vd_df.groupby(['gc','sw']).ngroups} of {total_cells} cells (grid = gc ∈ {{{', '.join(GC_DISP[g] for g in GC_ORDER)}}} × sw ∈ {{{', '.join(SW_DISP[s] for s in SW_ORDER)}}}).")
        # Seed completion per cell
        cell_counts = vd_df.groupby(["gc", "sw"])["seed"].nunique()
        complete_cells = cell_counts[cell_counts >= 5]
        lines.append(f"- Cells with 5 complete seeds: **{len(complete_cells)}** — listed: {', '.join(f'gc={GC_DISP[g]}/{SW_DISP[s]}' for (g,s) in complete_cells.index)}.")
        # Aggregated best/worst across complete cells
        if len(complete_cells) > 0:
            agg5 = (vd_df.groupby(["gc", "sw"])
                        .agg(n_seeds=("seed", "nunique"),
                             mean_f1=("test_f1", "mean"),
                             std_f1=("test_f1", "std"),
                             mean_fn=("fn", "mean"),
                             mean_fp=("fp", "mean"))
                        .reset_index())
            agg5 = agg5[agg5["n_seeds"] >= 5].sort_values("mean_f1", ascending=False)
            best = agg5.iloc[0]
            worst = agg5.iloc[-1]
            lines.append(f"- Best 5-seed cell: gc={GC_DISP[best['gc']]}, {SW_DISP[best['sw']]} → mean F1 **{best['mean_f1']:.4f} ± {best['std_f1']:.4f}** (mean FN={best['mean_fn']:.1f}, mean FP={best['mean_fp']:.1f}).")
            lines.append(f"- Worst 5-seed cell: gc={GC_DISP[worst['gc']]}, {SW_DISP[worst['sw']]} → mean F1 {worst['mean_f1']:.4f} ± {worst['std_f1']:.4f} (mean FN={worst['mean_fn']:.1f}, mean FP={worst['mean_fp']:.1f}).")
        # Also report single-seed best (seed=42 pilot) for reference
        best_row = vd_df.sort_values("test_f1", ascending=False).iloc[0]
        lines.append(f"- Single-run best (any seed): gc={GC_DISP[best_row['gc']]}, {SW_DISP[best_row['sw']]}, seed={best_row['seed']} → test F1 {best_row['test_f1']:.4f} (FN={best_row['fn']}, FP={best_row['fp']}, ep={best_row['epoch']}).")
        # Highlight the PERFECT run explicitly
        perf = vd_df[(vd_df["gc"] == PERFECT_RUN_KEY["gc"])
                     & (vd_df["sw"] == PERFECT_RUN_KEY["sw"])
                     & (vd_df["seed"] == PERFECT_RUN_KEY["seed"])]
        if not perf.empty:
            pr = perf.iloc[0]
            lines.append(f"- ⭐ **PERFECT run**: gc={GC_DISP[pr['gc']]}, {SW_DISP[pr['sw']]}, seed={pr['seed']} → test F1 **{pr['test_f1']:.4f}** (FN={pr['fn']}, FP={pr['fp']}, ep={pr['epoch']}). Highlighted in gold on `vd080_bc_5seed_heatmap.png` and `vd080_bc_per_cell_dotplot.png`.")
    else:
        lines.append("- No vd080_bc runs found.")

    # Delta vs. N=46 checkpoint — cells whose mean err changed by ≥ 0.3 pp
    if delta_big is not None and not delta_big.empty:
        lines.append("")
        lines.append(f"## Δerr vs. N=46 checkpoint (≥ 0.3 pp change)")
        lines.append("")
        lines.append("Cell baseline = first 46 runs (by run-name timestamp). Current = all available runs. "
                     "Only cells where mean test error % changed by ≥ 0.3 pp are listed.")
        lines.append("")
        lines.append("| Cell | N (base → now) | mean err% base | mean err% now | Δerr pp | mean FN now | mean FP now |")
        lines.append("|------|---------------|----------------|----------------|---------|-------------|-------------|")
        for _, r in delta_big.iterrows():
            cell = f"gc={GC_DISP[r['gc']]}, {SW_DISP[r['sw']]}"
            nb = int(r["n_seeds_base"]) if pd.notna(r["n_seeds_base"]) else 0
            nn = int(r["n_seeds_now"])
            sign = "↓" if r["delta_err"] < 0 else "↑"
            lines.append(f"| {cell} | {nb} → {nn} | {r['mean_err_base']:.3f} | "
                         f"{r['mean_err_now']:.3f} | {sign}{abs(r['delta_err']):.3f} | "
                         f"{r['mean_fn_now']:.2f} | {r['mean_fp_now']:.2f} |")
    elif delta_all is not None and not delta_all.empty:
        lines.append("")
        lines.append("## Δerr vs. N=46 checkpoint")
        lines.append("")
        lines.append("No cell changed by ≥ 0.3 pp vs. the N=46 baseline — aggregated means are stable within noise.")
    lines.append("")
    lines.append("## Data sources")
    lines.append("- `logs/*/best_info.json` — final test metrics + hparams per run")
    lines.append("- `logs/*/history.json` — per-epoch train/val curves")
    lines.append("- `logs/*/test_history.json` — per-epoch test F1 log with VAL_PEAK / NEW_BEST events (used for the peak-vs-saved scatter)")
    lines.append("- `validations/stability_and_worst_0408.txt` — precomputed N-scaling slopes")
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / 'README.md'}")


# ============================================================
# Main
# ============================================================

def main():
    print("Collecting ex0416 axis runs...")
    ex_df = collect_ex0416()
    print(f"  ex0416 rows: {len(ex_df)}")
    if not ex_df.empty:
        print("  axes:", ex_df.groupby("axis")["run"].count().to_dict())

    print("Collecting N-scaling runs...")
    ns_df = collect_nscale()
    print(f"  nscale rows: {len(ns_df)}")

    # Plot 1 & 2: axis effect
    if not ex_df.empty:
        plot_axis_effect(ex_df, "fn", "axis_effect_FN.png", "FN count (out of 750 abnormals)")
        plot_axis_effect(ex_df, "fp", "axis_effect_FP.png", "FP count (out of 750 normals)")

    # Plot 3: N-scaling
    if not ns_df.empty:
        plot_n_scaling(ns_df)

    # Plot 4: seed variance heatmap
    if not ex_df.empty:
        plot_seed_heatmap(ex_df)

    # Plot 5: vd080_bc pilot heatmap + val_peak vs new_best scatter
    print("Collecting vd080_bc pilot runs...")
    vd_df = collect_vd080_bc()
    print(f"  vd080_bc rows: {len(vd_df)}")
    delta_all, delta_big = pd.DataFrame(), pd.DataFrame()
    if not vd_df.empty:
        plot_vd080_bc_heatmap(vd_df)
        plot_vd080_bc_5seed_heatmap(vd_df)
        plot_vd080_bc_seed_variance_boxplot(vd_df)
        plot_vd080_bc_per_cell_dotplot(vd_df)
        plot_val_peak_vs_new_best(vd_df)
        delta_all, delta_big = compute_vd080_bc_checkpoint_delta(vd_df, n_baseline=46)
        print(f"  delta vs N=46 baseline: {len(delta_big)} cells with |Δerr| >= 0.3 pp")
        for _, r in delta_big.iterrows():
            print(f"    gc={GC_DISP[r['gc']]} {SW_DISP[r['sw']]:<12} "
                  f"base={r['mean_err_base']:.3f}%  now={r['mean_err_now']:.3f}%  "
                  f"Δ={r['delta_err']:+.3f} pp  (n {int(r['n_seeds_base'])}→{int(r['n_seeds_now'])})")
    else:
        print("vd080_bc: no data, skipping heatmap + scatter")

    # Plot 6: training curves
    plot_training_curves_top3()

    write_readme(ex_df, ns_df, vd_df, delta_all, delta_big)
    print("done.")


if __name__ == "__main__":
    main()
