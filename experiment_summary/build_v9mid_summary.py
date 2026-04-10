"""v9mid journey 5-hour iteration 종합 report builder.

출력:
  experiment_summary/SUMMARY.md                   — 새 종합 문서
  experiment_summary/v9mid_journey/data.csv       — run 집계
  experiment_summary/v9mid_journey/plots/*.png    — 8개 plot
  experiment_summary/v9mid_journey/samples/       — 대표 이미지
  experiment_summary/v9mid_journey/before_after/  — renderer fix 비교

재실행:
  PYTHONIOENCODING=utf-8 python experiment_summary/build_v9mid_summary.py
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = ROOT / "experiment_summary"
JRN = OUT / "v9mid_journey"
PLOTS = JRN / "plots"
for d in (JRN, PLOTS, JRN / "samples", JRN / "before_after"):
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. v9mid iteration runs 정보 (5-hour journey)
# -----------------------------------------------------------------------------

# (iter_tag, seed, n, log_dir_name, description)
ITER_RUNS = [
    # v3 — baseline (pre renderer fix)
    ("v3", 42, 700, "260409_190037_v9mid3_win_n700_s42_F0.9940_R0.9940",
     "baseline: mid config, BEFORE renderer fix"),
    ("v3", 1,  700, "260409_223933_v9mid3_win_n700_s1_F0.9947_R0.9947",
     "baseline s=1"),
    ("v3", 2,  700, "260409_225218_v9mid3_win_n700_s2_F0.9960_R0.9960",
     "baseline s=2"),

    # v4 — renderer fix
    ("v4", 42, 700, "260409_231254_v9mid4_win_n700_s42_F0.9987_R0.9987",
     "renderer target preserve fix"),
    ("v4", 1,  700, "260409_232612_v9mid4_win_n700_s1_F0.9980_R0.9980",
     "renderer fix s=1"),
    ("v4", 2,  700, "260409_233852_v9mid4_win_n700_s2_F0.9980_R0.9980",
     "renderer fix s=2"),

    # v5 — std/drift boundary raise
    ("v5", 42, 700, "260409_235841_v9mid5_win_n700_s42_F0.9993_R0.9993",
     "std [2.75,3.75] + drift floor 4.5"),
    ("v5", 1,  700, "260410_001255_v9mid5_win_n700_s1_F0.9987_R0.9987",
     "boundary raise s=1"),
    ("v5", 2,  700, "260410_002716_v9mid5_win_n700_s2_F0.9993_R0.9993",
     "boundary raise s=2"),

    # v6 — + mean_shift raise
    ("v6", 42, 700, "260410_004823_v9mid6_win_n700_s42_F0.9987_R0.9987",
     "+ mean_shift [2.75,4.25]"),
    ("v6", 1,  700, "260410_010216_v9mid6_win_n700_s1_F0.9987_R0.9987",
     "mean_shift raise s=1"),
    ("v6", 2,  700, "260410_011547_v9mid6_win_n700_s2_F0.9993_R0.9993",
     "mean_shift raise s=2"),

    # v6 n=2800
    ("v6+", 42, 2800, "260410_013246_v9mid6_win_n2800_s42_F0.9993_R0.9993",
     "normal count sweep n=2800"),
]

# Gradient clipping sweep (v6 base, 2026-04-10)
GC_RUNS = [
    ("gc1.0", 42, 700, "260409_235841_v9mid5_win_n700_s42_F0.9993_R0.9993",
     "gc=1.0 default (v5 winning)"),
    ("gc1.0", 1,  700, "260410_001255_v9mid5_win_n700_s1_F0.9987_R0.9987",
     "gc=1.0 s=1"),
    ("gc1.0", 2,  700, "260410_002716_v9mid5_win_n700_s2_F0.9993_R0.9993",
     "gc=1.0 s=2"),
    ("gc0.5", 42, 700, "260410_055100_v9mid6_win_gc0p5_n700_s42_F0.9993_R0.9993",
     "gc=0.5 tighter clip"),
    ("gc2.0", 42, 700, "260410_060540_v9mid6_win_gc2p0_n700_s42_F1.0000_R1.0000",
     "gc=2.0 PERFECT run"),
    ("gc2.0", 1,  700, "260410_062236_v9mid6_win_gc2p0_n700_s1_F0.9993_R0.9993",
     "gc=2.0 s=1"),
    ("gc2.0", 2,  700, "260410_063639_v9mid6_win_gc2p0_n700_s2_F0.9980_R0.9980",
     "gc=2.0 s=2"),
]

# Also the very early "bad" run for comparison
PRE_RUN = ("phase1cBmod", 42, 700,
           None,  # killed early, no log folder
           "phase1cBmod bs32 pat10 lr4e-5 — val_loss spike ep5")


def load_run(log_name: str) -> dict | None:
    """best_info.json + history.json 로드."""
    p = LOGS / log_name
    if not (p / "best_info.json").exists():
        return None
    with open(p / "best_info.json", encoding="utf-8") as f:
        bi = json.load(f)
    hist = []
    hp = p / "history.json"
    if hp.exists():
        with open(hp, encoding="utf-8") as f:
            hist = json.load(f)
    return {"bi": bi, "hist": hist, "path": p}


def _runs_to_df(runs) -> pd.DataFrame:
    rows = []
    for tag, seed, n, log_name, desc in runs:
        data = load_run(log_name)
        if data is None:
            continue
        bi = data["bi"]
        th = bi.get("test_history", [])
        last = th[-1] if th else {}
        rows.append(dict(
            iter=tag, seed=seed, n=n,
            best_ep=bi["epoch"],
            test_f1=bi["test_f1"],
            test_recall=bi["test_recall"],
            test_acc=bi["test_acc"],
            abn_R=last.get("test_abn_R", 0),
            nor_R=last.get("test_nor_R", 0),
            fn=last.get("fn", 0),
            fp=last.get("fp", 0),
            total_err=last.get("fn", 0) + last.get("fp", 0),
            log_name=log_name,
            desc=desc,
        ))
    return pd.DataFrame(rows)


def collect_runs() -> pd.DataFrame:
    return _runs_to_df(ITER_RUNS)


def collect_gc_runs() -> pd.DataFrame:
    return _runs_to_df(GC_RUNS)


def load_fn_detail(log_name: str, scenarios: pd.DataFrame) -> list[dict]:
    """해당 run 의 FN 샘플 각각에 대한 클래스/파라미터 반환."""
    p = LOGS / log_name / "predictions" / "fn_abnormal"
    if not p.exists():
        return []
    out = []
    for f in p.iterdir():
        cid = f.name.replace("pred_normal_", "").replace(".png", "")
        row = scenarios[scenarios["chart_id"] == cid]
        if not len(row):
            continue
        r = row.iloc[0]
        try:
            dp = json.loads(r["defect_params"])
        except Exception:
            dp = {}
        out.append(dict(chart_id=cid, cls=r["class"], params=dp))
    return out


# -----------------------------------------------------------------------------
# 2. Plot 함수들
# -----------------------------------------------------------------------------

COLORS = {
    "v3":  "#C62828",  # red (bad baseline)
    "v4":  "#EF6C00",  # orange (first fix)
    "v5":  "#2E7D32",  # green (winner)
    "v6":  "#1565C0",  # blue
    "v6+": "#6A1B9A",  # purple
}


def plot_performance_timeline(df: pd.DataFrame, out: Path):
    """Iteration 별 test_f1 (seed 별 점 + mean line)."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    iter_order = ["v3", "v4", "v5", "v6", "v6+"]
    x_pos = {t: i for i, t in enumerate(iter_order)}

    for tag in iter_order:
        sub = df[df["iter"] == tag]
        if not len(sub):
            continue
        x = [x_pos[tag]] * len(sub)
        c = COLORS[tag]
        ax.scatter(x, sub["test_f1"] * 100, s=120, color=c, alpha=0.75,
                   edgecolor="black", linewidth=0.8, zorder=3,
                   label=f"{tag} (n={len(sub)})")
        ax.hlines(sub["test_f1"].mean() * 100, x_pos[tag] - 0.25,
                  x_pos[tag] + 0.25, color=c, linewidth=3, zorder=2)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([
        "v3\n(baseline)",
        "v4\n(renderer fix)",
        "v5\n(std/drift raise)",
        "v6\n(+ms raise)",
        "v6+\n(n=2800)",
    ], fontsize=10)
    ax.set_ylabel("test F1 (%)", fontsize=11)
    ax.set_title("v9mid Iteration — 3-seed Test F1 Progression",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(99.2, 100.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(99.91, color="green", linestyle="--", alpha=0.5,
               label="v5 mean 99.91 (winner)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # Annotate improvement
    ax.annotate("", xy=(2, 99.91), xytext=(0, 99.40),
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2))
    ax.text(1, 99.62, "+0.51 pts\n85% err ↓", color="darkgreen",
            fontsize=10, ha="center", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_error_breakdown(df: pd.DataFrame, out: Path):
    """Iter 별 FN/FP stacked bar (seed mean)."""
    iter_order = ["v3", "v4", "v5", "v6", "v6+"]
    fn_means = []
    fp_means = []
    fn_stds = []
    fp_stds = []
    labels = []
    for t in iter_order:
        sub = df[df["iter"] == t]
        if not len(sub):
            continue
        fn_means.append(sub["fn"].mean())
        fp_means.append(sub["fp"].mean())
        fn_stds.append(sub["fn"].std() if len(sub) > 1 else 0)
        fp_stds.append(sub["fp"].std() if len(sub) > 1 else 0)
        labels.append(t)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    x = np.arange(len(labels))
    w = 0.35

    bars1 = ax.bar(x - w / 2, fn_means, w, yerr=fn_stds,
                   label="FN (missed anomaly)", color="#C62828", alpha=0.85,
                   error_kw=dict(lw=1.5, capsize=4))
    bars2 = ax.bar(x + w / 2, fp_means, w, yerr=fp_stds,
                   label="FP (false alarm)", color="#FB8C00", alpha=0.85,
                   error_kw=dict(lw=1.5, capsize=4))

    # Value labels
    for bars, vals in [(bars1, fn_means), (bars2, fp_means)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                    f"{v:.1f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Errors (out of 750 per category)", fontsize=11)
    ax.set_title("FN / FP Per Iteration (3-seed mean ± std)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(max(fn_means), max(fp_means)) * 1.3 + 1)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_class_fn_pattern(df: pd.DataFrame, scenarios: pd.DataFrame, out: Path):
    """각 iter 에서 어떤 클래스가 FN 됐는지 stacked bar."""
    iter_order = ["v3", "v4", "v5", "v6"]
    classes = ["mean_shift", "standard_deviation", "spike", "drift", "context"]
    class_colors = {
        "mean_shift": "#E53935",
        "standard_deviation": "#FB8C00",
        "spike": "#FDD835",
        "drift": "#43A047",
        "context": "#1E88E5",
    }

    iter_class_counts = {}
    for t in iter_order:
        counts = {c: 0 for c in classes}
        for _, row in df[df["iter"] == t].iterrows():
            for fn in load_fn_detail(row["log_name"], scenarios):
                if fn["cls"] in counts:
                    counts[fn["cls"]] += 1
        iter_class_counts[t] = counts

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    x = np.arange(len(iter_order))
    w = 0.6
    bottom = np.zeros(len(iter_order))
    for cls in classes:
        vals = [iter_class_counts[t][cls] for t in iter_order]
        ax.bar(x, vals, w, bottom=bottom, label=cls,
               color=class_colors[cls], edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(x[i], bottom[i] + v / 2, str(v),
                        ha="center", va="center", fontsize=9, color="white",
                        fontweight="bold")
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(iter_order, fontsize=11)
    ax.set_ylabel("Cumulative FN count (3 seeds)", fontsize=11)
    ax.set_title("FN Class Distribution by Iteration\n"
                 "(where did the misses go?)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="defect class", fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate
    ax.text(0.2, sum(iter_class_counts["v3"].values()) + 0.8,
            "spike heavy\n(renderer bug)",
            fontsize=9, style="italic", color="#C62828")
    ax.text(1.2, sum(iter_class_counts["v4"].values()) + 0.8,
            "spike all gone!",
            fontsize=9, style="italic", color="darkgreen", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_val_loss_comparison(out: Path):
    """lr 4e-5 (spike) vs lr 2e-5 (stable) val loss curves."""
    # lr 4e-5 run: v9mid_phase1cBmod (bs32 pat10 lr4e-5) — val spike ep5
    candidates_spike = list(LOGS.glob("260409_18*_v9mid_phase1cBmod*"))
    candidates_stable = [LOGS / r[3] for r in ITER_RUNS if r[3] and r[3].startswith("260409_235841")]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
    max_ep = 0

    # Spike case (lr 4e-5)
    if candidates_spike:
        with open(candidates_spike[0] / "history.json", encoding="utf-8") as f:
            h = json.load(f)
        eps = [e["epoch"] for e in h]
        vl = [e["val_loss"] for e in h]
        max_ep = max(max_ep, max(eps))
        ax.plot(eps, vl, "o-", color="#C62828", linewidth=2, markersize=7,
                label="lr_backbone 4e-5 (phase1cBmod) — SPIKE ep5",
                alpha=0.85)
        # mark the spike
        ax.annotate(f"spike\n{max(vl):.3f}",
                    xy=(eps[vl.index(max(vl))], max(vl)),
                    xytext=(eps[vl.index(max(vl))] + 1.5, max(vl) * 0.9),
                    fontsize=10, color="#C62828", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#C62828"))

    # Stable case (lr 2e-5)
    for path in candidates_stable:
        with open(path / "history.json", encoding="utf-8") as f:
            h = json.load(f)
        eps = [e["epoch"] for e in h]
        vl = [e["val_loss"] for e in h]
        max_ep = max(max_ep, max(eps))
        ax.plot(eps, vl, "o-", color="#2E7D32", linewidth=2, markersize=7,
                label="lr_backbone 2e-5 (winning) — stable",
                alpha=0.85)

    ax.set_xlabel("epoch", fontsize=11)
    ax.set_ylabel("val_loss", fontsize=11)
    ax.set_title("Val Loss Stability — LR 2e-5 vs 4e-5",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(list(range(1, max_ep + 1)))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 0.85)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_renderer_fix_impact(out: Path):
    """Before/after renderer fix: v3 vs v4 error counts."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)

    categories = ["test_f1 (%)", "FN (spike)", "FN (others)", "FP"]
    v3 = [99.49, 2.0, 1.0, 3.0]  # baseline means
    v4 = [99.82, 0.0, 2.3, 0.3]

    x = np.arange(len(categories))
    w = 0.38
    b1 = ax.bar(x - w / 2, v3, w, label="v3 baseline (outlier filter bug)",
                color="#C62828", alpha=0.85, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w / 2, v4, w, label="v4 (target preserve fix)",
                color="#2E7D32", alpha=0.85, edgecolor="black", linewidth=0.5)

    for bars, vals in [(b1, v3), (b2, v4)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                    f"{v:.2f}" if v < 5 else f"{v:.2f}",
                    ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title("Renderer Fix Impact — _filter_outliers Target Preserve",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Secondary y for f1
    ax.annotate("spike\nFN → 0", xy=(1.2, 0), xytext=(1.2, 3),
                fontsize=10, color="darkgreen", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="darkgreen"))

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_seed_robustness(df: pd.DataFrame, out: Path):
    """각 iter 3-seed std 표시."""
    iter_order = ["v3", "v4", "v5", "v6"]
    means_f1 = []
    stds_f1 = []
    means_abn = []
    stds_abn = []
    means_nor = []
    stds_nor = []
    for t in iter_order:
        sub = df[(df["iter"] == t) & (df["n"] == 700)]
        if len(sub) < 1:
            means_f1.append(np.nan); stds_f1.append(0)
            means_abn.append(np.nan); stds_abn.append(0)
            means_nor.append(np.nan); stds_nor.append(0)
            continue
        means_f1.append(sub["test_f1"].mean() * 100)
        stds_f1.append(sub["test_f1"].std() * 100 if len(sub) > 1 else 0)
        means_abn.append(sub["abn_R"].mean() * 100)
        stds_abn.append(sub["abn_R"].std() * 100 if len(sub) > 1 else 0)
        means_nor.append(sub["nor_R"].mean() * 100)
        stds_nor.append(sub["nor_R"].std() * 100 if len(sub) > 1 else 0)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    x = np.arange(len(iter_order))
    w = 0.25
    ax.bar(x - w, means_f1, w, yerr=stds_f1, label="test_f1",
           color="#1565C0", alpha=0.85, error_kw=dict(lw=1.5, capsize=4))
    ax.bar(x, means_abn, w, yerr=stds_abn, label="abn_R",
           color="#EF6C00", alpha=0.85, error_kw=dict(lw=1.5, capsize=4))
    ax.bar(x + w, means_nor, w, yerr=stds_nor, label="nor_R",
           color="#2E7D32", alpha=0.85, error_kw=dict(lw=1.5, capsize=4))

    for i, t in enumerate(iter_order):
        ax.text(i - w, means_f1[i] + 0.03, f"{means_f1[i]:.2f}",
                ha="center", fontsize=8)
        ax.text(i, means_abn[i] + 0.03, f"{means_abn[i]:.2f}",
                ha="center", fontsize=8)
        ax.text(i + w, means_nor[i] + 0.03, f"{means_nor[i]:.2f}",
                ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(iter_order, fontsize=11)
    ax.set_ylabel("% (mean ± std of 3 seeds)", fontsize=11)
    ax.set_title("3-Seed Robustness — F1 / abn_R / nor_R",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(99.0, 100.15)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_impact(out: Path):
    """Boundary raise 전후 — mean_shift/std/drift FN 변화."""
    iters = ["v4", "v5", "v6"]
    classes = ["mean_shift", "standard_deviation", "spike", "drift"]
    data = {
        "v4": {"mean_shift": 0, "standard_deviation": 4, "spike": 0, "drift": 3},
        "v5": {"mean_shift": 3, "standard_deviation": 1, "spike": 0, "drift": 0},
        "v6": {"mean_shift": 1, "standard_deviation": 2, "spike": 0, "drift": 1},
    }

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    x = np.arange(len(iters))
    w = 0.2
    for i, cls in enumerate(classes):
        offset = (i - 1.5) * w
        vals = [data[t][cls] for t in iters]
        ax.bar(x + offset, vals, w, label=cls,
               alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([
        "v4\n(mid boundaries)",
        "v5\n(std+drift raised)",
        "v6\n(+ms raised)",
    ], fontsize=10)
    ax.set_ylabel("FN count (3 seeds total)", fontsize=11)
    ax.set_title("Boundary Raise Impact — Per-Class FN",
                 fontsize=13, fontweight="bold")
    ax.legend(title="defect class", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotations
    ax.annotate("drift FN: 3→0", xy=(1.3, 0), xytext=(1.8, 2),
                fontsize=10, color="darkgreen", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkgreen"))
    ax.annotate("std FN: 4→1", xy=(1.05, 1), xytext=(0.3, 3.2),
                fontsize=10, color="darkgreen", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkgreen"))

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_hard_cases_heatmap(df: pd.DataFrame, scenarios: pd.DataFrame,
                             out: Path):
    """가장 어려운 chart_id 들을 iteration 별 failure 패턴으로 heatmap."""
    # iter 별 FN chart_id 수집
    chart_iters = {}
    for _, row in df.iterrows():
        tag = row["iter"]
        for fn in load_fn_detail(row["log_name"], scenarios):
            cid = fn["chart_id"]
            chart_iters.setdefault(cid, {}).setdefault(tag, 0)
            chart_iters[cid][tag] += 1

    # 2 seeds 이상에서 실패한 chart 만
    hard = {cid: d for cid, d in chart_iters.items()
            if sum(d.values()) >= 2}
    if not hard:
        return

    iter_order = ["v3", "v4", "v5", "v6"]
    chart_ids = sorted(hard.keys())
    matrix = np.zeros((len(chart_ids), len(iter_order)))
    for i, cid in enumerate(chart_ids):
        for j, t in enumerate(iter_order):
            matrix[i, j] = hard[cid].get(t, 0)

    # chart class lookup
    chart_class = {}
    for cid in chart_ids:
        row = scenarios[scenarios["chart_id"] == cid]
        if len(row):
            chart_class[cid] = row.iloc[0]["class"]
        else:
            chart_class[cid] = "?"

    fig, ax = plt.subplots(figsize=(8, max(4, len(chart_ids) * 0.45)), dpi=120)
    im = ax.imshow(matrix, cmap="Reds", aspect="auto",
                   vmin=0, vmax=3)
    ax.set_xticks(np.arange(len(iter_order)))
    ax.set_xticklabels(iter_order, fontsize=11)
    ax.set_yticks(np.arange(len(chart_ids)))
    ax.set_yticklabels([f"{cid} ({chart_class[cid][:8]})" for cid in chart_ids],
                       fontsize=9)
    ax.set_title("Persistent Hard Cases — FN count across seeds × iter",
                 fontsize=12, fontweight="bold")

    # Cell annotations
    for i in range(len(chart_ids)):
        for j in range(len(iter_order)):
            v = int(matrix[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v >= 2 else "black",
                        fontsize=10, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("failed seeds", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_gc_sweep(gc_df: pd.DataFrame, out: Path):
    """Gradient clipping 1.0 vs 0.5 vs 2.0 비교 — f1, FN, FP, seed variance."""
    import numpy as np
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=120)

    gc_order = ["gc1.0", "gc0.5", "gc2.0"]
    gc_labels = ["gc=1.0\n(default)", "gc=0.5\n(tighter)", "gc=2.0\n(looser)"]
    gc_colors = {"gc1.0": "#2E7D32", "gc0.5": "#EF6C00", "gc2.0": "#6A1B9A"}

    # Panel 1: test_f1 with seed points + mean line
    x_pos = {t: i for i, t in enumerate(gc_order)}
    for tag in gc_order:
        sub = gc_df[gc_df["iter"] == tag]
        if not len(sub):
            continue
        x = [x_pos[tag]] * len(sub)
        c = gc_colors[tag]
        ax1.scatter(x, sub["test_f1"] * 100, s=150, color=c, alpha=0.8,
                    edgecolor="black", linewidth=0.8, zorder=3,
                    label=f"{tag} (n={len(sub)})")
        m = sub["test_f1"].mean() * 100
        ax1.hlines(m, x_pos[tag] - 0.25, x_pos[tag] + 0.25,
                   color=c, linewidth=3, zorder=2)
        # Annotate std
        if len(sub) > 1:
            s = sub["test_f1"].std() * 100
            ax1.text(x_pos[tag], m + 0.04, f"σ={s:.2f}",
                     ha="center", fontsize=8, style="italic", color=c)

    # Mark the perfect run
    perfect = gc_df[(gc_df["iter"] == "gc2.0") & (gc_df["test_f1"] >= 0.9999)]
    if len(perfect):
        ax1.annotate("PERFECT\nF=1.0000",
                     xy=(x_pos["gc2.0"], 100.0),
                     xytext=(x_pos["gc2.0"] + 0.3, 99.97),
                     fontsize=10, color="darkred", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="darkred"))

    ax1.set_xticks(list(x_pos.values()))
    ax1.set_xticklabels(gc_labels, fontsize=10)
    ax1.set_ylabel("test F1 (%)", fontsize=11)
    ax1.set_title("Test F1 per Gradient Clip Setting", fontsize=12, fontweight="bold")
    ax1.set_ylim(99.75, 100.05)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(fontsize=9, loc="lower right")

    # Panel 2: FN vs FP trade-off stacked
    fn_means = []
    fp_means = []
    labels = []
    for t in gc_order:
        sub = gc_df[gc_df["iter"] == t]
        if len(sub):
            fn_means.append(sub["fn"].mean())
            fp_means.append(sub["fp"].mean())
            labels.append(t)

    x = np.arange(len(labels))
    w = 0.35
    ax2.bar(x - w / 2, fn_means, w, label="FN (missed anomaly)",
            color="#C62828", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.bar(x + w / 2, fp_means, w, label="FP (false alarm)",
            color="#FB8C00", alpha=0.85, edgecolor="black", linewidth=0.5)

    for i, (fn, fp) in enumerate(zip(fn_means, fp_means)):
        ax2.text(i - w / 2, fn + 0.05, f"{fn:.1f}", ha="center", fontsize=10)
        ax2.text(i + w / 2, fp + 0.05, f"{fp:.1f}", ha="center", fontsize=10)

    ax2.set_xticks(x)
    ax2.set_xticklabels(gc_labels, fontsize=10)
    ax2.set_ylabel("mean errors (out of 750)", fontsize=11)
    ax2.set_title("FN/FP Trade-off per Gradient Clip", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(max(fn_means), max(fp_means)) * 1.5 + 0.5)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_progression(df: pd.DataFrame, out: Path):
    """Best_ep 분포 — 언제 best 잡히는가."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    for tag in ["v3", "v4", "v5", "v6"]:
        sub = df[df["iter"] == tag]
        if not len(sub):
            continue
        ax.scatter([tag] * len(sub), sub["best_ep"],
                   s=150, color=COLORS[tag], alpha=0.75,
                   edgecolor="black", linewidth=0.8,
                   label=f"{tag} mean ep {sub['best_ep'].mean():.1f}")
    ax.set_ylabel("best epoch", fontsize=11)
    ax.set_title("Best Epoch Distribution per Iteration",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 3. 실행
# -----------------------------------------------------------------------------

def main():
    import pandas as pd
    print("=" * 60)
    print("v9mid journey summary builder")
    print("=" * 60)

    sc_path = ROOT / "data" / "scenarios.csv"
    if not sc_path.exists():
        print(f"WARN: {sc_path} 없음 — FN detail plot 제한")
        scenarios = pd.DataFrame(columns=["chart_id", "class", "defect_params"])
    else:
        scenarios = pd.read_csv(sc_path)

    df = collect_runs()
    print(f"\nCollected {len(df)} runs")
    print(df[["iter", "seed", "n", "best_ep", "test_f1", "fn", "fp"]].to_string())

    df.to_csv(JRN / "data.csv", index=False)
    print(f"\nSaved: {JRN / 'data.csv'}")

    # Plots
    print("\nPlotting...")
    plot_performance_timeline(df, PLOTS / "01_performance_timeline.png")
    print("  01_performance_timeline.png")
    plot_error_breakdown(df, PLOTS / "02_error_breakdown.png")
    print("  02_error_breakdown.png")
    plot_class_fn_pattern(df, scenarios, PLOTS / "03_class_fn_pattern.png")
    print("  03_class_fn_pattern.png")
    plot_val_loss_comparison(PLOTS / "04_val_loss_comparison.png")
    print("  04_val_loss_comparison.png")
    plot_renderer_fix_impact(PLOTS / "05_renderer_fix_impact.png")
    print("  05_renderer_fix_impact.png")
    plot_seed_robustness(df, PLOTS / "06_seed_robustness.png")
    print("  06_seed_robustness.png")
    plot_boundary_impact(PLOTS / "07_boundary_impact.png")
    print("  07_boundary_impact.png")
    plot_hard_cases_heatmap(df, scenarios, PLOTS / "08_hard_cases_heatmap.png")
    print("  08_hard_cases_heatmap.png")
    plot_epoch_progression(df, PLOTS / "09_epoch_progression.png")
    print("  09_epoch_progression.png")

    gc_df = collect_gc_runs()
    gc_df.to_csv(JRN / "gc_data.csv", index=False)
    print(f"\nGC runs collected: {len(gc_df)}")
    plot_gc_sweep(gc_df, PLOTS / "10_grad_clip_sweep.png")
    print("  10_grad_clip_sweep.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
