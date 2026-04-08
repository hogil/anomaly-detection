"""
실험 결과 요약 생성기.

logs/ 폴더의 v-series best_info.json 을 스캔해서 카테고리별로
- data.csv (전체 raw)
- cat1~cat4 PNG plots
- summary 표를 stdout 으로 출력 (SUMMARY.md 에 paste)

Categories
----------
1. dataset_scale       : v8_init / v8_init_n2~n5 — 데이터 크기 sweep (same hparams)
2. normal_count_sweep  : v8seed_nXXXX_sX (25) + v9_lr3tie_nXXXX_s1 (4) — multi-seed
3. noise_impact        : v8seed_n2800 vs v9_tiefix_n2800 / v9_ep10_n2800 (same hparams, 다른 데이터)
4. lr_tuning           : v9_ep10 (lr 2e-5) vs v9_tiefix (lr 5e-5) vs v9_lr3tie (lr 3e-5)

Usage:
    python experiment_summary/build_summary.py
"""
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = ROOT / "experiment_summary"
OUT.mkdir(exist_ok=True)


def load_runs():
    """logs/ 의 모든 best_info.json 을 dict 로 로드."""
    runs = []
    for d in sorted(LOGS.iterdir()):
        if not d.is_dir():
            continue
        info = d / "best_info.json"
        if not info.exists():
            continue
        try:
            with open(info, encoding="utf-8") as f:
                bi = json.load(f)
            hp = bi.get("hparams", {})
            tm = bi.get("test_metrics", {})
            runs.append({
                "name": d.name,
                "epoch": bi.get("epoch"),
                "val_f1": bi.get("val_f1"),
                "test_f1": bi.get("test_f1"),
                "test_acc": bi.get("test_acc"),
                "abn_R": tm.get("abnormal", {}).get("recall"),
                "nor_R": tm.get("normal", {}).get("recall"),
                "lr_bb": hp.get("lr_backbone"),
                "lr_head": hp.get("lr_head"),
                "epochs": hp.get("epochs"),
                "focal_gamma": hp.get("focal_gamma"),
                "abn_weight": hp.get("abnormal_weight"),
                "min_ep": hp.get("min_epochs"),
                "smooth": hp.get("smooth_window"),
                "mode": hp.get("mode"),
            })
        except Exception:
            pass
    return runs


def parse_n_seed(name):
    """'v8seed_n1400_s42' → (1400, 42); 매칭 안 되면 (None, None)."""
    m = re.search(r"_n(\d+)_s(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def fmt_pct(x):
    return f"{x*100:.2f}" if x is not None else "—"


def stats(values):
    """mean, std, min, max."""
    arr = np.array([v for v in values if v is not None])
    if len(arr) == 0:
        return None
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()), "n": int(len(arr))}


# =============================================================================
# Categorize
# =============================================================================
runs = load_runs()
v_runs = [r for r in runs if r["name"].startswith(("v8", "v9"))]
print(f"전체 v-series runs: {len(v_runs)}")

# --- Cat 1: dataset scale (v8_init series, single seed) ---
cat1_names = ["v8_init", "v8_init_n2", "v8_init_n3", "v8_init_n4", "v8_init_n5"]
cat1 = [r for r in v_runs if r["name"] in cat1_names]
# 메모리 노트에 의하면 n2=1400, n3=2100, n4=2800, n5=3500, base=700
n_map = {"v8_init": 700, "v8_init_n2": 1400, "v8_init_n3": 2100,
         "v8_init_n4": 2800, "v8_init_n5": 3500}
for r in cat1:
    r["normal_count"] = n_map.get(r["name"])

# --- Cat 2: normal-count sweep (multi-seed) ---
def in_seed_group(name, prefix):
    return name.startswith(prefix + "_n") and "_s" in name

cat2_v8 = [r for r in v_runs if in_seed_group(r["name"], "v8seed")]
cat2_v9_tie = [r for r in v_runs if in_seed_group(r["name"], "v9_lr3tie")]
for r in cat2_v8 + cat2_v9_tie:
    n, s = parse_n_seed(r["name"])
    r["normal_count"] = n
    r["seed"] = s

# --- Cat 3: noise impact (same n=2800, multi-seed, v8 vs v9) ---
cat3_v8 = [r for r in cat2_v8 if r["normal_count"] == 2800]
cat3_v9_tiefix = [r for r in v_runs if r["name"].startswith("v9_tiefix_n2800_s")]
cat3_v9_ep10 = [r for r in v_runs if r["name"].startswith("v9_ep10_n2800_s")]
cat3_v9_lr3tie_2800 = [r for r in cat2_v9_tie if r["normal_count"] == 2800]

# --- Cat 4: LR tuning (n2800 multi-seed groups) ---
# v8seed (lr 5e-5, no fix) vs v9_tiefix (lr 5e-5, +tie) vs v9_ep10 (lr 2e-5) vs v9_lr3tie (lr 3e-5)
cat4_groups = {
    "v8 baseline\n(lr 5e-5)":        cat3_v8,
    "v9 tie-fix\n(lr 5e-5)":         cat3_v9_tiefix,
    "v9 ep10\n(lr 2e-5)":            cat3_v9_ep10,
    "v9 lr3tie\n(lr 3e-5)":          cat3_v9_lr3tie_2800,
}

# =============================================================================
# Print tables
# =============================================================================
def print_table(title, header, rows):
    print(f"\n## {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---:"] * len(header)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


print("\n" + "=" * 70)
print(" CATEGORY 1: 데이터셋 크기 sweep (v8_init series, single seed)")
print("=" * 70)
cat1_sorted = sorted(cat1, key=lambda r: r["normal_count"] or 0)
rows = []
for r in cat1_sorted:
    rows.append([
        f"{r['normal_count']}",
        r["name"],
        fmt_pct(r["test_f1"]),
        fmt_pct(r["abn_R"]),
        fmt_pct(r["nor_R"]),
        f"{r['epoch']}",
    ])
print_table("Cat 1 — dataset scale", ["normal", "run", "F1 %", "abn_R %", "nor_R %", "best_ep"], rows)


print("\n" + "=" * 70)
print(" CATEGORY 2: Normal count sweep (multi-seed, v8 dataset)")
print("=" * 70)
# v8seed: 5 seeds × 5 counts
counts = [700, 1400, 2100, 2800, 3500]
v8_table = []
for n in counts:
    grp = [r for r in cat2_v8 if r["normal_count"] == n]
    s_f1 = stats([r["test_f1"] for r in grp])
    s_abn = stats([r["abn_R"] for r in grp])
    s_nor = stats([r["nor_R"] for r in grp])
    if s_f1:
        v8_table.append([
            f"{n}",
            f"{s_f1['n']}",
            f"{s_f1['mean']*100:.2f} ± {s_f1['std']*100:.2f}",
            f"{s_abn['mean']*100:.2f} ± {s_abn['std']*100:.2f}",
            f"{s_nor['mean']*100:.2f} ± {s_nor['std']*100:.2f}",
        ])
print_table("Cat 2 — v8 dataset, multi-seed",
            ["normal", "n_seeds", "F1 mean±std", "abn_R mean±std", "nor_R mean±std"],
            v8_table)

# v9_lr3tie: 1 seed × 4 counts
print("\n## Cat 2 — v9 dataset (winning lr3tie config, single seed s1)\n")
print("| normal | run | F1 % | abn_R % | nor_R % |")
print("|---:|---|---:|---:|---:|")
for n in [700, 1400, 2100, 2800]:
    for r in cat2_v9_tie:
        if r["normal_count"] == n:
            print(f"| {n} | {r['name']} | {fmt_pct(r['test_f1'])} | "
                  f"{fmt_pct(r['abn_R'])} | {fmt_pct(r['nor_R'])} |")


print("\n" + "=" * 70)
print(" CATEGORY 3: Noise impact (v8 vs v9, n=2800, multi-seed)")
print("=" * 70)

def group_stats(label, grp):
    s_f1 = stats([r["test_f1"] for r in grp])
    s_abn = stats([r["abn_R"] for r in grp])
    s_nor = stats([r["nor_R"] for r in grp])
    if not s_f1:
        return None
    return [
        label,
        f"{s_f1['n']}",
        f"{s_f1['mean']*100:.2f} ± {s_f1['std']*100:.2f}",
        f"{s_abn['mean']*100:.2f} ± {s_abn['std']*100:.2f}",
        f"{s_nor['mean']*100:.2f} ± {s_nor['std']*100:.2f}",
    ]

cat3_rows = []
for label, grp in [
    ("v8 (baseline noise)",       cat3_v8),
    ("v9 tie-fix (+25% noise)",   cat3_v9_tiefix),
    ("v9 ep10 (+25% noise)",      cat3_v9_ep10),
]:
    row = group_stats(label, grp)
    if row:
        cat3_rows.append(row)
print_table("Cat 3 — noise impact", ["dataset", "n_seeds", "F1 %", "abn_R %", "nor_R %"], cat3_rows)


print("\n" + "=" * 70)
print(" CATEGORY 4: LR tuning (n=2800)")
print("=" * 70)
cat4_rows = []
for label, grp in cat4_groups.items():
    row = group_stats(label.replace("\n", " "), grp)
    if row:
        cat4_rows.append(row)
print_table("Cat 4 — LR tuning", ["config", "n_seeds", "F1 %", "abn_R %", "nor_R %"], cat4_rows)


# =============================================================================
# CSV dump
# =============================================================================
import csv
csv_path = OUT / "data.csv"
fields = ["category", "name", "normal_count", "seed", "lr_bb", "lr_head", "epochs",
          "best_epoch", "test_f1", "abn_R", "nor_R", "val_f1"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    def dump(cat, rs):
        for r in rs:
            w.writerow({
                "category": cat,
                "name": r["name"],
                "normal_count": r.get("normal_count", ""),
                "seed": r.get("seed", ""),
                "lr_bb": r["lr_bb"],
                "lr_head": r["lr_head"],
                "epochs": r["epochs"],
                "best_epoch": r["epoch"],
                "test_f1": r["test_f1"],
                "abn_R": r["abn_R"],
                "nor_R": r["nor_R"],
                "val_f1": r["val_f1"],
            })
    dump("1_dataset_scale", cat1_sorted)
    dump("2_normal_count_v8", cat2_v8)
    dump("2_normal_count_v9_lr3tie", cat2_v9_tie)
    dump("3_noise_v9_tiefix", cat3_v9_tiefix)
    dump("3_noise_v9_ep10", cat3_v9_ep10)
print(f"\n[saved] {csv_path}")


# =============================================================================
# Plots
# =============================================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 130,
})


# --- Plot Cat 1: dataset scale (single seed) ---
fig, ax = plt.subplots(figsize=(8, 5))
xs = [r["normal_count"] for r in cat1_sorted]
f1s = [r["test_f1"] * 100 for r in cat1_sorted]
abns = [r["abn_R"] * 100 for r in cat1_sorted]
nors = [r["nor_R"] * 100 for r in cat1_sorted]
ax.plot(xs, f1s, "o-", lw=2, ms=9, label="F1", color="#1f77b4")
ax.plot(xs, abns, "s--", lw=1.5, ms=7, label="abn_R", color="#d62728")
ax.plot(xs, nors, "^--", lw=1.5, ms=7, label="nor_R", color="#2ca02c")
ax.set_xlabel("Normal samples (training)")
ax.set_ylabel("Score (%)")
ax.set_title("Cat 1 — Dataset scale (v8_init, single seed)")
ax.set_xticks(xs)
ax.legend()
ax.set_ylim(98.0, 100.05)
fig.tight_layout()
fig.savefig(OUT / "cat1_dataset_scale.png")
plt.close(fig)
print(f"[saved] cat1_dataset_scale.png")


# --- Plot Cat 2: normal_count multi-seed (v8 dataset, mean±std) ---
fig, ax = plt.subplots(figsize=(9, 5.5))
xs = counts
f1_mean, f1_std = [], []
abn_mean, abn_std = [], []
nor_mean, nor_std = [], []
for n in counts:
    grp = [r for r in cat2_v8 if r["normal_count"] == n]
    f1_mean.append(np.mean([r["test_f1"] for r in grp]) * 100)
    f1_std.append(np.std([r["test_f1"] for r in grp]) * 100)
    abn_mean.append(np.mean([r["abn_R"] for r in grp]) * 100)
    abn_std.append(np.std([r["abn_R"] for r in grp]) * 100)
    nor_mean.append(np.mean([r["nor_R"] for r in grp]) * 100)
    nor_std.append(np.std([r["nor_R"] for r in grp]) * 100)

ax.errorbar(xs, f1_mean, yerr=f1_std, fmt="o-", lw=2.5, ms=10, capsize=5,
            label="F1", color="#1f77b4")
ax.errorbar(xs, abn_mean, yerr=abn_std, fmt="s--", lw=1.5, ms=7, capsize=4,
            label="abn_R", color="#d62728", alpha=0.85)
ax.errorbar(xs, nor_mean, yerr=nor_std, fmt="^--", lw=1.5, ms=7, capsize=4,
            label="nor_R", color="#2ca02c", alpha=0.85)
# Annotate the sweet spot
peak_idx = int(np.argmax(f1_mean))
ax.annotate(f"sweet spot\nF1 {f1_mean[peak_idx]:.2f}%",
            xy=(xs[peak_idx], f1_mean[peak_idx]),
            xytext=(xs[peak_idx], f1_mean[peak_idx] - 0.15),
            ha="center", fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1f77b4"))
ax.set_xlabel("Normal samples (training)")
ax.set_ylabel("Score (%)")
ax.set_title("Cat 2 — Normal count sweep on v8 dataset (5 seeds, mean ± std)")
ax.set_xticks(xs)
ax.legend(loc="lower left")
ax.set_ylim(99.4, 100.05)
fig.tight_layout()
fig.savefig(OUT / "cat2_normal_count.png")
plt.close(fig)
print(f"[saved] cat2_normal_count.png")


# --- Plot Cat 3: noise impact ---
fig, ax = plt.subplots(figsize=(8, 5.5))
groups = [
    ("v8\n(baseline noise)",       cat3_v8,        "#2ca02c"),
    ("v9 tie-fix\n(+25% noise)",   cat3_v9_tiefix, "#ff7f0e"),
    ("v9 ep10\n(+25% noise)",      cat3_v9_ep10,   "#d62728"),
]
labels, f1m, f1s, abnm, abns_s, norm_, nors_ = [], [], [], [], [], [], []
for label, grp, _ in groups:
    if not grp:
        continue
    labels.append(label)
    f1m.append(np.mean([r["test_f1"] for r in grp]) * 100)
    f1s.append(np.std([r["test_f1"] for r in grp]) * 100)
    abnm.append(np.mean([r["abn_R"] for r in grp]) * 100)
    abns_s.append(np.std([r["abn_R"] for r in grp]) * 100)
    norm_.append(np.mean([r["nor_R"] for r in grp]) * 100)
    nors_.append(np.std([r["nor_R"] for r in grp]) * 100)

x = np.arange(len(labels))
w = 0.27
ax.bar(x - w, f1m, w, yerr=f1s, label="F1",     color="#1f77b4", capsize=4)
ax.bar(x,     abnm, w, yerr=abns_s, label="abn_R", color="#d62728", capsize=4)
ax.bar(x + w, norm_, w, yerr=nors_, label="nor_R", color="#2ca02c", capsize=4)
for i, m in enumerate(f1m):
    ax.text(i - w, m + 0.04, f"{m:.2f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Score (%)")
ax.set_title("Cat 3 — Noise impact (n=2800, multi-seed mean ± std)")
ax.legend(loc="lower right")
ax.set_ylim(98.5, 100.1)
fig.tight_layout()
fig.savefig(OUT / "cat3_noise_impact.png")
plt.close(fig)
print(f"[saved] cat3_noise_impact.png")


# --- Plot Cat 4: LR tuning ---
# 두 panel: (left) n=2800 apples-to-apples, (right) lr3tie 만 normal_count sweep
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))

labels, f1m, f1s, abnm, abns_v = [], [], [], [], []
for label, grp in cat4_groups.items():
    if not grp:
        continue
    labels.append(label)
    f1m.append(np.mean([r["test_f1"] for r in grp]) * 100)
    f1s.append(np.std([r["test_f1"] for r in grp]) * 100)
    abnm.append(np.mean([r["abn_R"] for r in grp]) * 100)
    abns_v.append(np.std([r["abn_R"] for r in grp]) * 100)

x = np.arange(len(labels))
w = 0.36
colors_f1  = "#1f77b4"
colors_abn = "#d62728"
ax1.bar(x - w/2, f1m,  w, yerr=f1s,    label="F1",    color=colors_f1,  capsize=4)
ax1.bar(x + w/2, abnm, w, yerr=abns_v, label="abn_R", color=colors_abn, capsize=4)
for i, (m, s) in enumerate(zip(f1m, f1s)):
    ax1.text(i - w/2, m + 0.15, f"{m:.2f}", ha="center", fontsize=9, fontweight="bold")
for i, (m, s) in enumerate(zip(abnm, abns_v)):
    ax1.text(i + w/2, m + 0.15, f"{m:.2f}", ha="center", fontsize=9)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("Score (%)")
ax1.set_title("Cat 4a — n=2800 (apples-to-apples)")
ax1.legend(loc="lower left")
ax1.set_ylim(91.0, 101.5)  # lr3tie n=2800 spike (~92) 도 보이게
# spike 화살표
if "v9 lr3tie\n(lr 3e-5)" in cat4_groups and len(labels) >= 4:
    spike_idx = labels.index("v9 lr3tie\n(lr 3e-5)")
    ax1.annotate("LR too high\nspike at n=2800",
                 xy=(spike_idx + w/2, abnm[spike_idx]),
                 xytext=(spike_idx - 0.3, 95),
                 ha="center", fontsize=9, color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728"))

# Right panel: lr3tie sweep over normal_count
xs2 = sorted({r["normal_count"] for r in cat2_v9_tie})
f1_lr3 = []
abn_lr3 = []
for n in xs2:
    grp = [r for r in cat2_v9_tie if r["normal_count"] == n]
    if grp:
        f1_lr3.append(grp[0]["test_f1"] * 100)
        abn_lr3.append(grp[0]["abn_R"] * 100)
ax2.plot(xs2, f1_lr3, "o-", lw=2.5, ms=10, label="F1",    color=colors_f1)
ax2.plot(xs2, abn_lr3, "s--", lw=2, ms=8,  label="abn_R", color=colors_abn)
for x_, y_ in zip(xs2, f1_lr3):
    ax2.text(x_, y_ + 0.4, f"{y_:.2f}", ha="center", fontsize=9, fontweight="bold")
ax2.set_xticks(xs2)
ax2.set_xlabel("Normal samples")
ax2.set_ylabel("Score (%)")
ax2.set_title("Cat 4b — v9 lr3tie sweep (lr 3e-5, single seed s1)")
ax2.legend(loc="lower left")
ax2.set_ylim(91.0, 101.5)

fig.suptitle("Cat 4 — LR tuning on v9 dataset", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "cat4_lr_tuning.png")
plt.close(fig)
print(f"[saved] cat4_lr_tuning.png")

print("\nALL DONE.")
