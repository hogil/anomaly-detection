"""
실험 결과 종합 요약 — 277 runs 전수조사.

logs/*/best_info.json 모두 스캔 → era 분류 → sweep group 별 stats + plot.

- 모든 277 run 의 raw dump → data.csv
- 카테고리별 markdown table 출력 (stdout) → SUMMARY.md 에 paste
- 카테고리별 plot 저장 → cat*.png

Categories
----------
1. dataset_scale       : v8_init 시리즈 (per-class equal scaling)
2. normal_count_sweep  : gm_nor / v8seed multi-seed / v9_lr3tie
3. noise_impact        : v8 vs v9 same hparams
4. lr_tuning           : convnextv2_lr* + imp2_cos_lr + ft_lr + v9_lr / v9lr_bb*
5. focal_gamma         : gm_g + imp2_cos_g + ft_g + v9 era
6. abnormal_weight     : imp2_cos_aw + ft_aw + v9_aw
7. dropout             : imp2_cos_d + ft_d + v9reg_drop
8. backbone            : v9_bb_* (4 backbones × multi-seed)
9. regularization      : v9reg_* (drop/ls/mix/wd/fg)
10. ema_averaging      : v9_ema*, v9_avg*
11. best_selection     : v9_med*, v9_smooth*, v9_tu, v9_tiefix vs v9_ep10

Usage:
    PYTHONIOENCODING=utf-8 python experiment_summary/build_summary.py > experiment_summary/build_log.txt
"""
import json
import csv
import re
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = ROOT / "experiment_summary"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 130,
})


# =============================================================================
# Load
# =============================================================================
def load_runs():
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
                "mtime": info.stat().st_mtime,
                "epoch": bi.get("epoch"),
                "val_f1": bi.get("val_f1"),
                "test_f1": bi.get("test_f1"),
                "test_acc": bi.get("test_acc"),
                "abn_R": tm.get("abnormal", {}).get("recall") if "abnormal" in tm else None,
                "nor_R": tm.get("normal", {}).get("recall") if "normal" in tm else None,
                "lr_bb": hp.get("lr_backbone"),
                "lr_head": hp.get("lr_head"),
                "epochs": hp.get("epochs"),
                "batch_size": hp.get("batch_size"),
                "warmup_epochs": hp.get("warmup_epochs"),
                "weight_decay": hp.get("weight_decay"),
                "patience": hp.get("patience"),
                "min_epochs": hp.get("min_epochs"),
                "smooth_window": hp.get("smooth_window"),
                "ema_decay": hp.get("ema_decay"),
                "focal_gamma": hp.get("focal_gamma"),
                "abnormal_weight": hp.get("abnormal_weight"),
                "dropout": hp.get("dropout"),
                "label_smoothing": hp.get("label_smoothing"),
                "use_mixup": hp.get("mixup"),
                "normal_ratio": hp.get("normal_ratio"),
                "model": hp.get("pretrained") or hp.get("model"),
                "mode": hp.get("mode"),
                "scheduler": hp.get("scheduler"),
            })
        except Exception:
            pass
    return runs


def stats(values):
    arr = np.array([v for v in values if v is not None])
    if len(arr) == 0:
        return None
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def fmt(x, d=2):
    return f"{x*100:.{d}f}" if x is not None else "—"


def header(title):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def md_table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---:"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


# =============================================================================
# Main
# =============================================================================
runs = load_runs()
print(f"Loaded {len(runs)} runs from {LOGS}")

by_name = {r["name"]: r for r in runs}

def find(*names):
    return [by_name[n] for n in names if n in by_name]

def find_prefix(prefix, exclude_substrings=None):
    exclude_substrings = exclude_substrings or []
    return [r for r in runs if r["name"].startswith(prefix)
            and not any(s in r["name"] for s in exclude_substrings)]


# =============================================================================
# CSV dump (모든 run)
# =============================================================================
csv_path = OUT / "data.csv"
fields = ["name", "mtime", "mode", "epoch", "val_f1", "test_f1", "test_acc",
          "abn_R", "nor_R", "lr_bb", "lr_head", "epochs", "batch_size",
          "focal_gamma", "abnormal_weight", "dropout", "label_smoothing",
          "use_mixup", "normal_ratio", "ema_decay", "smooth_window", "min_epochs"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in runs:
        rr = dict(r)
        rr["mtime"] = datetime.fromtimestamp(r["mtime"]).isoformat()
        w.writerow(rr)
print(f"[saved] {csv_path}  ({len(runs)} rows)")


# =============================================================================
# Cat 1 — Per-class equal scaling (v8_init series)
# =============================================================================
header("Category 1 — Per-class equal scaling (v8_init series, single seed)")

n_map = {"v8_init": 700, "v8_init_n2": 1400, "v8_init_n3": 2100,
         "v8_init_n4": 2800, "v8_init_n5": 3500}
cat1 = []
for n, c in n_map.items():
    if n in by_name:
        r = dict(by_name[n])
        r["normal_count"] = c
        cat1.append(r)
cat1.sort(key=lambda r: r["normal_count"])

rows = []
for r in cat1:
    rows.append([
        f"{r['normal_count']}",
        r["name"],
        fmt(r["test_f1"]),
        fmt(r["abn_R"]),
        fmt(r["nor_R"]),
        f"{r['epoch']}",
    ])
md_table(["normal", "run", "F1 %", "abn_R %", "nor_R %", "best_ep"], rows)

# Plot
fig, ax = plt.subplots(figsize=(7.5, 4.5))
xs = [r["normal_count"] for r in cat1]
ax.plot(xs, [r["test_f1"]*100 for r in cat1], "o-", lw=2, ms=9, label="F1", color="#1f77b4")
ax.plot(xs, [r["abn_R"]*100 for r in cat1], "s--", lw=1.5, ms=7, label="abn_R", color="#d62728")
ax.plot(xs, [r["nor_R"]*100 for r in cat1], "^--", lw=1.5, ms=7, label="nor_R", color="#2ca02c")
ax.set_xlabel("Normal samples")
ax.set_ylabel("Score (%)")
ax.set_title("Cat 1 — v8_init dataset scale (single seed)")
ax.set_xticks(xs)
ax.legend()
ax.set_ylim(98.0, 100.05)
fig.tight_layout()
fig.savefig(OUT / "cat1_dataset_scale.png")
plt.close(fig)
print("[plot] cat1_dataset_scale.png")


# =============================================================================
# Cat 2 — Normal count sweeps (3 series)
# =============================================================================
header("Category 2 — Normal count sweeps")

# 2a: gm_nor (older era, single seed each)
gm_nor_map = {"gm_nor100": 100, "gm_nor200": 200, "gm_nor350": 350,
              "gm_nor500": 500, "gm_nor700": 700, "gm_nor1000": 1000,
              "gm_nor1500": 1500, "gm_nor2000": 2000}
cat2a = []
for name, c in gm_nor_map.items():
    if name in by_name:
        r = dict(by_name[name]); r["normal_count"] = c
        cat2a.append(r)
cat2a.sort(key=lambda r: r["normal_count"])

print("\n## Cat 2a — gm_nor sweep (older era, focal gamma fixed, single seed)\n")
rows = []
for r in cat2a:
    rows.append([
        f"{r['normal_count']}",
        r["name"],
        fmt(r["test_f1"]),
        fmt(r["abn_R"]),
        fmt(r["nor_R"]),
        f"{r['focal_gamma']}",
    ])
md_table(["normal", "run", "F1 %", "abn_R %", "nor_R %", "γ"], rows)

# 2b: v8seed_n* (5 seeds × 5 counts = 25)
def parse_n_seed(name):
    m = re.search(r"_n(\d+)_s(\d+)", name)
    if m: return int(m.group(1)), int(m.group(2))
    return None, None

cat2b_runs = find_prefix("v8seed_n")
for r in cat2b_runs:
    n, s = parse_n_seed(r["name"])
    r["normal_count"] = n
    r["seed"] = s

print("\n## Cat 2b — v8seed multi-seed sweep (5 seeds × 5 normal counts = 25 runs)\n")
counts_2b = sorted({r["normal_count"] for r in cat2b_runs if r["normal_count"]})
rows = []
for n in counts_2b:
    g = [r for r in cat2b_runs if r["normal_count"] == n]
    sf1 = stats([r["test_f1"] for r in g])
    sab = stats([r["abn_R"] for r in g])
    snr = stats([r["nor_R"] for r in g])
    if sf1:
        rows.append([
            f"{n}",
            f"{sf1['n']}",
            f"{sf1['mean']*100:.2f} ± {sf1['std']*100:.2f}",
            f"{sab['mean']*100:.2f} ± {sab['std']*100:.2f}",
            f"{snr['mean']*100:.2f} ± {snr['std']*100:.2f}",
        ])
md_table(["normal", "n_seeds", "F1 %", "abn_R %", "nor_R %"], rows)

# 2c: v9_lr3tie (single seed s1)
cat2c_runs = find_prefix("v9_lr3tie_n")
for r in cat2c_runs:
    n, s = parse_n_seed(r["name"])
    r["normal_count"] = n
    r["seed"] = s
cat2c_runs = [r for r in cat2c_runs if r["normal_count"]]
cat2c_runs.sort(key=lambda r: r["normal_count"])

print("\n## Cat 2c — v9_lr3tie sweep (lr 3e-5, single seed s1)\n")
rows = []
for r in cat2c_runs:
    flag = " ⚠️" if r["test_f1"] and r["test_f1"] < 0.99 else ""
    rows.append([
        f"{r['normal_count']}",
        r["name"],
        fmt(r["test_f1"]) + flag,
        fmt(r["abn_R"]),
        fmt(r["nor_R"]),
    ])
md_table(["normal", "run", "F1 %", "abn_R %", "nor_R %"], rows)

# Plot Cat 2 — overlay all 3 series
fig, ax = plt.subplots(figsize=(10, 5.5))
# 2a
xs2a = [r["normal_count"] for r in cat2a]
f1_2a = [r["test_f1"]*100 for r in cat2a]
ax.plot(xs2a, f1_2a, "v-", lw=2, ms=8, label="2a · gm_nor (older era, 1 seed)", color="#888888", alpha=0.85)

# 2b: mean ± std
f1m_2b, f1s_2b = [], []
for n in counts_2b:
    g = [r for r in cat2b_runs if r["normal_count"] == n]
    f1m_2b.append(np.mean([r["test_f1"] for r in g])*100)
    f1s_2b.append(np.std([r["test_f1"] for r in g])*100)
ax.errorbar(counts_2b, f1m_2b, yerr=f1s_2b, fmt="o-", lw=2.5, ms=10, capsize=5,
            label="2b · v8seed (v8 data, 5 seeds)", color="#1f77b4")

# 2c
xs2c = [r["normal_count"] for r in cat2c_runs]
f1_2c = [r["test_f1"]*100 for r in cat2c_runs]
ax.plot(xs2c, f1_2c, "s--", lw=2, ms=9, label="2c · v9_lr3tie (v9 data, 1 seed)", color="#d62728")

# annotate spike
spike = next((r for r in cat2c_runs if r["test_f1"] < 0.99), None)
if spike:
    ax.annotate("LR spike\n(see Cat 11)",
                xy=(spike["normal_count"], spike["test_f1"]*100),
                xytext=(spike["normal_count"]-300, 95),
                ha="center", fontsize=9, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))

ax.set_xlabel("Normal samples (training)")
ax.set_ylabel("Test F1 (%)")
ax.set_title("Cat 2 — Normal count sweeps across 3 eras")
ax.legend(loc="lower left")
ax.set_ylim(85, 100.5)
fig.tight_layout()
fig.savefig(OUT / "cat2_normal_count.png")
plt.close(fig)
print("[plot] cat2_normal_count.png")


# =============================================================================
# Cat 3 — Noise impact (v8 vs v9 at n=2800)
# =============================================================================
header("Category 3 — Noise impact (v8 vs v9, n=2800, multi-seed)")

cat3_v8       = [r for r in cat2b_runs if r["normal_count"] == 2800]
cat3_v9_tief  = find_prefix("v9_tiefix_n2800_s")
cat3_v9_ep10  = find_prefix("v9_ep10_n2800_s")

def grpstat(label, grp):
    sf1 = stats([r["test_f1"] for r in grp])
    sab = stats([r["abn_R"] for r in grp])
    snr = stats([r["nor_R"] for r in grp])
    if not sf1:
        return None
    return [label, str(sf1["n"]),
            f"{sf1['mean']*100:.2f} ± {sf1['std']*100:.2f}",
            f"{sab['mean']*100:.2f} ± {sab['std']*100:.2f}",
            f"{snr['mean']*100:.2f} ± {snr['std']*100:.2f}"]

rows = []
for label, grp in [
    ("v8 baseline (lr 5e-5)",          cat3_v8),
    ("v9 tie-fix (+25% noise, 5e-5)",  cat3_v9_tief),
    ("v9 ep10 (+25% noise, 2e-5)",     cat3_v9_ep10),
]:
    row = grpstat(label, grp)
    if row: rows.append(row)
md_table(["dataset", "n_seeds", "F1 %", "abn_R %", "nor_R %"], rows)

fig, ax = plt.subplots(figsize=(8, 5))
labels, f1m, f1s, abm, abs_, norm_, nors_ = [], [], [], [], [], [], []
for label, grp in [("v8\n(baseline)", cat3_v8),
                   ("v9 tie-fix\n(+25% noise)", cat3_v9_tief),
                   ("v9 ep10\n(+25% noise)", cat3_v9_ep10)]:
    if not grp: continue
    labels.append(label)
    f1m.append(np.mean([r["test_f1"] for r in grp])*100)
    f1s.append(np.std([r["test_f1"] for r in grp])*100)
    abm.append(np.mean([r["abn_R"] for r in grp])*100)
    abs_.append(np.std([r["abn_R"] for r in grp])*100)
    norm_.append(np.mean([r["nor_R"] for r in grp])*100)
    nors_.append(np.std([r["nor_R"] for r in grp])*100)

x = np.arange(len(labels))
w = 0.27
ax.bar(x - w, f1m, w, yerr=f1s, label="F1", color="#1f77b4", capsize=4)
ax.bar(x, abm, w, yerr=abs_, label="abn_R", color="#d62728", capsize=4)
ax.bar(x + w, norm_, w, yerr=nors_, label="nor_R", color="#2ca02c", capsize=4)
for i, m in enumerate(f1m):
    ax.text(i - w, m + 0.05, f"{m:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Score (%)")
ax.set_title("Cat 3 — Noise impact (n=2800, multi-seed mean ± std)")
ax.legend(loc="lower right")
ax.set_ylim(98.5, 100.1)
fig.tight_layout()
fig.savefig(OUT / "cat3_noise_impact.png")
plt.close(fig)
print("[plot] cat3_noise_impact.png")


# =============================================================================
# Cat 4 — LR tuning (era별)
# =============================================================================
header("Category 4 — LR tuning (전 era)")

# convnextv2_lr* (older era)
cat4_old_map = {"convnextv2_lr1e4": 1e-4, "convnextv2_lr2e5": 2e-5, "convnextv2_lr5e5": 5e-5}
cat4_old = []
for name, lr in cat4_old_map.items():
    if name in by_name:
        r = dict(by_name[name]); r["lr"] = lr
        cat4_old.append(r)

# imp2_cos_lr*
cat4_imp = find_prefix("imp2_cos_lr")
for r in cat4_imp:
    r["lr"] = r["lr_bb"]

# ft_lr*
cat4_ft = find_prefix("ft_lr")
for r in cat4_ft:
    r["lr"] = r["lr_bb"]

# v9 era LR experiments
v9_lr_runs = (find_prefix("v9_lr") + find_prefix("v9lr_") + find_prefix("v9_ep10_n2800"))
v9_lr_unique = {r["name"]: r for r in v9_lr_runs}
cat4_v9 = list(v9_lr_unique.values())
for r in cat4_v9:
    r["lr"] = r["lr_bb"]

print(f"\n## Cat 4a — convnextv2_lr* (early era)\n")
rows = []
for r in sorted(cat4_old, key=lambda r: r["lr"]):
    rows.append([f"{r['lr']:.0e}", r["name"], fmt(r["test_f1"]), fmt(r["abn_R"]), fmt(r["nor_R"])])
md_table(["lr", "run", "F1 %", "abn_R %", "nor_R %"], rows)

print(f"\n## Cat 4b — ft_lr* (finetune era)\n")
rows = []
for r in sorted(cat4_ft, key=lambda r: r["lr"] or 0):
    rows.append([f"{r['lr']}", r["name"], fmt(r["test_f1"]), fmt(r["abn_R"]), fmt(r["nor_R"])])
md_table(["lr", "run", "F1 %", "abn_R %", "nor_R %"], rows)

print(f"\n## Cat 4c — imp2_cos_lr*\n")
rows = []
for r in sorted(cat4_imp, key=lambda r: r["lr"] or 0):
    rows.append([f"{r['lr']}", r["name"], fmt(r["test_f1"]), fmt(r["abn_R"]), fmt(r["nor_R"])])
md_table(["lr", "run", "F1 %", "abn_R %", "nor_R %"], rows)

print(f"\n## Cat 4d — v9 era LR sweep (mixed configs)\n")
rows = []
for r in sorted(cat4_v9, key=lambda r: (r["lr"] or 0, r["name"])):
    flag = " ⚠️" if r["test_f1"] and r["test_f1"] < 0.99 else ""
    rows.append([f"{r['lr']}", r["name"], fmt(r["test_f1"]) + flag, fmt(r["abn_R"]), fmt(r["nor_R"])])
md_table(["lr", "run", "F1 %", "abn_R %", "nor_R %"], rows)

# Plot: F1 vs LR across all eras (scatter)
fig, ax = plt.subplots(figsize=(10, 5.5))
def scat(grp, label, color, marker):
    if not grp: return
    xs = [r["lr"] for r in grp if r.get("lr")]
    ys = [r["test_f1"]*100 for r in grp if r.get("lr") and r["test_f1"]]
    if xs and ys:
        ax.scatter(xs, ys, label=label, color=color, marker=marker, s=70, alpha=0.85, edgecolor="black", linewidth=0.5)

scat(cat4_old, "convnextv2_lr* (early)", "#888888", "v")
scat(cat4_imp, "imp2_cos_lr*", "#9467bd", "D")
scat(cat4_ft,  "ft_lr*",        "#e377c2", "^")
scat(cat4_v9,  "v9 era",        "#1f77b4", "o")

ax.set_xscale("log")
ax.set_xlabel("Backbone learning rate (peak)")
ax.set_ylabel("Test F1 (%)")
ax.set_title("Cat 4 — LR tuning across all eras")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(80, 100.5)
fig.tight_layout()
fig.savefig(OUT / "cat4_lr_tuning.png")
plt.close(fig)
print("[plot] cat4_lr_tuning.png")


# =============================================================================
# Cat 5 — Focal gamma sweep
# =============================================================================
header("Category 5 — Focal gamma sweep")

# gm_g* (without _n suffix)
gamma_runs = []
for r in find_prefix("gm_g"):
    if "_n" in r["name"]:  # combo skip
        continue
    m = re.match(r"gm_g(\d+)", r["name"])
    if m:
        g = int(m.group(1)) / 10  # gm_g25 → 2.5
        rr = dict(r); rr["gamma"] = g
        gamma_runs.append(rr)

# imp2_cos_g*
for r in find_prefix("imp2_cos_g"):
    m = re.match(r"imp2_cos_g(\d+)", r["name"])
    if m:
        g = int(m.group(1)) / 10
        rr = dict(r); rr["gamma"] = g
        gamma_runs.append(rr)

# ft_g*
for r in find_prefix("ft_g"):
    m = re.match(r"ft_g(\d+\.\d+)", r["name"])
    if m:
        g = float(m.group(1))
        rr = dict(r); rr["gamma"] = g
        gamma_runs.append(rr)

# v9_aw1 / v9_aw15 등은 aw 카테고리

gamma_runs.sort(key=lambda r: r["gamma"])
rows = []
for r in gamma_runs:
    rows.append([f"{r['gamma']:.1f}", r["name"], fmt(r["test_f1"]), fmt(r["abn_R"]), fmt(r["nor_R"])])
md_table(["γ", "run", "F1 %", "abn_R %", "nor_R %"], rows)

if gamma_runs:
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [r["gamma"] for r in gamma_runs]
    ax.scatter(xs, [r["test_f1"]*100 for r in gamma_runs], color="#1f77b4", s=70, label="F1")
    ax.scatter(xs, [r["abn_R"]*100 for r in gamma_runs], color="#d62728", s=50, marker="s", label="abn_R")
    ax.set_xlabel("Focal gamma")
    ax.set_ylabel("Score (%)")
    ax.set_title("Cat 5 — Focal gamma sweep (mixed eras)")
    ax.legend()
    ax.set_ylim(85, 100)
    fig.tight_layout()
    fig.savefig(OUT / "cat5_focal_gamma.png")
    plt.close(fig)
    print("[plot] cat5_focal_gamma.png")


# =============================================================================
# Cat 6 — Abnormal weight
# =============================================================================
header("Category 6 — Abnormal weight sweep")

aw_runs = []
# imp2_cos_aw*
for r in find_prefix("imp2_cos_aw"):
    m = re.match(r"imp2_cos_aw(\d+)", r["name"])
    if m:
        aw = int(m.group(1))  # aw1, aw2, aw5, aw7, aw10
        rr = dict(r); rr["aw"] = aw
        aw_runs.append(rr)
# ft_aw*
for r in find_prefix("ft_aw"):
    m = re.match(r"ft_aw(\d+\.\d+)", r["name"])
    if m:
        rr = dict(r); rr["aw"] = float(m.group(1))
        aw_runs.append(rr)
# v9_aw*
for r in find_prefix("v9_aw"):
    name = r["name"]
    if "n700" in name and "aw1" in name:
        rr = dict(r); rr["aw"] = 1.0; aw_runs.append(rr)
    elif "aw15" in name:
        rr = dict(r); rr["aw"] = 1.5; aw_runs.append(rr)

aw_runs.sort(key=lambda r: r["aw"])
rows = []
for r in aw_runs:
    rows.append([f"{r['aw']:.2f}", r["name"], fmt(r["test_f1"]), fmt(r["abn_R"]), fmt(r["nor_R"])])
md_table(["abnormal_weight", "run", "F1 %", "abn_R %", "nor_R %"], rows)

if aw_runs:
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [r["aw"] for r in aw_runs]
    ax.scatter(xs, [r["test_f1"]*100 for r in aw_runs], color="#1f77b4", s=70, label="F1")
    ax.scatter(xs, [r["abn_R"]*100 for r in aw_runs], color="#d62728", s=50, marker="s", label="abn_R")
    ax.set_xlabel("abnormal_weight")
    ax.set_ylabel("Score (%)")
    ax.set_title("Cat 6 — abnormal_weight sweep (mixed eras)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cat6_abnormal_weight.png")
    plt.close(fig)
    print("[plot] cat6_abnormal_weight.png")


# =============================================================================
# Cat 7 — Dropout
# =============================================================================
header("Category 7 — Dropout sweep")

drop_runs = []
for r in find_prefix("imp2_cos_d"):
    m = re.match(r"imp2_cos_d(\d+)", r["name"])
    if m:
        d = int(m.group(1)) / 10  # d04 → 0.4
        rr = dict(r); rr["dropout_v"] = d
        drop_runs.append(rr)
for r in find_prefix("ft_d"):
    m = re.match(r"ft_d(\d+)", r["name"])
    if m:
        d = int(m.group(1)) / 10
        rr = dict(r); rr["dropout_v"] = d
        drop_runs.append(rr)
for r in find_prefix("v9reg_drop"):
    m = re.match(r"v9reg_drop(\d+)", r["name"])
    if m:
        d = int(m.group(1)) / 10
        rr = dict(r); rr["dropout_v"] = d
        drop_runs.append(rr)

drop_runs.sort(key=lambda r: r["dropout_v"])
rows = []
for r in drop_runs:
    rows.append([f"{r['dropout_v']:.1f}", r["name"], fmt(r["test_f1"]), fmt(r["abn_R"]), fmt(r["nor_R"])])
md_table(["dropout", "run", "F1 %", "abn_R %", "nor_R %"], rows)

if drop_runs:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = [r["dropout_v"] for r in drop_runs]
    ax.scatter(xs, [r["test_f1"]*100 for r in drop_runs], color="#1f77b4", s=70, label="F1")
    ax.scatter(xs, [r["abn_R"]*100 for r in drop_runs], color="#d62728", s=50, marker="s", label="abn_R")
    ax.set_xlabel("dropout")
    ax.set_ylabel("Score (%)")
    ax.set_title("Cat 7 — dropout sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cat7_dropout.png")
    plt.close(fig)
    print("[plot] cat7_dropout.png")


# =============================================================================
# Cat 8 — Backbone (v9_bb_*)
# =============================================================================
header("Category 8 — Backbone comparison (v9_bb_*)")

bb_runs = find_prefix("v9_bb_")
# group by backbone name
def parse_bb(name):
    # v9_bb_efficientnetv2_s_s42 → efficientnetv2_s, s42
    m = re.match(r"v9_bb_(.+)_s(\d+)$", name)
    if m: return m.group(1), int(m.group(2))
    return None, None

bb_groups = {}
for r in bb_runs:
    bb, s = parse_bb(r["name"])
    if bb:
        bb_groups.setdefault(bb, []).append(r)

rows = []
for bb, grp in sorted(bb_groups.items()):
    sf1 = stats([r["test_f1"] for r in grp])
    sab = stats([r["abn_R"] for r in grp])
    if sf1:
        rows.append([
            bb, str(sf1["n"]),
            f"{sf1['mean']*100:.2f} ± {sf1['std']*100:.2f}",
            f"{sab['mean']*100:.2f} ± {sab['std']*100:.2f}",
        ])
md_table(["backbone", "n_seeds", "F1 %", "abn_R %"], rows)

if bb_groups:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = sorted(bb_groups.keys())
    f1m = [np.mean([r["test_f1"] for r in bb_groups[k]])*100 for k in labels]
    f1s = [np.std([r["test_f1"] for r in bb_groups[k]])*100 for k in labels]
    abm = [np.mean([r["abn_R"] for r in bb_groups[k]])*100 for k in labels]
    abs_ = [np.std([r["abn_R"] for r in bb_groups[k]])*100 for k in labels]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, f1m, w, yerr=f1s, label="F1", color="#1f77b4", capsize=4)
    ax.bar(x + w/2, abm, w, yerr=abs_, label="abn_R", color="#d62728", capsize=4)
    for i, m in enumerate(f1m):
        ax.text(i - w/2, m + 0.07, f"{m:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("Cat 8 — Backbone comparison (v9 dataset, multi-seed)")
    ax.legend()
    ax.set_ylim(98, 100.5)
    fig.tight_layout()
    fig.savefig(OUT / "cat8_backbone.png")
    plt.close(fig)
    print("[plot] cat8_backbone.png")


# =============================================================================
# Cat 9 — Regularization (v9reg_*)
# =============================================================================
header("Category 9 — Regularization (v9reg_* on v9 data, n=2800, seed 42)")

reg_runs = find_prefix("v9reg_")
rows = []
for r in sorted(reg_runs, key=lambda r: r["name"]):
    rows.append([
        r["name"].replace("v9reg_", "").replace("_n2800_s42", ""),
        fmt(r["test_f1"]), fmt(r["abn_R"]), fmt(r["nor_R"]),
        f"{r['focal_gamma']}", f"{r['dropout']}", f"{r.get('label_smoothing','—')}",
    ])
md_table(["variant", "F1 %", "abn_R %", "nor_R %", "γ", "dropout", "ls"], rows)


# =============================================================================
# Cat 10 — EMA / averaging
# =============================================================================
header("Category 10 — EMA / weight averaging (v9 era)")

ema_runs = find_prefix("v9_ema") + find_prefix("v9_avg")
rows = []
for r in sorted(ema_runs, key=lambda r: r["name"]):
    rows.append([
        r["name"], fmt(r["test_f1"]), fmt(r["abn_R"]), fmt(r["nor_R"]),
        f"{r.get('ema_decay','—')}", f"{r['epochs']}",
    ])
md_table(["run", "F1 %", "abn_R %", "nor_R %", "ema_decay", "epochs"], rows)


# =============================================================================
# Cat 11 — Best selection / smoothing experiments
# =============================================================================
header("Category 11 — Best selection / smoothing (v9 era)")

bs_runs = (find_prefix("v9_med") + find_prefix("v9_smooth") + find_prefix("v9_tu")
           + find_prefix("v9_lrA") + find_prefix("v9_lrB") + find_prefix("v9_v8code")
           + find_prefix("v9_old") + find_prefix("v9_ref"))
bs_runs = list({r["name"]: r for r in bs_runs}.values())
rows = []
for r in sorted(bs_runs, key=lambda r: r["name"]):
    rows.append([
        r["name"], fmt(r["test_f1"]), fmt(r["abn_R"]), fmt(r["nor_R"]),
        f"{r.get('smooth_window','—')}", f"{r.get('min_epochs','—')}", f"{r['epochs']}",
    ])
md_table(["run", "F1 %", "abn_R %", "nor_R %", "smooth", "min_ep", "epochs"], rows)


# =============================================================================
# Top runs (binary mode, v9 era)
# =============================================================================
header("Top 10 by F1 (binary mode, all eras)")

binary_runs = [r for r in runs if r["mode"] == "binary" and r["test_f1"]]
top_f1 = sorted(binary_runs, key=lambda r: -r["test_f1"])[:10]
rows = []
for r in top_f1:
    rows.append([
        r["name"], fmt(r["test_f1"], 4), fmt(r["abn_R"], 4), fmt(r["nor_R"], 4),
        f"{r['lr_bb']}", f"{r['focal_gamma']}", f"{r.get('normal_ratio','—')}",
    ])
md_table(["run", "F1 %", "abn_R %", "nor_R %", "lr_bb", "γ", "n_ratio"], rows)


print("\n\nALL DONE.")
