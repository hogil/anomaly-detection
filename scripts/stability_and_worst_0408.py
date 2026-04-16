"""Oscillation / worst-seed / N-scaling analysis on runs since 2026-04-08.

Does NOT require retraining — reads history.json + best_info.json per run.

Outputs three tables to validations/stability_and_worst_0408.txt:
  A. Oscillation metrics by condition (gc, sw, wd, reg, lr) — worst epoch drop,
     val_f1 std in convergence window, count of drops>0.01, grad_norm max.
  B. Worst/min-seed per condition — min F1, max FN, spread.
  C. N-scaling — log-log error vs N with slope.
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

sys.stdout.reconfigure(encoding="utf-8")

LOGS = Path("D:/project/anomaly-detection/logs")
OUT = Path("D:/project/anomaly-detection/validations/stability_and_worst_0408.txt")
CUTOFF = "260408"


def load_run(p: Path):
    info_path = p / "best_info.json"
    hist_path = p / "history.json"
    if not (info_path.exists() and hist_path.exists()):
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    tm = info.get("test_metrics") or {}
    n, a = tm.get("normal") or {}, tm.get("abnormal") or {}
    n_c, a_c = n.get("count") or 0, a.get("count") or 0
    n_r, a_r = n.get("recall"), a.get("recall")
    fp = round(n_c * (1 - n_r)) if n_r is not None else None
    fn = round(a_c * (1 - a_r)) if a_r is not None else None

    # oscillation from val_f1 / val_loss after epoch 5 (convergence window)
    ep_after = [e for e in hist if e.get("epoch", 0) >= 5]
    vf = [e.get("val_f1") for e in ep_after if e.get("val_f1") is not None]
    vl = [e.get("val_loss") for e in ep_after if e.get("val_loss") is not None]
    gn = [e.get("grad_norm_max") for e in hist if e.get("grad_norm_max") is not None]

    max_f1_drop = 0.0
    drops_gt_01 = 0
    for i in range(1, len(vf)):
        d = vf[i - 1] - vf[i]
        if d > max_f1_drop:
            max_f1_drop = d
        if d > 0.01:
            drops_gt_01 += 1
    max_loss_jump = 0.0
    for i in range(1, len(vl)):
        j = vl[i] - vl[i - 1]
        if j > max_loss_jump:
            max_loss_jump = j
    vf_std = stdev(vf) if len(vf) > 1 else 0.0

    return {
        "run": p.name,
        "tag": p.name[14:] if len(p.name) > 14 else p.name,
        "test_f1": info.get("test_f1"),
        "val_f1": info.get("val_f1"),
        "fn": fn,
        "fp": fp,
        "epoch": info.get("epoch"),
        "vf_std": vf_std,
        "max_f1_drop": max_f1_drop,
        "max_loss_jump": max_loss_jump,
        "drops_gt_01": drops_gt_01,
        "grad_max": max(gn) if gn else 0.0,
        "n_epochs": len(hist),
    }


rows = []
for p in sorted(LOGS.iterdir()):
    if not (p.is_dir() and p.name[:6].isdigit() and p.name[:6] >= CUTOFF):
        continue
    r = load_run(p)
    if r and r["test_f1"] is not None:
        rows.append(r)

RE_N = re.compile(r"_n(\d+)(?:_|$)")
RE_PC = re.compile(r"_pc(\d+)(?:_|$)")
RE_LR = re.compile(r"_lr(\d+e\d+)(?:_|$)")
RE_WARM = re.compile(r"_lrwarm(\d+)(?:_|$)")
RE_GC = re.compile(r"_gc(\d+p?\d*)(?:_|$)")
RE_WD = re.compile(r"_wd(\d+)(?:_|$)")
RE_REG = re.compile(r"_reg([a-z]+\d+)(?:_|$)")
RE_SW = re.compile(r"_sw(\d+[a-z]+)(?:_|$)")


def extract(tag):
    return {
        "n": int(RE_N.search(tag).group(1)) if RE_N.search(tag) else None,
        "pc": int(RE_PC.search(tag).group(1)) if RE_PC.search(tag) else None,
        "lr": RE_LR.search(tag).group(1) if RE_LR.search(tag) else None,
        "warm": int(RE_WARM.search(tag).group(1)) if RE_WARM.search(tag) else None,
        "gc": RE_GC.search(tag).group(1) if RE_GC.search(tag) else None,
        "wd": RE_WD.search(tag).group(1) if RE_WD.search(tag) else None,
        "reg": RE_REG.search(tag).group(1) if RE_REG.search(tag) else None,
        "sw": RE_SW.search(tag).group(1) if RE_SW.search(tag) else None,
    }


for r in rows:
    r.update(extract(r["tag"]))

lines = []
lines.append(f"STABILITY & WORST-SEED ANALYSIS  (runs since {CUTOFF}, n={len(rows)})")

# ==== A. Oscillation by condition ====
def oscillation_table(title, filt, key_name):
    groups = defaultdict(list)
    for r in rows:
        if filt(r) and r.get(key_name) is not None:
            groups[r[key_name]].append(r)
    if not groups:
        return []
    L = [f"\n### {title}"]
    L.append(
        f"{'lvl':<8}{'n':>3}"
        f"{'F1_min':>9}{'F1_med':>9}{'F1_max':>9}"
        f"{'FN_max':>7}{'FP_max':>7}"
        f"{'vf_std':>9}{'maxDrop':>9}{'lossJmp':>9}{'drops>1%':>9}{'gradMax':>9}"
    )
    L.append("-" * 99)

    def skey(k):
        try:
            return (0, float(str(k).replace("p", ".").replace("e", "E")))
        except Exception:
            return (1, str(k))

    for k in sorted(groups.keys(), key=skey):
        v = groups[k]
        v = sorted(v, key=lambda r: r["test_f1"])
        tf = [r["test_f1"] for r in v]
        fns = [r["fn"] or 0 for r in v]
        fps = [r["fp"] or 0 for r in v]
        med = sorted(tf)[len(tf) // 2]
        L.append(
            f"{str(k):<8}{len(v):>3}"
            f"{min(tf):>9.4f}{med:>9.4f}{max(tf):>9.4f}"
            f"{max(fns):>7}{max(fps):>7}"
            f"{mean(r['vf_std'] for r in v):>9.4f}"
            f"{max(r['max_f1_drop'] for r in v):>9.4f}"
            f"{max(r['max_loss_jump'] for r in v):>9.3f}"
            f"{mean(r['drops_gt_01'] for r in v):>9.2f}"
            f"{mean(r['grad_max'] for r in v):>9.1f}"
        )
    return L


lines += oscillation_table(
    "A1. Gradient clip (fresh0412_v11, n2800)  — stability check",
    lambda r: r["tag"].startswith("fresh0412_v11_gc"),
    "gc",
)
lines += oscillation_table(
    "A2. Gradient clip (fresh0413_reset_v11, n700)",
    lambda r: r["tag"].startswith("fresh0413_reset_v11_gc"),
    "gc",
)
lines += oscillation_table(
    "A3. Gradient clip (v9mid6_win, n700)",
    lambda r: r["tag"].startswith("v9mid6_win_gc"),
    "gc",
)
lines += oscillation_table(
    "A4. Val smoothing (fresh0413_reset_v11, n700)",
    lambda r: r["tag"].startswith("fresh0413_reset_v11_sw"),
    "sw",
)
lines += oscillation_table(
    "A5. Weight decay (fresh0413_reset_v11, n700)",
    lambda r: r["tag"].startswith("fresh0413_reset_v11_wd"),
    "wd",
)
lines += oscillation_table(
    "A6. Regularization (fresh0413_reset_v11, n700)",
    lambda r: r["tag"].startswith("fresh0413_reset_v11_reg"),
    "reg",
)
lines += oscillation_table(
    "A7. Learning rate (fresh0413_reset_v11, n700)",
    lambda r: r["tag"].startswith("fresh0413_reset_v11_lr")
    and not r["tag"].startswith("fresh0413_reset_v11_lrwarm")
    and not r["tag"].startswith("fresh0413_reset_v11_rescue"),
    "lr",
)
lines += oscillation_table(
    "A8. Learning rate (fresh0412_v11, n2800)",
    lambda r: r["tag"].startswith("fresh0412_v11_lr")
    and not r["tag"].startswith("fresh0412_v11_lrwarm"),
    "lr",
)

# ==== B. N scaling log-log ====
def nscale_table(title, filt):
    groups = defaultdict(list)
    for r in rows:
        if filt(r) and r.get("n") is not None:
            groups[r["n"]].append(r)
    if len(groups) < 2:
        return []
    L = [f"\n### {title}"]
    L.append(
        f"{'N':>5}{'seeds':>7}{'F1_min':>9}{'F1_med':>9}{'F1_mean':>10}"
        f"{'err_mean':>10}{'log10_N':>9}{'log10_err':>11}{'FN_mean':>9}"
    )
    L.append("-" * 80)
    pts = []
    for n in sorted(groups.keys()):
        v = groups[n]
        tf = sorted([r["test_f1"] for r in v])
        err = [1 - f for f in tf]
        err_mean = mean(err)
        med = tf[len(tf) // 2]
        fns = [r["fn"] or 0 for r in v]
        L.append(
            f"{n:>5}{len(v):>7}{min(tf):>9.4f}{med:>9.4f}{mean(tf):>10.4f}"
            f"{err_mean:>10.5f}{math.log10(n):>9.3f}{math.log10(err_mean) if err_mean>0 else float('-inf'):>11.3f}"
            f"{mean(fns):>9.2f}"
        )
        if err_mean > 0:
            pts.append((math.log10(n), math.log10(err_mean)))
    if len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx, my = mean(xs), mean(ys)
        num = sum((x - mx) * (y - my) for x, y in pts)
        den = sum((x - mx) ** 2 for x in xs)
        slope = num / den if den else 0
        L.append(f"  power-law slope (log10 err vs log10 N) = {slope:+.3f}")
        L.append(
            "  interpretation: slope ≈ 0 → saturation, -0.3~-0.5 typical scaling, more negative → still improving"
        )
    return L


lines += nscale_table(
    "B1. N-scaling fresh0413_reset_v11 (plain)",
    lambda r: r["tag"].startswith("fresh0413_reset_v11_n")
    and r["pc"] is None and r["lr"] is None and r["gc"] is None
    and r["warm"] is None and r["wd"] is None and r["reg"] is None and r["sw"] is None,
)
lines += nscale_table(
    "B2. N-scaling fresh0412_v11 (plain)",
    lambda r: r["tag"].startswith("fresh0412_v11_n")
    and r["lr"] is None and r["gc"] is None and r["warm"] is None and r["sw"] is None,
)
lines += nscale_table(
    "B3. N-scaling v11_ (pre-reset, broken)",
    lambda r: r["tag"].startswith("v11_n") and "baseline" not in r["tag"],
)
lines += nscale_table(
    "B4. pc-scaling fresh0413_reset_v11_pc",
    lambda r: r["tag"].startswith("fresh0413_reset_v11_pc") and r["pc"] is not None,
)


OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nSaved to {OUT}")
