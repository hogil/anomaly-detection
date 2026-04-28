"""Aggregate vd080_bc (basecase gc × sw) sweep results.

Reads logs/*vd080_bc_* runs and produces:
  - per-(gc,sw) mean±std test_F1, FN mean, FP mean, peak_ep (from test_history.json)
  - best 10 runs
  - VAL_PEAK vs NEW_BEST analysis (which ep peaked first on raw val_f1)

Writes to validations/vd080_bc_summary.txt
"""
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("D:/project/anomaly-detection")
LOGS = ROOT / "logs"
OUT = ROOT / "validations" / "vd080_bc_summary.txt"

PATTERN = re.compile(
    r"^\d{6}_\d{6}_vd080_bc_gc(?P<gc>[^_]+)_sw(?P<sw>[^_]+)_n(?P<n>\d+)_s(?P<seed>\d+)"
)


def parse_run(d: Path):
    m = PATTERN.match(d.name)
    if not m:
        return None
    bi_path = d / "best_info.json"
    if not bi_path.exists():
        return None
    try:
        bi = json.loads(bi_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    tm = bi.get("test_metrics") or {}
    a = tm.get("abnormal") or {}
    n_ = tm.get("normal") or {}
    abn_r = a.get("recall")
    nor_r = n_.get("recall")
    a_count = a.get("count") or 0
    n_count = n_.get("count") or 0
    fn = round(a_count * (1.0 - abn_r)) if abn_r is not None else None
    fp = round(n_count * (1.0 - nor_r)) if nor_r is not None else None

    # VAL_PEAK / NEW_BEST history
    th_path = d / "test_history.json"
    peak_ep = None
    nb_ep = None
    val_peaks = []
    if th_path.exists():
        try:
            th = json.loads(th_path.read_text(encoding="utf-8"))
            for e in th:
                if e.get("event") == "VAL_PEAK":
                    val_peaks.append(e)
                    if peak_ep is None:
                        peak_ep = e["epoch"]
                elif e.get("event") == "NEW_BEST":
                    if nb_ep is None:
                        nb_ep = e["epoch"]
                    if peak_ep is None:
                        peak_ep = e["epoch"]
        except Exception:
            pass

    return {
        "run": d.name,
        "gc": m["gc"],
        "sw": m["sw"],
        "seed": int(m["seed"]),
        "ep": bi.get("epoch"),
        "val_f1": bi.get("val_f1"),
        "test_f1": bi.get("test_f1"),
        "abn_r": abn_r,
        "nor_r": nor_r,
        "fn": fn,
        "fp": fp,
        "first_val_peak_ep": peak_ep,
        "new_best_ep": nb_ep,
        "n_val_peaks": len(val_peaks),
    }


rows = []
for d in sorted(LOGS.iterdir()):
    if not d.is_dir():
        continue
    r = parse_run(d)
    if r:
        rows.append(r)

if not rows:
    print("no vd080_bc runs found")
    raise SystemExit

OUT.parent.mkdir(parents=True, exist_ok=True)
lines = []
lines.append(f"vd080_bc basecase gc × sw sweep  (runs: {len(rows)})")
lines.append("=" * 100)

# Per-(gc, sw)
grouped = defaultdict(list)
for r in rows:
    grouped[(r["gc"], r["sw"])].append(r)

lines.append("")
lines.append(f"{'gc':<6}{'sw':<7}{'n':>3}{'F1_mean':>9}{'F1_std':>8}{'F1_min':>8}{'F1_max':>8}{'FN_μ':>6}{'FP_μ':>6}{'peak_μ':>8}{'nb_μ':>6}")
lines.append("-" * 100)
agg = []
for (gc, sw), runs in sorted(grouped.items()):
    f1s = [r["test_f1"] for r in runs if r["test_f1"] is not None]
    fns = [r["fn"] for r in runs if r["fn"] is not None]
    fps = [r["fp"] for r in runs if r["fp"] is not None]
    peaks = [r["first_val_peak_ep"] for r in runs if r["first_val_peak_ep"] is not None]
    nbs = [r["new_best_ep"] for r in runs if r["new_best_ep"] is not None]
    f1_mean = mean(f1s) if f1s else 0
    f1_std = stdev(f1s) if len(f1s) > 1 else 0
    fn_mean = mean(fns) if fns else 0
    fp_mean = mean(fps) if fps else 0
    peak_mean = mean(peaks) if peaks else 0
    nb_mean = mean(nbs) if nbs else 0
    agg.append({
        "gc": gc, "sw": sw, "n": len(runs),
        "f1_mean": f1_mean, "f1_std": f1_std,
        "f1_min": min(f1s) if f1s else 0, "f1_max": max(f1s) if f1s else 0,
        "fn_mean": fn_mean, "fp_mean": fp_mean,
        "peak_mean": peak_mean, "nb_mean": nb_mean,
    })
    lines.append(
        f"{gc:<6}{sw:<7}{len(runs):>3}"
        f"{f1_mean:>9.4f}{f1_std:>8.4f}"
        f"{(min(f1s) if f1s else 0):>8.4f}{(max(f1s) if f1s else 0):>8.4f}"
        f"{fn_mean:>6.1f}{fp_mean:>6.1f}"
        f"{peak_mean:>8.1f}{nb_mean:>6.1f}"
    )

# Best overall combos
lines.append("")
lines.append("Top 10 (gc, sw) combos by F1_mean (n>=2)")
lines.append("-" * 100)
for a in sorted([x for x in agg if x["n"] >= 2], key=lambda x: -x["f1_mean"])[:10]:
    lines.append(
        f"  gc={a['gc']:<5} sw={a['sw']:<6}  n={a['n']:>2}  "
        f"F1={a['f1_mean']:.4f}±{a['f1_std']:.4f}  "
        f"FN_μ={a['fn_mean']:.1f}  FP_μ={a['fp_mean']:.1f}  "
        f"peak_ep={a['peak_mean']:.1f}  nb_ep={a['nb_mean']:.1f}"
    )

# Individual top runs
lines.append("")
lines.append("Top 15 individual runs")
lines.append("-" * 100)
for r in sorted(rows, key=lambda r: (-r["test_f1"] if r["test_f1"] is not None else 1, r.get("fn") or 999))[:15]:
    lines.append(
        f"  F1={r['test_f1']:.4f} FN={r['fn']:>3} FP={r['fp']:>3} ep={r['ep']:>2} "
        f"peakEp={r['first_val_peak_ep']} nbEp={r['new_best_ep']}  "
        f"gc={r['gc']} sw={r['sw']} s={r['seed']}"
    )

# VAL_PEAK analysis
lines.append("")
lines.append("Peak timing: avg first_val_peak_ep vs new_best_ep")
lines.append("-" * 100)
peak_gap = []
for r in rows:
    if r["first_val_peak_ep"] is not None and r["new_best_ep"] is not None:
        peak_gap.append(r["new_best_ep"] - r["first_val_peak_ep"])
if peak_gap:
    lines.append(f"  mean gap (nb_ep - first_peak_ep) = {mean(peak_gap):.2f}  "
                 f"(n={len(peak_gap)}, min={min(peak_gap)}, max={max(peak_gap)})")
    lines.append(f"  → positive = val_f1 peaks BEFORE best update start")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}  ({len(rows)} runs, {len(grouped)} (gc,sw) combos)")
print()
print("\n".join(lines))
