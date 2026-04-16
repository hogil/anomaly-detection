"""Aggregate val/test metrics from logs/ since 2026-04-08.

Writes full table to validations/results_since_0408.txt and prints a
group-by-prefix summary with best/worst/mean for each experiment family.
"""
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

LOGS = Path("D:/project/anomaly-detection/logs")
OUT = Path("D:/project/anomaly-detection/validations/results_since_0408.txt")
CUTOFF = "260408"


def parse(run_dir: Path):
    info = run_dir / "best_info.json"
    if not info.exists():
        return None
    try:
        data = json.loads(info.read_text(encoding="utf-8"))
    except Exception:
        return None

    tm = data.get("test_metrics") or {}
    normal = tm.get("normal") or {}
    abnormal = tm.get("abnormal") or {}
    n_count = normal.get("count") or 0
    a_count = abnormal.get("count") or 0
    n_recall = normal.get("recall")
    a_recall = abnormal.get("recall")
    fp = round(n_count * (1.0 - n_recall)) if n_recall is not None else None
    fn = round(a_count * (1.0 - a_recall)) if a_recall is not None else None

    return {
        "run": run_dir.name,
        "date": run_dir.name[:6],
        "time": run_dir.name[7:13],
        "epoch": data.get("epoch"),
        "val_f1": data.get("val_f1"),
        "val_recall": data.get("val_recall"),
        "test_f1": data.get("test_f1"),
        "test_recall": data.get("test_recall"),
        "abn_recall": a_recall,
        "nor_recall": n_recall,
        "fp": fp,
        "fn": fn,
    }


def group_key(name: str) -> str:
    """Extract experiment family. Strip date + trailing _sNN + _F/R scores."""
    tag = name[14:] if len(name) > 14 and name[:13].replace("_", "").isdigit() else name
    tag = re.sub(r"_F\d\.\d+_R\d\.\d+$", "", tag)
    tag = re.sub(r"_s\d+$", "", tag)
    return tag


def fmt(v, w=6, prec=4):
    if v is None:
        return "-".rjust(w)
    return f"{v:.{prec}f}".rjust(w)


rows = []
for p in sorted(LOGS.iterdir()):
    if not p.is_dir():
        continue
    if not (len(p.name) >= 6 and p.name[:6].isdigit()):
        continue
    if p.name[:6] < CUTOFF:
        continue
    r = parse(p)
    if r:
        rows.append(r)

rows.sort(key=lambda r: r["run"])

OUT.parent.mkdir(parents=True, exist_ok=True)
lines = []
lines.append(f"Total runs since {CUTOFF}: {len(rows)}")
lines.append("")
hdr = f"{'date':<7}{'time':<8}{'val_f1':>8}{'val_R':>8}{'test_f1':>8}{'test_R':>8}{'abn_R':>8}{'nor_R':>8}{'FP':>5}{'FN':>5}{'ep':>4}  run"
lines.append(hdr)
lines.append("-" * 110)
for r in rows:
    lines.append(
        f"{r['date']:<7}{r['time']:<8}"
        f"{fmt(r['val_f1'],8)}{fmt(r['val_recall'],8)}"
        f"{fmt(r['test_f1'],8)}{fmt(r['test_recall'],8)}"
        f"{fmt(r['abn_recall'],8)}{fmt(r['nor_recall'],8)}"
        f"{str(r['fp']) if r['fp'] is not None else '-':>5}"
        f"{str(r['fn']) if r['fn'] is not None else '-':>5}"
        f"{str(r['epoch']) if r['epoch'] is not None else '-':>4}"
        f"  {r['run'][14:] if len(r['run']) > 14 else r['run']}"
    )

# Group summary
groups = defaultdict(list)
for r in rows:
    if r["test_f1"] is None:
        continue
    groups[group_key(r["run"])].append(r)

lines.append("")
lines.append("=" * 110)
lines.append(f"Group summary (by experiment family, {len(groups)} families)")
lines.append("=" * 110)
lines.append(
    f"{'family':<40}{'n':>4}{'testF1_mean':>13}{'testF1_med':>12}"
    f"{'testF1_min':>12}{'testF1_max':>12}{'FN_mean':>9}{'FP_mean':>9}"
)
lines.append("-" * 110)
grp_rows = []
for k, lst in groups.items():
    tf = [r["test_f1"] for r in lst]
    fns = [r["fn"] for r in lst if r["fn"] is not None]
    fps = [r["fp"] for r in lst if r["fp"] is not None]
    grp_rows.append({
        "family": k,
        "n": len(lst),
        "mean": mean(tf),
        "median": median(tf),
        "min": min(tf),
        "max": max(tf),
        "fn_mean": mean(fns) if fns else None,
        "fp_mean": mean(fps) if fps else None,
    })
grp_rows.sort(key=lambda g: (-g["mean"], g["family"]))
for g in grp_rows:
    lines.append(
        f"{g['family'][:39]:<40}{g['n']:>4}"
        f"{g['mean']:>13.4f}{g['median']:>12.4f}"
        f"{g['min']:>12.4f}{g['max']:>12.4f}"
        f"{(g['fn_mean'] if g['fn_mean'] is not None else 0):>9.2f}"
        f"{(g['fp_mean'] if g['fp_mean'] is not None else 0):>9.2f}"
    )

# Top 15 runs
lines.append("")
lines.append("=" * 110)
lines.append("Top 15 by test_f1 (ties broken by lower FN)")
lines.append("=" * 110)
ranked = [r for r in rows if r["test_f1"] is not None]
ranked.sort(key=lambda r: (-r["test_f1"], r["fn"] if r["fn"] is not None else 9999))
for r in ranked[:15]:
    lines.append(
        f"  F1={r['test_f1']:.4f} R={r['test_recall']:.4f} "
        f"FP={r['fp']:>3} FN={r['fn']:>3} ep={r['epoch']:>2}  {r['run']}"
    )

# Worst 10
lines.append("")
lines.append("Worst 10 by test_f1")
lines.append("-" * 110)
for r in ranked[-10:]:
    lines.append(
        f"  F1={r['test_f1']:.4f} R={r['test_recall']:.4f} "
        f"FP={r['fp']:>3} FN={r['fn']:>3} ep={r['epoch']:>2}  {r['run']}"
    )

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}  ({len(rows)} runs, {len(groups)} families)")
print()
print("\n".join(lines[-(len(grp_rows) + 5 + 17 + 12):]))
