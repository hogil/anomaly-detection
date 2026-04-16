"""Topic-based comparison of training conditions from 2026-04-08 onwards.

Groups runs by experimental axis (normal count, total n, lr, gc, warmup,
wd, regularization, smoothing) while holding the baseline family fixed,
then reports mean/min/max test_F1, FN, FP across seeds.
"""
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

LOGS = Path("D:/project/anomaly-detection/logs")
OUT = Path("D:/project/anomaly-detection/validations/compare_conditions_0408.txt")
CUTOFF = "260408"


def parse(run_dir: Path):
    info = run_dir / "best_info.json"
    if not info.exists():
        return None
    try:
        d = json.loads(info.read_text(encoding="utf-8"))
    except Exception:
        return None
    tm = d.get("test_metrics") or {}
    n, a = tm.get("normal") or {}, tm.get("abnormal") or {}
    n_count, a_count = n.get("count") or 0, a.get("count") or 0
    n_r, a_r = n.get("recall"), a.get("recall")
    fp = round(n_count * (1.0 - n_r)) if n_r is not None else None
    fn = round(a_count * (1.0 - a_r)) if a_r is not None else None
    return {
        "run": run_dir.name,
        "tag": run_dir.name[14:] if len(run_dir.name) > 14 else run_dir.name,
        "test_f1": d.get("test_f1"),
        "val_f1": d.get("val_f1"),
        "fn": fn,
        "fp": fp,
        "epoch": d.get("epoch"),
    }


rows = []
for p in sorted(LOGS.iterdir()):
    if not p.is_dir() or not p.name[:6].isdigit() or p.name[:6] < CUTOFF:
        continue
    r = parse(p)
    if r and r["test_f1"] is not None:
        rows.append(r)

RE_SEED = re.compile(r"_s(\d+)(?:_F|$)")
RE_N = re.compile(r"_n(\d+)(?:_|$)")
RE_PC = re.compile(r"_pc(\d+)(?:_|$)")
RE_LR = re.compile(r"_lr(\d+e\d+)(?:_|$)")
RE_WARM = re.compile(r"_lrwarm(\d+)(?:_|$)")
RE_GC = re.compile(r"_gc(\d+p?\d*)(?:_|$)")
RE_WD = re.compile(r"_wd(\d+)(?:_|$)")
RE_REG = re.compile(r"_reg([a-z]+\d+)(?:_|$)")
RE_SW = re.compile(r"_sw(\d+[a-z]+)(?:_|$)")


def extract(tag: str):
    seed = RE_SEED.search(tag)
    n = RE_N.search(tag)
    pc = RE_PC.search(tag)
    lr = RE_LR.search(tag)
    warm = RE_WARM.search(tag)
    gc = RE_GC.search(tag)
    wd = RE_WD.search(tag)
    reg = RE_REG.search(tag)
    sw = RE_SW.search(tag)
    return {
        "seed": seed.group(1) if seed else None,
        "n": int(n.group(1)) if n else None,
        "pc": int(pc.group(1)) if pc else None,
        "lr": lr.group(1) if lr else None,
        "warm": int(warm.group(1)) if warm else None,
        "gc": gc.group(1) if gc else None,
        "wd": wd.group(1) if wd else None,
        "reg": reg.group(1) if reg else None,
        "sw": sw.group(1) if sw else None,
    }


for r in rows:
    r.update(extract(r["tag"]))


def family_prefix(tag: str) -> str:
    """Strip seed / score suffix and parameter tokens to get baseline family."""
    t = re.sub(r"_F\d\.\d+_R\d\.\d+$", "", tag)
    t = re.sub(r"_s\d+$", "", t)
    return t


def agg(items, label):
    tf = [x["test_f1"] for x in items]
    fns = [x["fn"] for x in items if x["fn"] is not None]
    fps = [x["fp"] for x in items if x["fp"] is not None]
    vf = [x["val_f1"] for x in items if x["val_f1"] is not None]
    sd_tf = f"{stdev(tf):.4f}" if len(tf) > 1 else "  -   "
    sd_fn = f"{stdev(fns):.2f}" if len(fns) > 1 else "  -  "
    return (
        f"{label:<24}{len(items):>4}"
        f"{mean(tf):>10.4f}{sd_tf:>10}"
        f"{min(tf):>10.4f}{max(tf):>10.4f}"
        f"{(mean(fns) if fns else 0):>8.2f}{sd_fn:>8}"
        f"{(mean(fps) if fps else 0):>8.2f}"
        f"{(mean(vf) if vf else 0):>9.4f}"
    )


def table(title, groups_dict):
    L = [f"\n### {title}"]
    L.append(
        f"{'level':<24}{'n':>4}{'F1_mean':>10}{'F1_std':>10}"
        f"{'F1_min':>10}{'F1_max':>10}{'FN_mean':>8}{'FN_std':>8}{'FP_mean':>8}{'val_F1':>9}"
    )
    L.append("-" * 105)
    # sort by a meaningful key (numeric if possible, else alpha)
    def skey(k):
        if k is None:
            return (1, "")
        try:
            return (0, float(str(k).replace("p", ".").replace("e", "E")))
        except Exception:
            return (0, str(k))
    for k in sorted(groups_dict.keys(), key=skey):
        items = groups_dict[k]
        if not items:
            continue
        L.append(agg(items, str(k)))
    return L


lines = []
lines.append("CONDITION COMPARISON (runs since 2026-04-08)")
lines.append(f"Total runs: {len(rows)}")

# TOPIC 1: normal count sweep (pc100..pc900) within fresh0413_reset_v11
topic = defaultdict(list)
for r in rows:
    if r["tag"].startswith("fresh0413_reset_v11_pc") and r["pc"] is not None:
        topic[r["pc"]].append(r)
lines += table(
    "Topic 1: normal count per class (pc) — baseline fresh0413_reset_v11, n700 implicit",
    topic,
)

# Also fresh0413 reset baseline n700 (no sweep flag) as pc=default reference
ref = [r for r in rows if r["tag"].startswith("fresh0413_reset_v11_n700_s") and r["pc"] is None and r["lr"] is None and r["gc"] is None and r["warm"] is None and r["wd"] is None and r["reg"] is None and r["sw"] is None]
if ref:
    lines.append(agg(ref, "default_n700"))

# TOPIC 2: total dataset size (n) — within fresh0413_reset_v11 (plain) and v11_
topic = defaultdict(list)
for r in rows:
    if r["n"] is None:
        continue
    # fresh0413_reset_v11 plain (no other flag)
    plain = r["tag"].startswith("fresh0413_reset_v11_n") and r["pc"] is None and r["lr"] is None and r["gc"] is None and r["warm"] is None and r["wd"] is None and r["reg"] is None and r["sw"] is None
    if plain:
        topic[r["n"]].append(r)
lines += table(
    "Topic 2a: total N (fresh0413_reset_v11 plain sweep)",
    topic,
)

topic = defaultdict(list)
for r in rows:
    if r["n"] is None:
        continue
    if r["tag"].startswith("v11_n") and "baseline" not in r["tag"]:
        topic[r["n"]].append(r)
lines += table(
    "Topic 2b: total N (v11_ plain, pre-reset)",
    topic,
)

topic = defaultdict(list)
for r in rows:
    if r["n"] is None:
        continue
    if r["tag"].startswith("fresh0412_v11_n") and r["lr"] is None and r["gc"] is None and r["warm"] is None and r["sw"] is None:
        topic[r["n"]].append(r)
lines += table(
    "Topic 2c: total N (fresh0412_v11 plain)",
    topic,
)

# TOPIC 3: learning rate — fresh0413_reset_v11 (n700) and fresh0412_v11 (n2800)
topic = defaultdict(list)
for r in rows:
    if r["lr"] is None:
        continue
    if r["tag"].startswith("fresh0413_reset_v11_lr") and not r["tag"].startswith("fresh0413_reset_v11_lrwarm") and not r["tag"].startswith("fresh0413_reset_v11_rescue"):
        topic[r["lr"]].append(r)
lines += table(
    "Topic 3a: learning rate (fresh0413_reset_v11, n700)",
    topic,
)

topic = defaultdict(list)
for r in rows:
    if r["lr"] is None:
        continue
    if r["tag"].startswith("fresh0412_v11_lr") and not r["tag"].startswith("fresh0412_v11_lrwarm"):
        topic[r["lr"]].append(r)
lines += table(
    "Topic 3b: learning rate (fresh0412_v11, n2800)",
    topic,
)

# TOPIC 4: warmup epochs
topic = defaultdict(list)
for r in rows:
    if r["warm"] is None:
        continue
    if r["tag"].startswith("fresh0413_reset_v11_lrwarm"):
        topic[r["warm"]].append(r)
lines += table("Topic 4a: warmup epochs (fresh0413_reset_v11, n700)", topic)

topic = defaultdict(list)
for r in rows:
    if r["warm"] is None:
        continue
    if r["tag"].startswith("fresh0412_v11_lrwarm"):
        topic[r["warm"]].append(r)
lines += table("Topic 4b: warmup epochs (fresh0412_v11, n2800)", topic)

# TOPIC 5: gradient clip
topic = defaultdict(list)
for r in rows:
    if r["gc"] is None:
        continue
    if r["tag"].startswith("fresh0413_reset_v11_gc"):
        topic[r["gc"]].append(r)
lines += table("Topic 5a: gradient clip (fresh0413_reset_v11, n700)", topic)

topic = defaultdict(list)
for r in rows:
    if r["gc"] is None:
        continue
    if r["tag"].startswith("fresh0412_v11_gc"):
        topic[r["gc"]].append(r)
lines += table("Topic 5b: gradient clip (fresh0412_v11, n2800)", topic)

topic = defaultdict(list)
for r in rows:
    if r["gc"] is None:
        continue
    if r["tag"].startswith("v9mid6_win_gc"):
        topic[r["gc"]].append(r)
lines += table("Topic 5c: gradient clip (v9mid6_win, n700)", topic)

# TOPIC 6: weight decay
topic = defaultdict(list)
for r in rows:
    if r["wd"] is None:
        continue
    if r["tag"].startswith("fresh0413_reset_v11_wd"):
        topic[r["wd"]].append(r)
lines += table("Topic 6: weight decay × 1e-3 (fresh0413_reset_v11, n700)", topic)

# TOPIC 7: regularization
topic = defaultdict(list)
for r in rows:
    if r["reg"] is None:
        continue
    if r["tag"].startswith("fresh0413_reset_v11_reg"):
        topic[r["reg"]].append(r)
lines += table("Topic 7: regularization (fresh0413_reset_v11, n700)", topic)

# TOPIC 8: val smoothing
topic = defaultdict(list)
for r in rows:
    if r["sw"] is None:
        continue
    if r["tag"].startswith("fresh0413_reset_v11_sw"):
        topic[r["sw"]].append(r)
lines += table("Topic 8: val smoothing window (fresh0413_reset_v11, n700)", topic)

# Cross-topic: impact summary (delta F1 from best-seed baseline)
lines.append("")
lines.append("=" * 105)
lines.append("IMPACT RANKING — each topic's spread of mean test_F1 across its levels")
lines.append("=" * 105)
impact = []

def spread(d):
    means = [mean([x["test_f1"] for x in v]) for v in d.values() if v]
    if len(means) < 2:
        return None
    return (max(means) - min(means), min(means), max(means))


topics_all = [
    ("pc (normal count)", lambda r: r["tag"].startswith("fresh0413_reset_v11_pc"), "pc"),
    ("n (total size, fresh0413)", lambda r: r["tag"].startswith("fresh0413_reset_v11_n") and r["pc"] is None and r["lr"] is None and r["gc"] is None and r["warm"] is None and r["wd"] is None and r["reg"] is None and r["sw"] is None, "n"),
    ("n (total size, fresh0412)", lambda r: r["tag"].startswith("fresh0412_v11_n") and r["lr"] is None and r["gc"] is None and r["warm"] is None and r["sw"] is None, "n"),
    ("n (total size, v11_ plain)", lambda r: r["tag"].startswith("v11_n"), "n"),
    ("lr (fresh0413)", lambda r: r["tag"].startswith("fresh0413_reset_v11_lr") and not r["tag"].startswith("fresh0413_reset_v11_lrwarm") and not r["tag"].startswith("fresh0413_reset_v11_rescue"), "lr"),
    ("lr (fresh0412)", lambda r: r["tag"].startswith("fresh0412_v11_lr") and not r["tag"].startswith("fresh0412_v11_lrwarm"), "lr"),
    ("warmup (fresh0413)", lambda r: r["tag"].startswith("fresh0413_reset_v11_lrwarm"), "warm"),
    ("warmup (fresh0412)", lambda r: r["tag"].startswith("fresh0412_v11_lrwarm"), "warm"),
    ("gc (fresh0412)", lambda r: r["tag"].startswith("fresh0412_v11_gc"), "gc"),
    ("wd (fresh0413)", lambda r: r["tag"].startswith("fresh0413_reset_v11_wd"), "wd"),
    ("reg (fresh0413)", lambda r: r["tag"].startswith("fresh0413_reset_v11_reg"), "reg"),
    ("sw (fresh0413)", lambda r: r["tag"].startswith("fresh0413_reset_v11_sw"), "sw"),
]

lines.append(f"{'axis':<30}{'levels':>8}{'delta_F1':>10}{'F1_min':>10}{'F1_max':>10}")
lines.append("-" * 75)
ranking = []
for name, filt, key in topics_all:
    d = defaultdict(list)
    for r in rows:
        if filt(r) and r.get(key) is not None:
            d[r[key]].append(r)
    s = spread(d)
    if s is None:
        continue
    ranking.append((name, len(d), *s))
ranking.sort(key=lambda x: -x[2])
for name, lv, delta, lo, hi in ranking:
    lines.append(f"{name:<30}{lv:>8}{delta:>10.4f}{lo:>10.4f}{hi:>10.4f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
import sys
sys.stdout.reconfigure(encoding="utf-8")
print("\n".join(lines))
print(f"\nSaved to {OUT}")
