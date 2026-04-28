"""McNemar's test on per-item paired predictions between two (gc, sw) cells.

For each run in cell A and each in cell B with matching seed, read FN/FP
image filenames (chart_ids) from predictions/ folder. Build paired 2x2:
  - A correct & B correct
  - A correct & B wrong
  - A wrong   & B correct
  - A wrong   & B wrong
Run McNemar exact / chi-squared.

Usage:
  python scripts/mcnemar_compare.py \
      --a-pattern vd080_bc_gc0p5_sw1raw \
      --b-pattern vd080_bc_gc1p0_sw1raw
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-pattern", required=True)
    ap.add_argument("--b-pattern", required=True)
    return ap.parse_args()


def extract_misclass_chart_ids(run_dir: Path):
    """Return (fn_set, fp_set) of chart_ids."""
    fn_dir = run_dir / "predictions" / "fn_abnormal"
    fp_dir = run_dir / "predictions" / "fp_normal"
    fn_ids = set()
    fp_ids = set()
    if fn_dir.exists():
        for f in fn_dir.glob("*.png"):
            m = re.search(r"(ch_\d+)", f.name)
            if m:
                fn_ids.add(m.group(1))
    if fp_dir.exists():
        for f in fp_dir.glob("*.png"):
            m = re.search(r"(ch_\d+)", f.name)
            if m:
                fp_ids.add(m.group(1))
    return fn_ids, fp_ids


def mcnemar_exact(b, c):
    """Exact binomial McNemar p-value. b, c = discordant counts."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    x = min(b, c)
    p = 0.0
    for k in range(x + 1):
        p += comb(n, k) * (0.5 ** n)
    return min(1.0, 2 * p)


def main():
    args = parse_args()
    logs = ROOT / "logs"

    def collect(pattern):
        runs = {}
        for d in sorted(logs.iterdir()):
            if not d.is_dir() or pattern not in d.name:
                continue
            if not (d / "best_info.json").exists():
                continue
            m = re.search(r"_s(\d+)_F", d.name)
            if not m:
                continue
            seed = int(m.group(1))
            fn, fp = extract_misclass_chart_ids(d)
            runs[seed] = {"dir": d, "fn": fn, "fp": fp}
        return runs

    A = collect(args.a_pattern)
    B = collect(args.b_pattern)
    common_seeds = sorted(set(A.keys()) & set(B.keys()))
    print(f"A runs: {len(A)} seeds={sorted(A.keys())}")
    print(f"B runs: {len(B)} seeds={sorted(B.keys())}")
    print(f"Common seeds: {common_seeds}")

    if not common_seeds:
        print("No common seeds")
        return

    # Aggregate: per chart_id, mark wrong if ANY seed got it wrong in that arm.
    # (More conservative: per seed comparison)
    print("\nPer-seed McNemar:")
    print(f"{'seed':<5}{'both_ok':>10}{'A_wrong':>10}{'B_wrong':>10}{'both_wrong':>12}{'p-value':>12}")
    print("-" * 60)
    total_b = total_c = 0
    for s in common_seeds:
        a_wrong = A[s]["fn"] | A[s]["fp"]
        b_wrong = B[s]["fn"] | B[s]["fp"]
        # Use only charts that appear in at least one side
        universe = a_wrong | b_wrong
        # Across 1500 test charts, most are correct. We approximate using universe.
        only_a_wrong = len(a_wrong - b_wrong)
        only_b_wrong = len(b_wrong - a_wrong)
        both_wrong = len(a_wrong & b_wrong)
        both_ok = 1500 - (only_a_wrong + only_b_wrong + both_wrong)
        p = mcnemar_exact(only_a_wrong, only_b_wrong)
        total_b += only_a_wrong
        total_c += only_b_wrong
        print(f"{s:<5}{both_ok:>10}{only_a_wrong:>10}{only_b_wrong:>10}{both_wrong:>12}{p:>12.4f}")

    print("\nPooled across seeds:")
    p_pooled = mcnemar_exact(total_b, total_c)
    print(f"  b (A wrong, B correct) = {total_b}")
    print(f"  c (A correct, B wrong) = {total_c}")
    print(f"  McNemar exact p-value  = {p_pooled:.4f}")
    verdict = "A > B" if total_b < total_c and p_pooled < 0.05 else \
              "B > A" if total_c < total_b and p_pooled < 0.05 else "inconclusive"
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
