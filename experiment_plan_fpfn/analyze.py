"""
Phase 별 결과 분석 + 다음 phase 권장.

Usage:
    python experiment_plan_fpfn/analyze.py --phase 1
    python experiment_plan_fpfn/analyze.py --phase all
"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"


def load_run(log_dir: Path):
    info_f = log_dir / "best_info.json"
    if not info_f.exists():
        return None
    try:
        bi = json.loads(info_f.read_text(encoding="utf-8"))
        tm = bi.get("test_metrics", {})
        hp = bi.get("hparams", {})
        th = bi.get("test_history", [])
        # FN/FP from last test_history entry
        fn = fp = None
        if th:
            last = th[-1]
            fn = last.get("fn")
            fp = last.get("fp")
        return {
            "run": log_dir.name,
            "best_ep": bi.get("epoch"),
            "test_f1": bi.get("test_f1"),
            "abn_R": tm.get("abnormal", {}).get("recall"),
            "nor_R": tm.get("normal", {}).get("recall"),
            "fn": fn,
            "fp": fp,
            "lr_bb": hp.get("lr_backbone"),
            "warmup": hp.get("warmup_epochs"),
            "patience": hp.get("patience"),
            "epochs": hp.get("epochs"),
            "ema_decay": hp.get("ema_decay"),
            "seed": hp.get("seed"),
        }
    except Exception as e:
        print(f"  WARN load {log_dir.name}: {e}")
        return None


def phase_runs(phase: int):
    pattern = f"v9_phase{phase}_*"
    return sorted([d for d in LOGS.glob(pattern) if d.is_dir()])


def print_phase(phase: int):
    dirs = phase_runs(phase)
    if not dirs:
        print(f"\n[Phase {phase}] 완료된 run 없음.")
        return []

    runs = [r for r in (load_run(d) for d in dirs) if r]
    if not runs:
        print(f"\n[Phase {phase}] best_info.json 있는 run 없음.")
        return []

    print(f"\n{'='*95}")
    print(f" Phase {phase} — {len(runs)} runs")
    print(f"{'='*95}")
    print(f"  {'run':<45} {'best_ep':>7} {'f1%':>7} {'abn%':>7} {'nor%':>7} {'FN':>4} {'FP':>4}")
    print("  " + "-" * 90)

    for r in sorted(runs, key=lambda r: -r.get("test_f1", 0) if r.get("test_f1") else 0):
        f1 = r.get("test_f1") or 0
        abn = r.get("abn_R") or 0
        nor = r.get("nor_R") or 0
        fn = r.get("fn") if r.get("fn") is not None else "—"
        fp = r.get("fp") if r.get("fp") is not None else "—"

        target = "⭐" if (isinstance(fn, int) and isinstance(fp, int) and fn <= 2 and fp <= 2) else "  "
        print(f"{target}{r['run']:<45} {r.get('best_ep','—'):>7} {f1*100:>6.2f} {abn*100:>6.2f} {nor*100:>6.2f} {str(fn):>4} {str(fp):>4}")

    # 통계
    f1s = [r["test_f1"] for r in runs if r.get("test_f1") is not None]
    if f1s:
        print(f"\n  F1 mean ± std: {np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}")
        print(f"  F1 best: {max(f1s)*100:.2f}")
    target_count = sum(1 for r in runs if (r.get("fn") or 999) <= 2 and (r.get("fp") or 999) <= 2)
    print(f"  🎯 target 달성 (FN≤2 AND FP≤2): {target_count} / {len(runs)}")

    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", help="1/2/3/4/all")
    args = ap.parse_args()

    phases = [1, 2, 3, 4] if args.phase == "all" else [int(args.phase)]

    all_runs = []
    for p in phases:
        rs = print_phase(p)
        all_runs.extend(rs)

    if len(phases) > 1 and all_runs:
        print(f"\n{'='*95}")
        print(f" OVERALL — {len(all_runs)} runs")
        print(f"{'='*95}")
        top = sorted(all_runs, key=lambda r: -r.get("test_f1", 0) if r.get("test_f1") else 0)[:10]
        print(f"\n  Top 10 by F1:")
        print(f"  {'run':<50} {'f1%':>7} {'abn%':>7} {'nor%':>7} {'FN':>4} {'FP':>4}")
        for r in top:
            f1 = r.get("test_f1") or 0
            print(f"  {r['run']:<50} {f1*100:>6.2f} {(r.get('abn_R') or 0)*100:>6.2f} {(r.get('nor_R') or 0)*100:>6.2f} {str(r.get('fn','—')):>4} {str(r.get('fp','—')):>4}")


if __name__ == "__main__":
    main()
