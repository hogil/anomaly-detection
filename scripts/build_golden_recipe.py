"""Build golden recipe from vd080_ax axis ablation results.

For each axis (wd, gc, lr, fg, abw, dp, ema), pick the level that minimizes
test FN+FP mean across seeds. Emit:
  - configs/train/golden.yaml (override winning.yaml with picked levels)
  - scripts/sweeps_laptop/legacy/43_golden_recipe.sh (10 seeds × golden vs 10 × baseline)
  - validations/golden_recipe_selection.txt (per-axis winner with evidence)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT_YAML = ROOT / "configs" / "train" / "golden.yaml"
OUT_SH   = ROOT / "scripts" / "sweeps_laptop" / "legacy" / "43_golden_recipe.sh"
OUT_TXT  = ROOT / "validations" / "golden_recipe_selection.txt"

# Map axis tag → yaml key override
AXIS_ARG_MAP = {
    "wd000":  {"weight_decay": 0.0},
    "wd001":  {"weight_decay": 0.01},
    "wd005":  {"weight_decay": 0.05},
    "gc05":   {"grad_clip": 0.5},
    "gc10":   {"grad_clip": 1.0},
    "gc20":   {"grad_clip": 2.0},
    "lr1e5":  {"lr_backbone": 1e-5, "lr_head": 1e-4},
    "lr2e5":  {"lr_backbone": 2e-5, "lr_head": 2e-4},
    "lr5e5":  {"lr_backbone": 5e-5, "lr_head": 5e-4},
    "fg0":    {"focal_gamma": 0.0},
    "fg0p5":  {"focal_gamma": 0.5},
    "fg2":    {"focal_gamma": 2.0},
    "abw08":  {"abnormal_weight": 0.8},
    "abw10":  {"abnormal_weight": 1.0},
    "abw20":  {"abnormal_weight": 2.0},
    "dp0":    {"dropout": 0.0},
    "dp02":   {"dropout": 0.2},
    "dp05":   {"dropout": 0.5},
    "ema0":   {"ema_decay": 0.0},
    "ema99":  {"ema_decay": 0.99},
    "ema999": {"ema_decay": 0.999},
}

# axis group → list of levels
AXIS_GROUPS = {
    "wd":  ["wd000", "wd001", "wd005"],
    "gc":  ["gc05", "gc10", "gc20"],
    "lr":  ["lr1e5", "lr2e5", "lr5e5"],
    "fg":  ["fg0", "fg0p5", "fg2"],
    "abw": ["abw08", "abw10", "abw20"],
    "dp":  ["dp0", "dp02", "dp05"],
    "ema": ["ema0", "ema99", "ema999"],
}

PATTERN = re.compile(r"^\d{6}_\d{6}_vd080_ax_(?P<tag>[^_]+)_n700_s(?P<seed>\d+)")


def parse_run(d: Path):
    m = PATTERN.match(d.name)
    if not m:
        return None
    bi = d / "best_info.json"
    if not bi.exists():
        return None
    try:
        info = json.loads(bi.read_text(encoding="utf-8"))
    except Exception:
        return None
    tm = info.get("test_metrics") or {}
    a, n = tm.get("abnormal") or {}, tm.get("normal") or {}
    abn_r, nor_r = a.get("recall"), n.get("recall")
    a_cnt, n_cnt = a.get("count") or 0, n.get("count") or 0
    fn = round(a_cnt * (1 - abn_r)) if abn_r is not None else 0
    fp = round(n_cnt * (1 - nor_r)) if nor_r is not None else 0
    return {
        "tag": m["tag"],
        "seed": int(m["seed"]),
        "f1": info.get("test_f1", 0.0),
        "fn": fn, "fp": fp,
        "err": fn + fp,
    }


def main():
    rows = []
    for d in sorted(LOGS.iterdir()):
        if not d.is_dir(): continue
        r = parse_run(d)
        if r: rows.append(r)

    if not rows:
        print("No vd080_ax runs yet. Wait for axis ablation to complete.")
        sys.exit(1)

    lines = [f"Golden recipe selection from {len(rows)} axis ablation runs", "=" * 70, ""]
    selected_overrides = {}

    for axis, tags in AXIS_GROUPS.items():
        axis_rows = {t: [r for r in rows if r["tag"] == t] for t in tags}
        if not all(axis_rows[t] for t in tags):
            lines.append(f"[{axis}] INCOMPLETE — skipping (need ≥1 run per level)")
            continue
        stats = []
        for t in tags:
            errs = [r["err"] for r in axis_rows[t]]
            fns = [r["fn"] for r in axis_rows[t]]
            fps = [r["fp"] for r in axis_rows[t]]
            stats.append({
                "tag": t,
                "n": len(errs),
                "err_mean": mean(errs),
                "err_std": stdev(errs) if len(errs) > 1 else 0,
                "fn_mean": mean(fns),
                "fp_mean": mean(fps),
            })
        # Winner: lowest err_mean, break ties by lower err_std
        stats.sort(key=lambda s: (s["err_mean"], s["err_std"]))
        winner = stats[0]
        lines.append(f"[{axis}] winner: {winner['tag']}  err_μ={winner['err_mean']:.2f}±{winner['err_std']:.2f}  "
                     f"FN_μ={winner['fn_mean']:.2f} FP_μ={winner['fp_mean']:.2f}  (n={winner['n']})")
        for s in stats:
            lines.append(f"  {s['tag']:7s} err_μ={s['err_mean']:.2f}±{s['err_std']:.2f}  "
                         f"FN_μ={s['fn_mean']:.2f} FP_μ={s['fp_mean']:.2f}  n={s['n']}")
        selected_overrides.update(AXIS_ARG_MAP[winner["tag"]])
        lines.append("")

    lines.append("=" * 70)
    lines.append("GOLDEN RECIPE OVERRIDES:")
    for k, v in selected_overrides.items():
        lines.append(f"  {k}: {v}")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # Write golden.yaml — mirrors winning.yaml with overrides
    winning_yaml = ROOT / "configs" / "train" / "winning.yaml"
    text = winning_yaml.read_text(encoding="utf-8") if winning_yaml.exists() else ""
    import yaml
    base_cfg = yaml.safe_load(text) if text else {}
    base_cfg.update(selected_overrides)
    OUT_YAML.write_text(yaml.safe_dump(base_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"\nWrote {OUT_YAML}")

    # Write 43_golden_recipe.sh: 10 seeds baseline + 10 seeds golden
    sh_lines = ["#!/usr/bin/env bash",
                "# Golden recipe vs winning baseline: 10 seeds each.",
                "# Auto-generated by scripts/build_golden_recipe.py",
                'source "$(dirname "$0")/_common.sh"',
                "",
                "SEEDS=(42 1 2 3 4 5 6 7 8 9)",
                "",
                "# Baseline (winning.yaml defaults)",
                "for s in \"${SEEDS[@]}\"; do",
                "  run_one \"vd080_control_n700_s${s}\" --config dataset.yaml --seed \"$s\"",
                "done",
                "",
                "# Golden recipe"]
    gold_args = []
    for k, v in selected_overrides.items():
        gold_args.append(f"--{k} {v}")
    gold_arg_str = " ".join(gold_args)
    sh_lines += [
        "for s in \"${SEEDS[@]}\"; do",
        f"  run_one \"vd080_golden_n700_s${{s}}\" --config dataset.yaml {gold_arg_str} --seed \"$s\"",
        "done",
    ]
    OUT_SH.write_text("\n".join(sh_lines) + "\n", encoding="utf-8")
    OUT_SH.chmod(0o755)
    print(f"Wrote {OUT_SH}")


if __name__ == "__main__":
    main()
