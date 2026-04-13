import argparse
import csv
import json
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_experiments_v11 as rev11


RESULT_DIR = ROOT / "logs"
TABLE_DIR = RESULT_DIR / "paper_tables"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build paper-style ablation tables and optionally auto-launch combo runs."
    )
    ap.add_argument("--prefix", required=True, help="Experiment prefix, e.g. fresh0413_reset")
    ap.add_argument("--base-n", type=int, default=700, help="Reference normal_ratio")
    ap.add_argument("--python", default=sys.executable, help="Python executable for combo launch")
    ap.add_argument("--num-workers", type=int, default=1, help="train.py num_workers override for combo")
    ap.add_argument("--poll-sec", type=int, default=60, help="Watch polling interval")
    ap.add_argument("--watch", action="store_true", help="Keep polling until combo is done")
    ap.add_argument("--launch-combo", action="store_true", help="Launch combo automatically after ablation completes")
    return ap.parse_args()


def extract_metrics(best_info: dict) -> dict:
    tm = best_info.get("test_metrics", {})
    abn = tm.get("abnormal", {})
    nor = tm.get("normal", {})
    abn_r = float(abn.get("recall", best_info.get("test_abn_recall", 0.0)))
    nor_r = float(nor.get("recall", best_info.get("test_nor_recall", 0.0)))
    f1 = float(best_info.get("test_f1", 0.0))
    abn_sup = int(abn.get("count", abn.get("support", 750)))
    nor_sup = int(nor.get("count", nor.get("support", 750)))
    fn = abn.get("false_negatives")
    fp = nor.get("false_positives")
    if fn is None:
        fn = int(round((1.0 - abn_r) * abn_sup))
    if fp is None:
        fp = int(round((1.0 - nor_r) * nor_sup))
    return {
        "f1": f1,
        "abn_r": abn_r,
        "nor_r": nor_r,
        "fn": int(fn),
        "fp": int(fp),
    }


def scan_results(prefix: str, base_n: int) -> dict:
    data = {
        "baseline": [],
        "lr": {},
        "gc": {},
        "smooth": {},
        "reg": {},
        "combo": {},
    }

    patterns = {
        "baseline": re.compile(rf"{re.escape(prefix)}_v11_n{base_n}_s(?P<seed>\d+)"),
        "lr": re.compile(rf"{re.escape(prefix)}_v11_lr(?P<tag>[^_]+)_n{base_n}_s(?P<seed>\d+)"),
        "gc": re.compile(rf"{re.escape(prefix)}_v11_gc(?P<tag>[^_]+)_n{base_n}_s(?P<seed>\d+)"),
        "smooth": re.compile(rf"{re.escape(prefix)}_v11_sw(?P<tag>[^_]+)_n{base_n}_s(?P<seed>\d+)"),
        "reg": re.compile(rf"{re.escape(prefix)}_v11_reg(?P<tag>[^_]+)_n{base_n}_s(?P<seed>\d+)"),
        "combo": re.compile(
            rf"{re.escape(prefix)}_v11_combo_lr(?P<lr>[^_]+)_gc(?P<gc>[^_]+)_sw(?P<sw>[^_]+)_reg(?P<reg>[^_]+)_n{base_n}_s(?P<seed>\d+)"
        ),
    }

    for d in RESULT_DIR.iterdir():
        bi = d / "best_info.json"
        if not (d.is_dir() and bi.exists()):
            continue
        name = d.name
        best_info = json.loads(bi.read_text(encoding="utf-8"))
        metric = extract_metrics(best_info)

        m = patterns["combo"].search(name)
        if m:
            tag = f"lr={m.group('lr')},gc={m.group('gc')},sw={m.group('sw')},reg={m.group('reg')}"
            data["combo"].setdefault(tag, []).append(
                {"seed": int(m.group("seed")), "name": name, **metric}
            )
            continue

        for group in ("lr", "gc", "smooth", "reg"):
            m = patterns[group].search(name)
            if m:
                data[group].setdefault(m.group("tag"), []).append(
                    {"seed": int(m.group("seed")), "name": name, **metric}
                )
                break
        else:
            m = patterns["baseline"].search(name)
            if m:
                data["baseline"].append({"seed": int(m.group("seed")), "name": name, **metric})

    for group, variants in data.items():
        if isinstance(variants, dict):
            for tag in list(variants.keys()):
                variants[tag] = sorted(variants[tag], key=lambda r: r["seed"])
        else:
            data[group] = sorted(variants, key=lambda r: r["seed"])

    return data


def aggregate(records: list[dict]) -> dict | None:
    if not records:
        return None
    f1s = [r["f1"] for r in records]
    abn_rs = [r["abn_r"] for r in records]
    nor_rs = [r["nor_r"] for r in records]
    fns = [r["fn"] for r in records]
    fps = [r["fp"] for r in records]
    return {
        "count": len(records),
        "f1_mean": st.mean(f1s),
        "f1_std": st.pstdev(f1s) if len(f1s) > 1 else 0.0,
        "abn_r_mean": st.mean(abn_rs),
        "abn_r_std": st.pstdev(abn_rs) if len(abn_rs) > 1 else 0.0,
        "nor_r_mean": st.mean(nor_rs),
        "nor_r_std": st.pstdev(nor_rs) if len(nor_rs) > 1 else 0.0,
        "fn_mean": st.mean(fns),
        "fp_mean": st.mean(fps),
    }


def choose_best_variant(variants: dict[str, list[dict]], expected_count: int) -> tuple[str, dict] | tuple[None, None]:
    candidates = []
    for tag, records in variants.items():
        if len(records) != expected_count:
            continue
        agg = aggregate(records)
        candidates.append((tag, agg))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (-x[1]["f1_mean"], x[1]["fn_mean"], x[1]["fp_mean"], x[0]))
    return candidates[0]


def build_main_rows(prefix: str, base_n: int, scanned: dict) -> tuple[list[dict], dict]:
    baseline = aggregate(scanned["baseline"])
    winners = {}
    for group in ("lr", "gc", "smooth", "reg"):
        tag, agg = choose_best_variant(scanned[group], expected_count=len(rev11.ABLATION_SEEDS))
        if tag is not None:
            winners[group] = {"tag": tag, **agg}

    combo_key = None
    combo_agg = None
    if len(winners) == 4:
        combo_key = (
            f"lr={winners['lr']['tag']},gc={winners['gc']['tag']},"
            f"sw={winners['smooth']['tag']},reg={winners['reg']['tag']}"
        )
        combo_agg = aggregate(scanned["combo"].get(combo_key, []))

    rows = []
    if baseline:
        rows.append({
            "setting": "Ref",
            "variant": f"n={base_n} baseline",
            **baseline,
        })
    if "lr" in winners:
        rows.append({"setting": "+LR", "variant": winners["lr"]["tag"], **winners["lr"]})
    if "gc" in winners:
        rows.append({"setting": "+GC", "variant": winners["gc"]["tag"], **winners["gc"]})
    if "smooth" in winners:
        rows.append({"setting": "+Smooth", "variant": winners["smooth"]["tag"], **winners["smooth"]})
    if "reg" in winners:
        rows.append({"setting": "+Reg", "variant": winners["reg"]["tag"], **winners["reg"]})
    if combo_key and combo_agg:
        rows.append({"setting": "+All", "variant": combo_key, **combo_agg})

    return rows, winners


def write_tables(prefix: str, base_n: int, scanned: dict) -> dict:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows, winners = build_main_rows(prefix, base_n, scanned)

    md_path = TABLE_DIR / f"{prefix}_n{base_n}_paper_table.md"
    csv_path = TABLE_DIR / f"{prefix}_n{base_n}_paper_table.csv"
    json_path = TABLE_DIR / f"{prefix}_n{base_n}_paper_table.json"

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Paper-style Ablation Table\n\n")
        f.write(f"- prefix: `{prefix}`\n")
        f.write(f"- base_n: `{base_n}`\n\n")
        f.write("| Setting | Variant | Seeds | F1 mean ± std | Abn R mean ± std | Nor R mean ± std | FN mean | FP mean |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['setting']} | `{row['variant']}` | {row['count']} | "
                f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f} | "
                f"{row['abn_r_mean']:.4f} ± {row['abn_r_std']:.4f} | "
                f"{row['nor_r_mean']:.4f} ± {row['nor_r_std']:.4f} | "
                f"{row['fn_mean']:.1f} | {row['fp_mean']:.1f} |\n"
            )

        f.write("\n## Group Winners\n\n")
        for group in ("lr", "gc", "smooth", "reg"):
            if group in winners:
                f.write(f"- `{group}`: `{winners[group]['tag']}`\n")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "setting", "variant", "count", "f1_mean", "f1_std",
            "abn_r_mean", "abn_r_std", "nor_r_mean", "nor_r_std", "fn_mean", "fp_mean",
        ]
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    json_path.write_text(
        json.dumps({"rows": rows, "winners": winners}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"rows": rows, "winners": winners, "md_path": md_path, "csv_path": csv_path, "json_path": json_path}


def all_individual_groups_complete(scanned: dict) -> bool:
    return (
        len(scanned["baseline"]) == len(rev11.SWEEP_SEEDS)
        and all(len(scanned["lr"].get(tag, [])) == len(rev11.ABLATION_SEEDS) for tag in rev11.LR_VARIANTS)
        and all(len(scanned["gc"].get(tag, [])) == len(rev11.ABLATION_SEEDS) for tag in rev11.GC_VARIANTS)
        and all(len(scanned["smooth"].get(tag, [])) == len(rev11.ABLATION_SEEDS) for tag in rev11.SMOOTH_VARIANTS)
        and all(len(scanned["reg"].get(tag, [])) == len(rev11.ABLATION_SEEDS) for tag in rev11.REG_VARIANTS)
    )


def combo_marker_path(prefix: str, base_n: int) -> Path:
    return TABLE_DIR / f"{prefix}_n{base_n}_combo_launch.json"


def maybe_launch_combo(args, winners: dict, scanned: dict) -> tuple[bool, str]:
    if len(winners) != 4:
        return False, "best-per-group 미확정"

    combo_key = (
        f"lr={winners['lr']['tag']},gc={winners['gc']['tag']},"
        f"sw={winners['smooth']['tag']},reg={winners['reg']['tag']}"
    )
    if len(scanned["combo"].get(combo_key, [])) == len(rev11.ABLATION_SEEDS):
        return False, "combo already complete"

    marker = combo_marker_path(args.prefix, args.base_n)
    if marker.exists():
        return False, "combo already launched"

    out = RESULT_DIR / f"run_experiments_{args.prefix}_combo_n{args.base_n}.out"
    err = RESULT_DIR / f"run_experiments_{args.prefix}_combo_n{args.base_n}.err"
    cmd = [
        args.python, "-u", "run_experiments_v11.py",
        "--groups", "combo",
        "--base_n", str(args.base_n),
        "--num_workers", str(args.num_workers),
        "--name-prefix", args.prefix,
        "--combo-lr-tag", winners["lr"]["tag"],
        "--combo-gc-tag", winners["gc"]["tag"],
        "--combo-smooth-tag", winners["smooth"]["tag"],
        "--combo-reg-tag", winners["reg"]["tag"],
    ]

    with out.open("w", encoding="utf-8") as so, err.open("w", encoding="utf-8") as se:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=so, stderr=se)

    marker.write_text(
        json.dumps(
            {
                "pid": proc.pid,
                "cmd": cmd,
                "combo_key": combo_key,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return True, combo_key


def print_status(prefix: str, base_n: int, scanned: dict, table_info: dict):
    rows = table_info["rows"]
    print()
    print(f"[paper] prefix={prefix} base_n={base_n}")
    print(f"[paper] baseline={len(scanned['baseline'])}/{len(rev11.SWEEP_SEEDS)}")
    for group, variants in (
        ("lr", rev11.LR_VARIANTS),
        ("gc", rev11.GC_VARIANTS),
        ("smooth", rev11.SMOOTH_VARIANTS),
        ("reg", rev11.REG_VARIANTS),
    ):
        done = sum(1 for tag in variants if len(scanned[group].get(tag, [])) == len(rev11.ABLATION_SEEDS))
        print(f"[paper] {group}: {done}/{len(variants)} variants complete")
    if rows:
        best_row = max(rows, key=lambda r: r["f1_mean"])
        print(f"[paper] best row: {best_row['setting']} ({best_row['variant']}) f1={best_row['f1_mean']:.4f}")
    print(f"[paper] table: {table_info['md_path']}")


def main():
    args = parse_args()
    while True:
        scanned = scan_results(args.prefix, args.base_n)
        table_info = write_tables(args.prefix, args.base_n, scanned)
        print_status(args.prefix, args.base_n, scanned, table_info)

        if args.launch_combo and all_individual_groups_complete(scanned):
            launched, msg = maybe_launch_combo(args, table_info["winners"], scanned)
            print(f"[paper] combo: {msg}")
            if launched:
                print("[paper] combo launched")

        if not args.watch:
            break

        combo_done = False
        if len(table_info["winners"]) == 4:
            combo_key = (
                f"lr={table_info['winners']['lr']['tag']},gc={table_info['winners']['gc']['tag']},"
                f"sw={table_info['winners']['smooth']['tag']},reg={table_info['winners']['reg']['tag']}"
            )
            combo_done = len(scanned["combo"].get(combo_key, [])) == len(rev11.ABLATION_SEEDS)

        if all_individual_groups_complete(scanned) and ((not args.launch_combo) or combo_done):
            break

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
