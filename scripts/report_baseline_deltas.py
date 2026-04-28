from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RUN_DIR_RE = re.compile(r"^\d{6}_\d{6}_(?P<candidate>.+)_s(?P<seed>\d+)_F[0-9.]+_R[0-9.]+$")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build baseline-relative delta report for completed candidates")
    ap.add_argument("--baseline-candidate", default="fresh0412_v11_n700", help="Actual completed baseline candidate name in logs")
    ap.add_argument("--baseline-alias", default="fresh0412_v11_n700_existing", help="Human-facing baseline alias written into reports")
    ap.add_argument("--candidate-prefix", default="fresh0412_v11_", help="Only include candidates with this prefix")
    ap.add_argument("--out-prefix", required=True, help="Output prefix path, e.g. validations/baseline_delta_latest")
    ap.add_argument("--min-complete", type=int, default=1, help="Minimum completed seeds required to show a candidate")
    return ap.parse_args()


def infer_axis(candidate: str) -> str:
    if "_regls" in candidate:
        return "label_smoothing"
    if "_regdp" in candidate:
        return "stochastic_depth"
    if "_lrwarm" in candidate:
        return "warmup"
    if "_lr" in candidate:
        return "lr"
    if "_gc" in candidate:
        return "gc"
    if "_wd" in candidate:
        return "wd"
    if "_sw" in candidate:
        return "smooth"
    if "_fg" in candidate:
        return "focal_gamma"
    if "_aw" in candidate:
        return "abnormal_weight"
    if "_ema" in candidate:
        return "ema"
    if "_tie_" in candidate:
        return "tie_save"
    if "_color_" in candidate:
        return "rendering"
    if candidate.endswith("_n700") and candidate.count("_") == 3:
        return "baseline"
    if re.search(r"_n\d+$", candidate):
        return "normal_ratio"
    return "other"


def extract_metrics(best_info: dict[str, Any]) -> dict[str, float] | None:
    tm = best_info.get("test_metrics") or {}
    abnormal = tm.get("abnormal") or {}
    normal = tm.get("normal") or {}
    ac = abnormal.get("count")
    ar = abnormal.get("recall")
    nc = normal.get("count")
    nr = normal.get("recall")
    tf1 = best_info.get("test_f1") or best_info.get("f1") or best_info.get("best_f1")
    if None in (ac, ar, nc, nr, tf1):
        return None
    fn = int(ac - round(ar * ac))
    fp = int(nc - round(nr * nc))
    return {"f1": float(tf1), "fn": float(fn), "fp": float(fp)}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def collect_runs(prefix: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(LOGS.iterdir()):
        if not path.is_dir():
            continue
        m = RUN_DIR_RE.match(path.name)
        if not m:
            continue
        candidate = m.group("candidate")
        if not candidate.startswith(prefix):
            continue
        best_path = path / "best_info.json"
        if not best_path.exists():
            continue
        try:
            best = json.loads(best_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metric = extract_metrics(best)
        if metric is None:
            continue
        grouped[candidate].append(
            {
                "seed": int(m.group("seed")),
                "f1": metric["f1"],
                "fn": metric["fn"],
                "fp": metric["fp"],
                "run_dir": str(path),
            }
        )
    for rows in grouped.values():
        rows.sort(key=lambda row: row["seed"])
    return grouped


def build_rows(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    baseline_candidate: str,
    min_complete: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_rows = grouped.get(baseline_candidate)
    if not base_rows:
        raise SystemExit(f"baseline candidate not found: {baseline_candidate}")

    base_by_seed = {row["seed"]: row for row in base_rows}
    baseline = {
        "candidate": baseline_candidate,
        "axis": "baseline",
        "complete": len(base_rows),
        "f1_mean": mean([row["f1"] for row in base_rows]),
        "fn_mean": mean([row["fn"] for row in base_rows]),
        "fp_mean": mean([row["fp"] for row in base_rows]),
        "rows": base_rows,
    }

    rows: list[dict[str, Any]] = []
    for candidate, cand_rows in grouped.items():
        if len(cand_rows) < min_complete:
            continue
        row = {
            "candidate": candidate,
            "axis": infer_axis(candidate),
            "complete": len(cand_rows),
            "f1_mean": mean([r["f1"] for r in cand_rows]),
            "fn_mean": mean([r["fn"] for r in cand_rows]),
            "fp_mean": mean([r["fp"] for r in cand_rows]),
            "rows": cand_rows,
        }
        row["delta_f1_mean"] = row["f1_mean"] - baseline["f1_mean"]
        row["delta_fn_mean"] = row["fn_mean"] - baseline["fn_mean"]
        row["delta_fp_mean"] = row["fp_mean"] - baseline["fp_mean"]

        paired = []
        for cand_row in cand_rows:
            seed = cand_row["seed"]
            if seed not in base_by_seed:
                continue
            base_row = base_by_seed[seed]
            paired.append(
                {
                    "seed": seed,
                    "delta_f1": cand_row["f1"] - base_row["f1"],
                    "delta_fn": cand_row["fn"] - base_row["fn"],
                    "delta_fp": cand_row["fp"] - base_row["fp"],
                    "candidate_f1": cand_row["f1"],
                    "candidate_fn": cand_row["fn"],
                    "candidate_fp": cand_row["fp"],
                    "baseline_f1": base_row["f1"],
                    "baseline_fn": base_row["fn"],
                    "baseline_fp": base_row["fp"],
                }
            )
        row["paired"] = paired
        if paired:
            row["paired_delta_f1_mean"] = mean([p["delta_f1"] for p in paired])
            row["paired_delta_fn_mean"] = mean([p["delta_fn"] for p in paired])
            row["paired_delta_fp_mean"] = mean([p["delta_fp"] for p in paired])
        else:
            row["paired_delta_f1_mean"] = None
            row["paired_delta_fn_mean"] = None
            row["paired_delta_fp_mean"] = None
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["axis"] == "baseline",
            -row["delta_f1_mean"],
            row["delta_fn_mean"],
            row["delta_fp_mean"],
            row["candidate"],
        )
    )
    return baseline, rows


def write_outputs(out_prefix: Path, payload: dict[str, Any], rows: list[dict[str, Any]], baseline_alias: str) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "candidate",
        "axis",
        "complete",
        "f1_mean",
        "fn_mean",
        "fp_mean",
        "delta_f1_mean",
        "delta_fn_mean",
        "delta_fp_mean",
        "paired_delta_f1_mean",
        "paired_delta_fn_mean",
        "paired_delta_fp_mean",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    baseline = payload["baseline"]
    lines = [
        "# Baseline Delta Report",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Baseline alias: `{baseline_alias}`",
        f"- Baseline actual candidate: `{baseline['candidate']}`",
        f"- Baseline mean: `F1={baseline['f1_mean']:.4f}`, `FN={baseline['fn_mean']:.1f}`, `FP={baseline['fp_mean']:.1f}`",
        "",
        "## Candidate Means vs Baseline",
        "",
        "| candidate | axis | seeds | F1 | FN | FP | dF1 | dFN | dFP | paired dF1 | paired dFN | paired dFP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        pdf1 = "-" if row["paired_delta_f1_mean"] is None else f"{row['paired_delta_f1_mean']:+.4f}"
        pdfn = "-" if row["paired_delta_fn_mean"] is None else f"{row['paired_delta_fn_mean']:+.1f}"
        pdfp = "-" if row["paired_delta_fp_mean"] is None else f"{row['paired_delta_fp_mean']:+.1f}"
        lines.append(
            f"| `{row['candidate']}` | `{row['axis']}` | {row['complete']} | {row['f1_mean']:.4f} | {row['fn_mean']:.1f} | {row['fp_mean']:.1f} | {row['delta_f1_mean']:+.4f} | {row['delta_fn_mean']:+.1f} | {row['delta_fp_mean']:+.1f} | {pdf1} | {pdfn} | {pdfp} |"
        )

    focus = [row for row in rows if row["candidate"] in {"fresh0412_v11_regls01_n700", "fresh0412_v11_regdp02_n700"}]
    if focus:
        lines.extend(["", "## Focus Seeds", ""])
        for row in focus:
            lines.append(f"### `{row['candidate']}`")
            lines.append("")
            lines.append("| seed | cand F1 | cand FN | cand FP | base F1 | base FN | base FP | dF1 | dFN | dFP |")
            lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for paired in row["paired"]:
                lines.append(
                    f"| {paired['seed']} | {paired['candidate_f1']:.4f} | {paired['candidate_fn']} | {paired['candidate_fp']} | {paired['baseline_f1']:.4f} | {paired['baseline_fn']} | {paired['baseline_fp']} | {paired['delta_f1']:+.4f} | {paired['delta_fn']:+.0f} | {paired['delta_fp']:+.0f} |"
                )
            lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    grouped = collect_runs(args.candidate_prefix)
    baseline, rows = build_rows(
        grouped,
        baseline_candidate=args.baseline_candidate,
        min_complete=args.min_complete,
    )
    payload = {
        "generated_at": now_iso(),
        "baseline_alias": args.baseline_alias,
        "baseline": baseline,
        "rows": rows,
    }
    write_outputs(Path(args.out_prefix), payload, rows, args.baseline_alias)
    print(
        json.dumps(
            {
                "baseline_candidate": args.baseline_candidate,
                "baseline_f1": baseline["f1_mean"],
                "baseline_fn": baseline["fn_mean"],
                "baseline_fp": baseline["fp_mean"],
                "candidates": len(rows),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
