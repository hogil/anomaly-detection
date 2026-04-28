from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RUN_DIR_RE = re.compile(r"^\d{6}_\d{6}_(?P<candidate>.+)_s(?P<seed>\d+)_F[0-9.]+_R[0-9.]+$")
CHART_RE = re.compile(r"(ch_\d+)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze per-sample FP/FN trends across runs")
    ap.add_argument("--config", required=True, help="Dataset config used by the active experiment family")
    ap.add_argument("--candidate-prefix", default="fresh0412_v11_", help="Only include candidates with this prefix")
    ap.add_argument("--min-f1", type=float, default=0.99, help="Minimum test_f1 for a run to count as strong")
    ap.add_argument("--out-prefix", required=True, help="Output prefix path, e.g. validations/prediction_trend_latest")
    ap.add_argument("--top-k", type=int, default=40, help="Top rows to keep per category in markdown")
    ap.add_argument("--review-k", type=int, default=15, help="How many images to copy per category for manual review")
    ap.add_argument("--report-label", default="", help="Optional label written into the markdown report")
    return ap.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def infer_axis(candidate: str) -> str:
    if "_stressw3_gc" in candidate:
        return "gc_stress"
    if "_stresslr5e5_wd" in candidate:
        return "wd_stress"
    if "_stressw3_sw" in candidate:
        return "smooth_stress"
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
        return "color"
    m = re.search(r"_n(\d+)$", candidate)
    if m:
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
    return {"test_f1": float(tf1), "fn": float(fn), "fp": float(fp)}


def extract_chart_ids(pred_dir: Path, subdir: str) -> set[str]:
    out: set[str] = set()
    folder = pred_dir / subdir
    if not folder.exists():
        return out
    for path in folder.glob("*.png"):
        m = CHART_RE.search(path.name)
        if m:
            out.add(m.group(1))
    return out


@dataclass
class RunSummary:
    candidate: str
    seed: int
    axis: str
    run_dir: Path
    test_f1: float
    fn: int
    fp: int
    strong: bool
    fn_ids: set[str]
    fp_ids: set[str]


def collect_runs(prefix: str, min_f1: float) -> list[RunSummary]:
    runs: list[RunSummary] = []
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
        pred_dir = path / "predictions"
        if not best_path.exists() or not pred_dir.exists():
            continue
        try:
            best = json.loads(best_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metric = extract_metrics(best)
        if metric is None:
            continue
        runs.append(
            RunSummary(
                candidate=candidate,
                seed=int(m.group("seed")),
                axis=infer_axis(candidate),
                run_dir=path,
                test_f1=float(metric["test_f1"]),
                fn=int(metric["fn"]),
                fp=int(metric["fp"]),
                strong=float(metric["test_f1"]) >= min_f1,
                fn_ids=extract_chart_ids(pred_dir, "fn_abnormal"),
                fp_ids=extract_chart_ids(pred_dir, "fp_normal"),
            )
        )
    return runs


def load_scenarios(config_path: Path) -> tuple[pd.DataFrame, Path, Path]:
    cfg = load_yaml(config_path)
    output = cfg.get("output") or {}
    data_dir = ROOT / str(output.get("data_dir", "data"))
    display_dir = ROOT / str(output.get("display_dir", "display"))
    image_dir = ROOT / str(output.get("image_dir", "images"))
    scenarios = pd.read_csv(data_dir / "scenarios.csv")
    scenarios = scenarios[scenarios["split"] == "test"].copy()
    return scenarios, display_dir, image_dir


def classify_row(row: dict[str, Any]) -> str:
    strong_total = row["strong_total"]
    strong_wrong = row["strong_wrong"]
    strong_rate = row["strong_wrong_rate"]
    axis_div = row["strong_axis_diversity"]
    candidate_div = row["strong_candidate_diversity"]
    ever_correct = row["strong_total"] - row["strong_wrong"]
    total_wrong = row["all_wrong"]

    if strong_total >= 5 and strong_rate >= 0.8 and axis_div >= 3 and candidate_div >= 5:
        return "label_or_annotation_suspect"
    if strong_total >= 4 and strong_rate >= 0.5 and axis_div >= 2:
        return "persistent_hard_sample"
    if strong_wrong >= 2 and ever_correct >= 2:
        return "config_sensitive"
    if total_wrong > 0 and strong_wrong == 0:
        return "mostly_unstable_only"
    if total_wrong > 0:
        return "sporadic"
    return "clean"


def build_rows(runs: list[RunSummary], scenarios: pd.DataFrame) -> list[dict[str, Any]]:
    meta_by_chart = scenarios.set_index("chart_id").to_dict(orient="index")
    per_chart: dict[str, dict[str, Any]] = {}

    def init_chart(chart_id: str) -> dict[str, Any]:
        meta = meta_by_chart[chart_id]
        true_cls = str(meta["class"])
        role = "fp" if true_cls == "normal" else "fn"
        return {
            "chart_id": chart_id,
            "true_class": true_cls,
            "error_role": role,
            "device": meta.get("device", ""),
            "step": meta.get("step", ""),
            "item": meta.get("item", ""),
            "context_column": meta.get("context_column", ""),
            "target": meta.get("target", ""),
            "defect_start_idx": meta.get("defect_start_idx", ""),
            "target_value": meta.get("target_value", ""),
            "all_total": 0,
            "strong_total": 0,
            "all_wrong": 0,
            "strong_wrong": 0,
            "all_candidates": set(),
            "all_axes": set(),
            "strong_candidates": set(),
            "strong_axes": set(),
            "wrong_candidates": set(),
            "wrong_axes": set(),
            "strong_wrong_candidates": set(),
            "strong_wrong_axes": set(),
            "wrong_seeds": set(),
            "strong_wrong_seeds": set(),
            "wrong_runs": [],
            "strong_wrong_runs": [],
        }

    test_chart_ids = set(meta_by_chart.keys())

    for run in runs:
        for chart_id in test_chart_ids:
            row = per_chart.setdefault(chart_id, init_chart(chart_id))
            row["all_total"] += 1
            row["all_candidates"].add(run.candidate)
            row["all_axes"].add(run.axis)
            if run.strong:
                row["strong_total"] += 1
                row["strong_candidates"].add(run.candidate)
                row["strong_axes"].add(run.axis)

        wrong_ids = run.fn_ids | run.fp_ids
        for chart_id in wrong_ids:
            if chart_id not in test_chart_ids:
                continue
            row = per_chart.setdefault(chart_id, init_chart(chart_id))
            row["all_wrong"] += 1
            row["wrong_candidates"].add(run.candidate)
            row["wrong_axes"].add(run.axis)
            row["wrong_seeds"].add(run.seed)
            row["wrong_runs"].append(
                {
                    "candidate": run.candidate,
                    "axis": run.axis,
                    "seed": run.seed,
                    "test_f1": run.test_f1,
                    "run_dir": str(run.run_dir),
                }
            )
            if run.strong:
                row["strong_wrong"] += 1
                row["strong_wrong_candidates"].add(run.candidate)
                row["strong_wrong_axes"].add(run.axis)
                row["strong_wrong_seeds"].add(run.seed)
                row["strong_wrong_runs"].append(
                    {
                        "candidate": run.candidate,
                        "axis": run.axis,
                        "seed": run.seed,
                        "test_f1": run.test_f1,
                        "run_dir": str(run.run_dir),
                    }
                )

    rows: list[dict[str, Any]] = []
    for chart_id, row in per_chart.items():
        strong_total = row["strong_total"]
        strong_wrong = row["strong_wrong"]
        strong_rate = (strong_wrong / strong_total) if strong_total > 0 else 0.0
        sample = {
            k: v
            for k, v in row.items()
            if k not in {
                "all_candidates",
                "all_axes",
                "strong_candidates",
                "strong_axes",
                "wrong_candidates",
                "wrong_axes",
                "strong_wrong_candidates",
                "strong_wrong_axes",
                "wrong_seeds",
                "strong_wrong_seeds",
            }
        }
        sample.update(
            {
                "all_candidate_diversity": len(row["all_candidates"]),
                "all_axis_diversity": len(row["all_axes"]),
                "strong_wrong_rate": strong_rate,
                "strong_candidate_diversity": len(row["strong_candidates"]),
                "strong_axis_diversity": len(row["strong_axes"]),
                "wrong_candidate_diversity": len(row["wrong_candidates"]),
                "wrong_axis_diversity": len(row["wrong_axes"]),
                "strong_wrong_candidate_diversity": len(row["strong_wrong_candidates"]),
                "strong_wrong_axis_diversity": len(row["strong_wrong_axes"]),
                "wrong_seed_diversity": len(row["wrong_seeds"]),
                "strong_wrong_seed_diversity": len(row["strong_wrong_seeds"]),
            }
        )
        sample["status"] = classify_row(sample)
        rows.append(sample)
    rows.sort(
        key=lambda row: (
            {"label_or_annotation_suspect": 0, "persistent_hard_sample": 1, "config_sensitive": 2, "mostly_unstable_only": 3, "sporadic": 4, "clean": 5}[row["status"]],
            -row["strong_wrong_rate"],
            -row["strong_wrong"],
            -row["all_wrong"],
            row["chart_id"],
        )
    )
    return rows


def copy_review_images(
    rows: list[dict[str, Any]],
    *,
    display_dir: Path,
    image_dir: Path,
    out_dir: Path,
    top_k: int,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = ["label_or_annotation_suspect", "persistent_hard_sample", "config_sensitive"]
    for status in categories:
        subset = [row for row in rows if row["status"] == status][:top_k]
        for row in subset:
            cls = row["true_class"]
            chart_id = row["chart_id"]
            src_display = display_dir / "test" / cls / f"{chart_id}.png"
            src_image = image_dir / "test" / cls / f"{chart_id}.png"
            src = src_display if src_display.exists() else src_image
            if not src.exists():
                continue
            dest_dir = out_dir / status
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{chart_id}__{cls}__wrong{row['strong_wrong']}-of-{row['strong_total']}.png"
            shutil.copy2(src, dest)


def write_outputs(
    *,
    out_prefix: Path,
    rows: list[dict[str, Any]],
    runs: list[RunSummary],
    report_label: str,
    review_dir: Path,
) -> None:
    ensure_parent(out_prefix)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")

    status_counts = Counter(row["status"] for row in rows if row["all_wrong"] > 0)
    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "report_label": report_label,
        "runs_total": len(runs),
        "runs_strong": sum(1 for run in runs if run.strong),
        "status_counts": dict(status_counts),
        "rows": rows,
        "review_dir": str(review_dir),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_fields = [
        "chart_id",
        "true_class",
        "error_role",
        "status",
        "strong_wrong",
        "strong_total",
        "strong_wrong_rate",
        "all_wrong",
        "all_total",
        "strong_wrong_axis_diversity",
        "strong_wrong_candidate_diversity",
        "device",
        "step",
        "item",
        "context_column",
        "target",
        "defect_start_idx",
        "target_value",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in csv_fields})

    def top_rows(status: str, top_k: int) -> list[dict[str, Any]]:
        return [row for row in rows if row["status"] == status][:top_k]

    lines = [
        "# Prediction Trend Review",
        "",
        f"- Label: `{report_label or 'latest'}`",
        f"- Runs scanned: `{len(runs)}`",
        f"- Strong runs (`test_f1 >= threshold`): `{sum(1 for run in runs if run.strong)}`",
        f"- Review images: `{review_dir}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(
        [
            "",
            "## Heuristic Meaning",
            "",
            "- `label_or_annotation_suspect`: strong runs across diverse axes still repeatedly disagree with the label. Treat as likely annotation issue or inherently ambiguous sample. Manual review required.",
            "- `persistent_hard_sample`: many strong runs miss it. Usually weak defect visibility or near-boundary sample rather than a simple tuning issue.",
            "- `config_sensitive`: some strong settings recover it and others miss it. Use this to explain FP/FN movement by axis.",
            "- `mostly_unstable_only`: mainly wrong in low-quality or unstable runs.",
            "",
        ]
    )

    for status, title in [
        ("label_or_annotation_suspect", "Label Suspect / Ambiguous"),
        ("persistent_hard_sample", "Persistent Hard Samples"),
        ("config_sensitive", "Config-Sensitive Samples"),
    ]:
        lines.extend(
            [
                f"## {title}",
                "",
                "| chart_id | true_class | role | strong wrong/total | strong wrong rate | wrong axis div | device | step | item | target |",
                "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        subset = top_rows(status, top_k=40)
        if not subset:
            lines.append("| none | - | - | - | - | - | - | - | - | - |")
        else:
            for row in subset:
                lines.append(
                    "| `{chart}` | `{cls}` | `{role}` | {wrong}/{total} | {rate:.2f} | {axis_div} | `{device}` | `{step}` | `{item}` | `{target}` |".format(
                        chart=row["chart_id"],
                        cls=row["true_class"],
                        role=row["error_role"],
                        wrong=row["strong_wrong"],
                        total=row["strong_total"],
                        rate=row["strong_wrong_rate"],
                        axis_div=row["strong_wrong_axis_diversity"],
                        device=row["device"],
                        step=row["step"],
                        item=row["item"],
                        target=row["target"],
                    )
                )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    scenarios, display_dir, image_dir = load_scenarios(Path(args.config))
    runs = collect_runs(args.candidate_prefix, args.min_f1)
    rows = build_rows(runs, scenarios)
    review_dir = out_prefix.parent / f"{out_prefix.name}_review"
    copy_review_images(
        rows,
        display_dir=display_dir,
        image_dir=image_dir,
        out_dir=review_dir,
        top_k=args.review_k,
    )
    write_outputs(
        out_prefix=out_prefix,
        rows=rows,
        runs=runs,
        report_label=args.report_label,
        review_dir=review_dir,
    )
    print(
        json.dumps(
            {
                "runs_total": len(runs),
                "runs_strong": sum(1 for run in runs if run.strong),
                "rows_with_errors": sum(1 for row in rows if row["all_wrong"] > 0),
                "review_dir": str(review_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
