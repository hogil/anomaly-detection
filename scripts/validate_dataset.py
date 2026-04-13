import argparse
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


@dataclass
class ScenarioMeta:
    chart_id: str
    split: str
    cls: str
    context_column: str
    target: str
    defect_start_idx: int
    defect_params: dict


def parse_args():
    parser = argparse.ArgumentParser(description="Validate generated anomaly dataset.")
    parser.add_argument("--config", default="dataset.yaml", help="Dataset config YAML")
    parser.add_argument("--scenarios", default="data/scenarios.csv", help="Scenario CSV")
    parser.add_argument("--timeseries", default="data/timeseries.csv", help="Timeseries CSV")
    parser.add_argument("--display-dir", default="display", help="Rendered display image root")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Validation output directory. Default: validations/<timestamp>",
    )
    parser.add_argument(
        "--weak-per-class",
        type=int,
        default=20,
        help="Number of weakest samples to copy per class",
    )
    return parser.parse_args()


def scaled_enforcement(enforcement: dict, split: str, test_scale: float) -> dict:
    if split != "test":
        return dict(enforcement)

    scaled = dict(enforcement)
    for key in (
        "mean_shift_floor_sigma",
        "spike_floor_sigma",
        "drift_floor_sigma",
        "context_floor_sigma",
    ):
        if key in scaled:
            scaled[key] = float(scaled[key]) * test_scale
    if "std_floor_ratio" in scaled:
        scaled["std_floor_ratio"] = 1 + (float(scaled["std_floor_ratio"]) - 1) * test_scale
    return scaled


def load_scenarios(path: Path) -> dict[str, ScenarioMeta]:
    scenarios: dict[str, ScenarioMeta] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            defect_params = {}
            if row.get("defect_params"):
                defect_params = json.loads(row["defect_params"])
            scenarios[row["chart_id"]] = ScenarioMeta(
                chart_id=row["chart_id"],
                split=row["split"],
                cls=row["class"],
                context_column=row["context_column"],
                target=row["target"],
                defect_start_idx=int(row["defect_start_idx"]),
                defect_params=defect_params,
            )
    return scenarios


def _append_row(chart_rows: list[dict], row: dict):
    chart_rows.append(
        {
            "time_index": int(row["time_index"]),
            "value": float(row["value"]),
            "context_value": row.get("eqp_id")
            if row["context_column"] == "eqp_id"
            else row.get(row["context_column"], ""),
        }
    )


def compute_ref_noise(target_rows: list[dict], fleet_groups: dict[str, list[float]], defect_start: int) -> float:
    left_std = compute_baseline_std(target_rows, defect_start)
    fleet_within_stds = [
        float(np.std(np.array(vals, dtype=np.float64)))
        for vals in fleet_groups.values()
        if len(vals) >= 2
    ]
    fleet_within_std = float(np.mean(fleet_within_stds)) if fleet_within_stds else 0.0
    return max(left_std, fleet_within_std, 1e-6)


def compute_baseline_std(target_rows: list[dict], defect_start: int) -> float:
    if defect_start >= 0:
        left_vals = np.array(
            [r["value"] for r in target_rows if r["time_index"] < defect_start],
            dtype=np.float64,
        )
    else:
        left_vals = np.array([r["value"] for r in target_rows], dtype=np.float64)
    return float(np.std(left_vals)) if len(left_vals) >= 2 else 0.0


def compute_time_span(rows: list[dict]) -> int:
    if len(rows) < 2:
        return 0
    indices = [int(r["time_index"]) for r in rows]
    return max(indices) - min(indices)


def compute_metric(
    meta: ScenarioMeta,
    chart_rows: list[dict],
    enforcement: dict,
    display_dir: Path,
    drift_visual_floor: float,
    drift_span_floor_ratio: float,
) -> dict:
    target_rows = sorted(
        [r for r in chart_rows if r["context_value"] == meta.target],
        key=lambda r: r["time_index"],
    )
    fleet_rows = [r for r in chart_rows if r["context_value"] != meta.target]

    fleet_groups: dict[str, list[float]] = defaultdict(list)
    for row in fleet_rows:
        fleet_groups[row["context_value"]].append(row["value"])

    left_std = compute_baseline_std(target_rows, meta.defect_start_idx)
    ref_noise = compute_ref_noise(target_rows, fleet_groups, meta.defect_start_idx)
    threshold = None
    weak_score = None
    details = {}

    if meta.cls == "mean_shift":
        baseline = np.array(
            [r["value"] for r in target_rows if r["time_index"] < meta.defect_start_idx],
            dtype=np.float64,
        )
        affected = np.array(
            [r["value"] for r in target_rows if r["time_index"] >= meta.defect_start_idx],
            dtype=np.float64,
        )
        fleet_affected = np.array(
            [r["value"] for r in fleet_rows if r["time_index"] >= meta.defect_start_idx],
            dtype=np.float64,
        )
        fleet_mean = float(np.mean(fleet_affected)) if len(fleet_affected) else 0.0
        baseline_mean = float(np.mean(baseline)) if len(baseline) else fleet_mean
        self_ref_noise = max(left_std, 1e-6)
        fleet_gap_sigma = abs(float(np.mean(affected)) - fleet_mean) / ref_noise
        self_shift_sigma = abs(float(np.mean(affected)) - baseline_mean) / self_ref_noise
        fleet_threshold = float(enforcement["mean_shift_floor_sigma"])
        self_threshold = float(enforcement.get("mean_shift_self_floor_sigma", fleet_threshold))
        weak_score = min(
            fleet_gap_sigma / max(fleet_threshold, 1e-6),
            self_shift_sigma / max(self_threshold, 1e-6),
        )
        threshold = 1.0
        details = {
            "fleet_gap_sigma": round(fleet_gap_sigma, 4),
            "self_shift_sigma": round(self_shift_sigma, 4),
            "baseline_mean": round(baseline_mean, 4),
            "fleet_threshold": round(fleet_threshold, 4),
            "self_threshold": round(self_threshold, 4),
        }

    elif meta.cls == "standard_deviation":
        affected = np.array(
            [r["value"] for r in target_rows if r["time_index"] >= meta.defect_start_idx],
            dtype=np.float64,
        )
        weak_score = float(np.std(affected)) / ref_noise
        threshold = float(enforcement["std_floor_ratio"])
        details = {
            "std_ratio": round(weak_score, 4),
            "ref_noise": round(ref_noise, 4),
        }

    elif meta.cls == "spike":
        affected = np.array(
            [r["value"] for r in target_rows if r["time_index"] >= meta.defect_start_idx],
            dtype=np.float64,
        )
        fleet_affected = np.array(
            [r["value"] for r in fleet_rows if r["time_index"] >= meta.defect_start_idx],
            dtype=np.float64,
        )
        fleet_mean = float(np.mean(fleet_affected)) if len(fleet_affected) else 0.0
        deviations = np.abs(affected - fleet_mean)
        peak_sigma = float(np.max(deviations)) / ref_noise if len(deviations) else 0.0
        points_ge_floor = int(np.sum(deviations / ref_noise >= enforcement["spike_floor_sigma"]))
        weak_score = peak_sigma
        threshold = float(enforcement["spike_floor_sigma"])
        details = {
            "peak_sigma": round(peak_sigma, 4),
            "points_ge_floor": points_ge_floor,
            "affected_points": int(len(affected)),
        }

    elif meta.cls == "drift":
        affected = np.array(
            [r["value"] for r in target_rows if r["time_index"] >= meta.defect_start_idx],
            dtype=np.float64,
        )
        affected_rows = [r for r in target_rows if r["time_index"] >= meta.defect_start_idx]
        q = max(2, len(affected) // 4)
        start_mean = float(np.mean(affected[:q])) if len(affected) else 0.0
        end_mean = float(np.mean(affected[-q:])) if len(affected) else 0.0
        drift_ref_noise = max(left_std, 1e-6)
        weak_score = abs(end_mean - start_mean) / drift_ref_noise
        threshold = max(float(enforcement["drift_floor_sigma"]), float(drift_visual_floor))
        affected_time_span = compute_time_span(affected_rows)
        chart_time_span = compute_time_span(chart_rows)
        span_ratio = affected_time_span / max(chart_time_span, 1)
        span_pass = span_ratio >= drift_span_floor_ratio
        details = {
            "drift_sigma_baseline": round(weak_score, 4),
            "drift_sigma_ref": round(abs(end_mean - start_mean) / ref_noise, 4),
            "baseline_std": round(drift_ref_noise, 4),
            "segment_points": int(len(affected)),
            "segment_time_span": int(affected_time_span),
            "chart_time_span": int(chart_time_span),
            "span_ratio": round(span_ratio, 4),
            "span_floor_ratio": round(drift_span_floor_ratio, 4),
            "span_pass": bool(span_pass),
        }

    elif meta.cls == "context":
        deviation_type = meta.defect_params.get("deviation_type", "mean")
        mean_vis = float(meta.defect_params.get("post_mean_nsigma", 0.0))
        std_vis = float(meta.defect_params.get("post_std_nsigma", 0.0))
        if deviation_type == "mean":
            weak_score = mean_vis
        elif deviation_type == "std":
            weak_score = std_vis
        else:
            weak_score = min(mean_vis, std_vis)
        threshold = float(enforcement["context_floor_sigma"])
        details = {
            "mean_vis": round(mean_vis, 4),
            "std_vis": round(std_vis, 4),
            "deviation_type": deviation_type,
        }
    else:
        raise ValueError(f"Unsupported class for validation: {meta.cls}")

    pass_threshold = weak_score >= threshold
    if meta.cls == "drift":
        pass_threshold = pass_threshold and span_pass
    image_path = display_dir / meta.split / meta.cls / f"{meta.chart_id}.png"
    sort_score = weak_score / max(threshold, 1e-6)
    if meta.cls == "drift" and drift_span_floor_ratio > 0:
        sort_score = min(sort_score, span_ratio / drift_span_floor_ratio)
    return {
        "chart_id": meta.chart_id,
        "class": meta.cls,
        "split": meta.split,
        "threshold": round(float(threshold), 4),
        "weak_score": round(float(weak_score), 6),
        "sort_score": round(float(sort_score), 6),
        "pass_threshold": pass_threshold,
        "details": json.dumps(details, ensure_ascii=False),
        "image_path": str(image_path),
    }


def process_timeseries(
    scenarios: dict[str, ScenarioMeta],
    timeseries_path: Path,
    enforcement_cfg: dict,
    test_scale: float,
    display_dir: Path,
    drift_visual_floor: float,
    drift_span_floor_ratio: float,
) -> list[dict]:
    results: list[dict] = []
    current_chart_id = None
    current_rows: list[dict] = []
    current_context_column = None

    def finalize(chart_id: str, rows: list[dict], context_column: str):
        if not chart_id or chart_id not in scenarios:
            return
        meta = scenarios[chart_id]
        if meta.cls == "normal":
            return
        enforcement = scaled_enforcement(enforcement_cfg, meta.split, test_scale)
        results.append(
            compute_metric(
                meta,
                rows,
                enforcement,
                display_dir,
                drift_visual_floor,
                drift_span_floor_ratio,
            )
        )

    with timeseries_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            chart_id = raw["chart_id"]
            if chart_id != current_chart_id:
                finalize(current_chart_id, current_rows, current_context_column)
                current_chart_id = chart_id
                current_rows = []
                current_context_column = scenarios[chart_id].context_column if chart_id in scenarios else None
            if current_context_column is None:
                continue
            raw["context_column"] = current_context_column
            _append_row(current_rows, raw)
        finalize(current_chart_id, current_rows, current_context_column)

    return results


def summarize(rows: list[dict]) -> dict:
    summary = {}
    grouped: dict[str, list[float]] = defaultdict(list)
    pass_count: dict[str, int] = defaultdict(int)

    for row in rows:
        grouped[row["class"]].append(float(row["weak_score"]))
        pass_count[row["class"]] += int(bool(row["pass_threshold"]))

    for cls, values in grouped.items():
        arr = np.array(values, dtype=np.float64)
        summary[cls] = {
            "count": int(len(arr)),
            "pass_count": int(pass_count[cls]),
            "fail_count": int(len(arr) - pass_count[cls]),
            "min": round(float(arr.min()), 6),
            "p1": round(float(np.percentile(arr, 1)), 6),
            "p5": round(float(np.percentile(arr, 5)), 6),
            "median": round(float(np.median(arr)), 6),
        }
    return summary


def export_weak_samples(rows: list[dict], output_dir: Path, weak_per_class: int):
    weak_rows = []
    weak_root = output_dir / "weak_samples"
    weak_root.mkdir(parents=True, exist_ok=True)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)

    for cls, cls_rows in by_class.items():
        cls_dir = weak_root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        weakest = sorted(cls_rows, key=lambda r: r.get("sort_score", r["weak_score"]))[:weak_per_class]
        for rank, row in enumerate(weakest, start=1):
            src = Path(row["image_path"])
            dst = cls_dir / f"{rank:02d}_score_{row['weak_score']:.4f}_{row['split']}_{row['chart_id']}.png"
            if src.exists():
                shutil.copy2(src, dst)
                copied_to = str(dst)
            else:
                copied_to = ""
            weak_rows.append(
                {
                    "rank_in_class": rank,
                    "class": cls,
                    "split": row["split"],
                    "chart_id": row["chart_id"],
                    "weak_score": row["weak_score"],
                    "sort_score": row.get("sort_score", row["weak_score"]),
                    "threshold": row["threshold"],
                    "pass_threshold": row["pass_threshold"],
                    "details": row["details"],
                    "copied_to": copied_to,
                    "original_display": row["image_path"],
                }
            )

    summary_csv = weak_root / "summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank_in_class",
                "class",
                "split",
                "chart_id",
                "weak_score",
                "sort_score",
                "threshold",
                "pass_threshold",
                "details",
                "copied_to",
                "original_display",
            ],
        )
        writer.writeheader()
        writer.writerows(weak_rows)


def save_outputs(rows: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    per_chart_path = output_dir / "per_chart.csv"
    with per_chart_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chart_id",
                "class",
                "split",
                "weak_score",
                "sort_score",
                "threshold",
                "pass_threshold",
                "details",
                "image_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main():
    args = parse_args()
    config_path = Path(args.config)
    scenarios_path = Path(args.scenarios)
    timeseries_path = Path(args.timeseries)
    display_dir = Path(args.display_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        output_dir = Path("validations") / f"dataset_{ts}"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    enforcement_cfg = config["defect"]["enforcement"]
    drift_visual_floor = float(
        config.get("defect", {}).get("drift", {}).get(
            "visual_floor_sigma",
            enforcement_cfg.get("drift_floor_sigma", 1.8),
        )
    )
    drift_span_floor_ratio = float(
        config.get("defect", {}).get("drift", {}).get(
            "visible_span_ratio_range",
            [0.0, 0.0],
        )[0]
    )
    test_scale = float(config["dataset"].get("test_difficulty_scale", 1.0))

    print(f"[load] scenarios: {scenarios_path}")
    scenarios = load_scenarios(scenarios_path)
    print(f"[load] scenarios loaded: {len(scenarios)}")
    print(f"[stream] timeseries: {timeseries_path}")
    rows = process_timeseries(
        scenarios=scenarios,
        timeseries_path=timeseries_path,
        enforcement_cfg=enforcement_cfg,
        test_scale=test_scale,
        display_dir=display_dir,
        drift_visual_floor=drift_visual_floor,
        drift_span_floor_ratio=drift_span_floor_ratio,
    )
    print(f"[done] validated charts: {len(rows)}")

    summary = save_outputs(rows, output_dir)
    export_weak_samples(rows, output_dir, args.weak_per_class)

    print(f"[saved] {output_dir / 'per_chart.csv'}")
    print(f"[saved] {output_dir / 'summary.json'}")
    print(f"[saved] {output_dir / 'weak_samples' / 'summary.csv'}")
    print()
    print("Class summary:")
    for cls in sorted(summary):
        item = summary[cls]
        print(
            f"  {cls:20s} count={item['count']:4d} "
            f"pass={item['pass_count']:4d} fail={item['fail_count']:4d} "
            f"min={item['min']:.4f} p5={item['p5']:.4f} median={item['median']:.4f}"
        )


if __name__ == "__main__":
    main()
