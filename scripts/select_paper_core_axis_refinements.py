from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
DEFAULT_SEEDS = [42, 1, 2]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def fmt_float_tag(value: float) -> str:
    text = f"{value:.4g}"
    return text.replace(".", "p").replace("-", "m")


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


def aggregate_candidate_records(candidate: str) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        rf"^\d{{6}}_\d{{6}}_{re.escape(candidate)}_s(?P<seed>\d+)(?:_F[0-9.]+_R[0-9.]+)?$"
    )
    for path in LOGS.iterdir():
        if not path.is_dir():
            continue
        m = pattern.match(path.name)
        if not m:
            continue
        best = path / "best_info.json"
        if not best.exists():
            continue
        try:
            info = json.loads(best.read_text(encoding="utf-8"))
        except Exception:
            continue
        metric = extract_metrics(info)
        if metric is None:
            continue
        rows.append(
            {
                "seed": int(m.group("seed")),
                "f1": metric["f1"],
                "fn": metric["fn"],
                "fp": metric["fp"],
                "run_dir": str(path),
            }
        )
    if not rows:
        return None
    rows.sort(key=lambda row: row["seed"])
    return {
        "candidate": candidate,
        "complete": len(rows),
        "f1_mean": mean([row["f1"] for row in rows]),
        "fn_mean": mean([row["fn"] for row in rows]),
        "fp_mean": mean([row["fp"] for row in rows]),
        "worst_f1": min(row["f1"] for row in rows),
        "rows": rows,
    }


def tag_to_cli_suffix(tag: str) -> str:
    return tag


def candidate_exists(candidate: str) -> bool:
    if aggregate_candidate_records(candidate) is not None:
        return True
    return any(path.is_dir() for path in LOGS.glob(f"*_{candidate}_s*"))


@dataclass(frozen=True)
class AxisPoint:
    label: str
    candidate: str
    args: dict[str, Any]


def score_row(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        row["f1_mean"],
        row["worst_f1"],
        -row["fn_mean"],
        -row["fp_mean"],
    )


def error_sum(row: dict[str, Any]) -> float:
    return float(row["fn_mean"]) + float(row["fp_mean"])


def tested_points(points: list[AxisPoint], aggregates: dict[str, dict[str, Any]]) -> list[AxisPoint]:
    return [point for point in points if point.candidate in aggregates and aggregates[point.candidate]["complete"] >= 3]


def weak_axis_signal(
    *,
    points: list[AxisPoint],
    aggregates: dict[str, dict[str, Any]],
    min_err_gap: float = 2.0,
    min_f1_gap: float = 0.0015,
) -> bool:
    tested = tested_points(points, aggregates)
    if len(tested) < 2:
        return True
    ranked = sorted(
        tested,
        key=lambda point: (
            error_sum(aggregates[point.candidate]),
            -aggregates[point.candidate]["f1_mean"],
            point.label,
        ),
    )
    best = aggregates[ranked[0].candidate]
    second = aggregates[ranked[1].candidate]
    err_gap = error_sum(second) - error_sum(best)
    f1_gap = best["f1_mean"] - second["f1_mean"]
    return err_gap < min_err_gap and f1_gap < min_f1_gap


def build_numeric_queue(
    *,
    axis_name: str,
    points: list[AxisPoint],
    center_label: str | None,
    aggregates: dict[str, dict[str, Any]],
    queued_candidates: set[str],
    reason_prefix: str,
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    tested = [point for point in points if point.candidate in aggregates and aggregates[point.candidate]["complete"] >= 3]
    if not tested:
        return queue

    def maybe_enqueue(point: AxisPoint, reason: str) -> None:
        if point.candidate in queued_candidates or candidate_exists(point.candidate):
            return
        queued_candidates.add(point.candidate)
        queue.append(
            {
                "candidate": point.candidate,
                "args": point.args,
                "reason": f"{reason_prefix}: {reason}",
            }
        )

    if center_label is not None:
        center_point = next(point for point in points if point.label == center_label)
        if center_point.candidate not in aggregates:
            maybe_enqueue(center_point, f"explicit center control for {axis_name}")

    best_point = max(tested, key=lambda point: score_row(aggregates[point.candidate]))
    best_idx = points.index(best_point)
    tested_idxs = {points.index(point) for point in tested}

    def choose_lower() -> AxisPoint | None:
        lower_tested = [idx for idx in tested_idxs if idx < best_idx]
        if lower_tested:
            left_idx = max(lower_tested)
            between = [
                points[idx]
                for idx in range(left_idx + 1, best_idx)
                if points[idx].candidate not in aggregates and points[idx].candidate not in queued_candidates
            ]
            if between:
                return between[-1]
        below = [
            points[idx]
            for idx in range(0, best_idx)
            if points[idx].candidate not in aggregates and points[idx].candidate not in queued_candidates
        ]
        return below[-1] if below else None

    def choose_upper() -> AxisPoint | None:
        upper_tested = [idx for idx in tested_idxs if idx > best_idx]
        if upper_tested:
            right_idx = min(upper_tested)
            between = [
                points[idx]
                for idx in range(best_idx + 1, right_idx)
                if points[idx].candidate not in aggregates and points[idx].candidate not in queued_candidates
            ]
            if between:
                return between[0]
        above = [
            points[idx]
            for idx in range(best_idx + 1, len(points))
            if points[idx].candidate not in aggregates and points[idx].candidate not in queued_candidates
        ]
        return above[0] if above else None

    lower_point = choose_lower()
    upper_point = choose_upper()
    if lower_point is not None:
        maybe_enqueue(lower_point, f"refine below current best neighborhood around `{best_point.label}`")
    if upper_point is not None:
        maybe_enqueue(upper_point, f"refine above current best neighborhood around `{best_point.label}`")
    return queue


def build_smooth_queue(
    *,
    points: list[AxisPoint],
    aggregates: dict[str, dict[str, Any]],
    queued_candidates: set[str],
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []

    def maybe_enqueue(point: AxisPoint, reason: str) -> None:
        if point.candidate in queued_candidates or candidate_exists(point.candidate):
            return
        queued_candidates.add(point.candidate)
        queue.append(
            {
                "candidate": point.candidate,
                "args": point.args,
                "reason": f"smooth axis: {reason}",
            }
        )

    by_label = {point.label: point for point in points}
    if by_label["3med"].candidate not in aggregates:
        maybe_enqueue(by_label["3med"], "explicit center control for smoothing")
    if by_label["5mean"].candidate not in aggregates:
        maybe_enqueue(by_label["5mean"], "complete the 3/5 × mean/median smoothing grid")
    return queue


def build_stress_queue(
    *,
    prefix: str,
    base_n: int,
    config_path: str,
    aggregates: dict[str, dict[str, Any]],
    queued_candidates: set[str],
    stable_axes: dict[str, list[AxisPoint]],
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []

    common = {
        "--mode": "binary",
        "--config": config_path,
        "--epochs": 20,
        "--patience": 5,
        "--batch_size": 32,
        "--dropout": 0.0,
        "--precision": "fp16",
        "--num_workers": 0,
        "--ema_decay": 0.0,
        "--normal_ratio": base_n,
    }

    def maybe_enqueue(candidate: str, args: dict[str, Any], reason: str) -> None:
        if candidate in queued_candidates or candidate_exists(candidate):
            return
        queued_candidates.add(candidate)
        queue.append({"candidate": candidate, "args": args, "reason": reason})

    def candidate(tag: str) -> str:
        return f"{prefix}_v11_{tag}_n{base_n}"

    for tag, clip in [
        ("stressw3_gc00", 0.0),
        ("stressw3_gc025", 0.25),
        ("stressw3_gc05", 0.5),
        ("stressw3_gc10", 1.0),
        ("stressw3_gc20", 2.0),
        ("stressw3_gc50", 5.0),
    ]:
        args = dict(common)
        args.update(
            {
                "--smooth_window": 3,
                "--smooth_method": "median",
                "--lr_backbone": "2e-5",
                "--lr_head": "2e-4",
                "--warmup_epochs": 3,
                "--grad_clip": clip,
                "--weight_decay": 0.01,
            }
        )
        maybe_enqueue(
            candidate(tag),
            args,
            "gc stress axis: run dedicated rescue sweep on oscillatory warmup=3 instability cases",
        )

    if weak_axis_signal(points=stable_axes["wd"], aggregates=aggregates):
        for tag, wd in [
            ("stresslr5e5_wd000", 0.0),
            ("stresslr5e5_wd0005", 0.005),
            ("stresslr5e5_wd001", 0.01),
            ("stresslr5e5_wd002", 0.02),
            ("stresslr5e5_wd005", 0.05),
        ]:
            args = dict(common)
            args.update(
                {
                    "--smooth_window": 3,
                    "--smooth_method": "median",
                    "--lr_backbone": "5e-5",
                    "--lr_head": "5e-4",
                    "--warmup_epochs": 5,
                    "--grad_clip": 1.0,
                    "--weight_decay": wd,
                }
            )
            maybe_enqueue(
                candidate(tag),
                args,
                "wd stress axis: stable baseline gap is too small, so sweep WD on high-LR parent",
            )

    for tag, window, method in [
        ("stressw3_sw1raw", 1, "median"),
        ("stressw3_sw3med", 3, "median"),
        ("stressw3_sw3mean", 3, "mean"),
        ("stressw3_sw5med", 5, "median"),
        ("stressw3_sw5mean", 5, "mean"),
    ]:
        args = dict(common)
        args.update(
            {
                "--smooth_window": window,
                "--smooth_method": method,
                "--lr_backbone": "2e-5",
                "--lr_head": "2e-4",
                "--warmup_epochs": 3,
                "--grad_clip": 1.0,
                "--weight_decay": 0.01,
            }
        )
        maybe_enqueue(
            candidate(tag),
            args,
            "smooth stress axis: run dedicated rescue sweep on oscillatory warmup=3 instability cases",
        )

    for tag, ls_val in [
        ("stressw3_ls000", 0.0),
        ("stressw3_ls002", 0.02),
        ("stressw3_ls005", 0.05),
        ("stressw3_ls01", 0.10),
        ("stressw3_ls015", 0.15),
    ]:
        args = dict(common)
        args.update(
            {
                "--smooth_window": 3,
                "--smooth_method": "median",
                "--lr_backbone": "2e-5",
                "--lr_head": "2e-4",
                "--warmup_epochs": 3,
                "--grad_clip": 1.0,
                "--weight_decay": 0.01,
                "--label_smoothing": ls_val,
            }
        )
        maybe_enqueue(
            candidate(tag),
            args,
            "label-smoothing stress axis: run dedicated rescue sweep on oscillatory warmup=3 instability cases",
        )

    return queue


def queue_runs_payload(
    prefix: str,
    base_n: int,
    config_path: str,
    runs: list[dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    out_runs: list[dict[str, Any]] = []
    for row in runs:
        for seed in seeds:
            tag = f"{row['candidate']}_s{seed}"
            if candidate_exists(row["candidate"]):
                # partial candidate reruns are not expected in this selector
                pass
            out_runs.append(
                {
                    "tag": tag,
                    "candidate": row["candidate"],
                    "seed": int(seed),
                    "args": row["args"],
                    "reason": row["reason"],
                }
            )
    return {
        "created_at": now_iso(),
        "selected_reference": "fresh0412_v11_n700_existing",
        "selected_config": config_path,
        "prefix": prefix,
        "base_n": base_n,
        "note": "Adaptive core-axis refinement queue built from completed first-pass results.",
        "runs": out_runs,
    }


def build_axis_points(prefix: str, base_n: int, config_path: str) -> dict[str, list[AxisPoint]]:
    common = {
        "--mode": "binary",
        "--config": config_path,
        "--epochs": 20,
        "--patience": 5,
        "--batch_size": 32,
        "--dropout": 0.0,
        "--precision": "fp16",
        "--num_workers": 0,
        "--ema_decay": 0.0,
        "--normal_ratio": base_n,
    }

    def candidate(tag: str) -> str:
        return f"{prefix}_v11_{tag}_n{base_n}"

    def baseline_candidate() -> str:
        return f"{prefix}_v11_n{base_n}"

    def base_args() -> dict[str, Any]:
        return dict(common)

    lr_points = []
    for tag, lr_backbone, lr_head in [
        ("lr1e5", "1e-5", "1e-4"),
        ("lr1p5e5", "1.5e-5", "1.5e-4"),
        ("lr2e5", "2e-5", "2e-4"),
        ("lr2p5e5", "2.5e-5", "2.5e-4"),
        ("lr3e5", "3e-5", "3e-4"),
        ("lr4e5", "4e-5", "4e-4"),
        ("lr5e5", "5e-5", "5e-4"),
        ("lr7e5", "7e-5", "7e-4"),
        ("lr1e4", "1e-4", "1e-3"),
    ]:
        args = base_args()
        args.update(
            {
                "--smooth_window": 3,
                "--smooth_method": "median",
                "--lr_backbone": lr_backbone,
                "--lr_head": lr_head,
                "--warmup_epochs": 5,
                "--grad_clip": 1.0,
                "--weight_decay": 0.01,
            }
        )
        lr_points.append(AxisPoint(tag, candidate(tag), args))

    warm_points = []
    for label, warm, candidate_tag in [
        ("warm0", 0, "lrwarm0"),
        ("warm1", 1, "lrwarm1"),
        ("warm3", 3, "lrwarm3"),
        ("warm5", 5, "lr2e5"),
        ("warm8", 8, "lrwarm8"),
    ]:
        args = base_args()
        args.update(
            {
                "--smooth_window": 3,
                "--smooth_method": "median",
                "--lr_backbone": "2e-5",
                "--lr_head": "2e-4",
                "--warmup_epochs": warm,
                "--grad_clip": 1.0,
                "--weight_decay": 0.01,
            }
        )
        warm_points.append(AxisPoint(label, candidate(candidate_tag), args))

    gc_points = []
    for label, clip in [
        ("gc00", 0.0),
        ("gc025", 0.25),
        ("gc05", 0.5),
        ("gc10", 1.0),
        ("gc20", 2.0),
        ("gc50", 5.0),
    ]:
        args = base_args()
        args.update(
            {
                "--smooth_window": 3,
                "--smooth_method": "median",
                "--lr_backbone": "2e-5",
                "--lr_head": "2e-4",
                "--warmup_epochs": 5,
                "--grad_clip": clip,
                "--weight_decay": 0.01,
            }
        )
        gc_points.append(AxisPoint(label, candidate(label), args))

    wd_points = []
    for label, wd in [
        ("wd000", 0.0),
        ("wd0005", 0.005),
        ("wd001", 0.01),
        ("wd002", 0.02),
        ("wd005", 0.05),
    ]:
        args = base_args()
        args.update(
            {
                "--smooth_window": 3,
                "--smooth_method": "median",
                "--lr_backbone": "2e-5",
                "--lr_head": "2e-4",
                "--warmup_epochs": 5,
                "--grad_clip": 1.0,
                "--weight_decay": wd,
            }
        )
        wd_points.append(AxisPoint(label, candidate(label), args))

    smooth_points = []
    for label, window, method in [
        ("1raw", 1, "median"),
        ("3med", 3, "median"),
        ("3mean", 3, "mean"),
        ("5med", 5, "median"),
        ("5mean", 5, "mean"),
    ]:
        args = base_args()
        args.update(
            {
                "--smooth_window": window,
                "--smooth_method": method,
                "--lr_backbone": "2e-5",
                "--lr_head": "2e-4",
                "--warmup_epochs": 5,
                "--grad_clip": 1.0,
                "--weight_decay": 0.01,
            }
        )
        smooth_points.append(AxisPoint(label, candidate(f"sw{label}"), args))

    ls_points = []
    for label, cand_name, ls_val in [
        ("ls000", baseline_candidate(), 0.0),
        ("ls002", candidate("regls002"), 0.02),
        ("ls005", candidate("regls005"), 0.05),
        ("ls010", candidate("regls01"), 0.10),
        ("ls015", candidate("regls015"), 0.15),
        ("ls020", candidate("regls02"), 0.20),
    ]:
        args = base_args()
        args.update(
            {
                "--smooth_window": 3,
                "--smooth_method": "median",
                "--lr_backbone": "2e-5",
                "--lr_head": "2e-4",
                "--warmup_epochs": 5,
                "--grad_clip": 1.0,
                "--weight_decay": 0.01,
                "--label_smoothing": ls_val,
            }
        )
        ls_points.append(AxisPoint(label, cand_name, args))

    dp_points = []
    for label, cand_name, dp_val in [
        ("dp000", baseline_candidate(), 0.0),
        ("dp005", candidate("regdp005"), 0.05),
        ("dp010", candidate("regdp01"), 0.10),
        ("dp020", candidate("regdp02"), 0.20),
        ("dp030", candidate("regdp03"), 0.30),
    ]:
        args = base_args()
        args.update(
            {
                "--smooth_window": 3,
                "--smooth_method": "median",
                "--lr_backbone": "2e-5",
                "--lr_head": "2e-4",
                "--warmup_epochs": 5,
                "--grad_clip": 1.0,
                "--weight_decay": 0.01,
                "--stochastic_depth_rate": dp_val,
            }
        )
        dp_points.append(AxisPoint(label, cand_name, args))

    return {
        "lr": lr_points,
        "warmup": warm_points,
        "gc": gc_points,
        "wd": wd_points,
        "smooth": smooth_points,
        "label_smoothing": ls_points,
        "stochastic_depth": dp_points,
    }


def collect_aggregates(points_by_axis: dict[str, list[AxisPoint]]) -> dict[str, dict[str, Any]]:
    candidates = {
        point.candidate
        for axis_points in points_by_axis.values()
        for point in axis_points
    }
    aggregates: dict[str, dict[str, Any]] = {}
    for candidate in sorted(candidates):
        agg = aggregate_candidate_records(candidate)
        if agg is not None:
            aggregates[candidate] = agg
    return aggregates


def write_decision_markdown(
    path: Path,
    *,
    axes: dict[str, list[AxisPoint]],
    aggregates: dict[str, dict[str, Any]],
    queued: list[dict[str, Any]],
) -> None:
    lines = [
        "# Adaptive Core-Axis Refinement Decision",
        "",
        f"- Updated: `{now_iso()}`",
        f"- New candidates queued: `{len(queued)}`",
        "",
    ]
    for axis_name, points in axes.items():
        lines.append(f"## {axis_name}")
        lines.append("")
        lines.append("| Point | Complete | F1 Mean | FN Mean | FP Mean |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for point in points:
            agg = aggregates.get(point.candidate)
            if agg is None:
                lines.append(f"| `{point.label}` | 0 | - | - | - |")
            else:
                lines.append(
                    f"| `{point.label}` | {agg['complete']} | {agg['f1_mean']:.4f} | {agg['fn_mean']:.1f} | {agg['fp_mean']:.1f} |"
                )
        lines.append("")
    lines.append("## Queue")
    lines.append("")
    if queued:
        for row in queued:
            lines.append(f"- `{row['candidate']}`: {row['reason']}")
    else:
        lines.append("- no new refinement candidates selected")
    lines.append("")
    write_text(path, "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Select adaptive follow-up points for paper core axes")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--base-n", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-queue", type=Path, required=True)
    parser.add_argument("--decision-md", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    points_by_axis = build_axis_points(args.prefix, args.base_n, args.config)
    aggregates = collect_aggregates(points_by_axis)
    queued_candidates: set[str] = set()
    queued: list[dict[str, Any]] = []

    queued.extend(
        build_numeric_queue(
            axis_name="lr",
            points=points_by_axis["lr"],
            center_label="lr2e5",
            aggregates=aggregates,
            queued_candidates=queued_candidates,
            reason_prefix="lr axis",
        )
    )
    queued.extend(
        build_numeric_queue(
            axis_name="warmup",
            points=points_by_axis["warmup"],
            center_label="warm5",
            aggregates=aggregates,
            queued_candidates=queued_candidates,
            reason_prefix="warmup axis",
        )
    )
    queued.extend(
        build_numeric_queue(
            axis_name="grad_clip",
            points=points_by_axis["gc"],
            center_label="gc10",
            aggregates=aggregates,
            queued_candidates=queued_candidates,
            reason_prefix="grad-clip axis",
        )
    )
    queued.extend(
        build_numeric_queue(
            axis_name="weight_decay",
            points=points_by_axis["wd"],
            center_label="wd001",
            aggregates=aggregates,
            queued_candidates=queued_candidates,
            reason_prefix="weight-decay axis",
        )
    )
    queued.extend(
        build_smooth_queue(
            points=points_by_axis["smooth"],
            aggregates=aggregates,
            queued_candidates=queued_candidates,
        )
    )
    queued.extend(
        build_numeric_queue(
            axis_name="label_smoothing",
            points=points_by_axis["label_smoothing"],
            center_label="ls005",
            aggregates=aggregates,
            queued_candidates=queued_candidates,
            reason_prefix="label-smoothing axis",
        )
    )
    queued.extend(
        build_numeric_queue(
            axis_name="stochastic_depth",
            points=points_by_axis["stochastic_depth"],
            center_label="dp010",
            aggregates=aggregates,
            queued_candidates=queued_candidates,
            reason_prefix="stochastic-depth axis",
        )
    )
    queued.extend(
        build_stress_queue(
            prefix=args.prefix,
            base_n=args.base_n,
            config_path=args.config,
            aggregates=aggregates,
            queued_candidates=queued_candidates,
            stable_axes={
                "gc": points_by_axis["gc"],
                "wd": points_by_axis["wd"],
                "smooth": points_by_axis["smooth"],
                "label_smoothing": points_by_axis["label_smoothing"],
            },
        )
    )

    payload = queue_runs_payload(
        prefix=args.prefix,
        base_n=args.base_n,
        config_path=args.config,
        runs=queued,
        seeds=args.seeds,
    )
    write_json(args.out_queue, payload)
    write_decision_markdown(
        args.decision_md,
        axes=points_by_axis,
        aggregates=aggregates,
        queued=queued,
    )
    print(json.dumps({"queued_candidates": len(queued), "queued_runs": len(payload["runs"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
