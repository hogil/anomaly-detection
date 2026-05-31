#!/usr/bin/env python
"""Run dataset-level and global consensus BKM candidates.

Input groups use the same format as generate_cross_dataset_report.py:

    group=config.yaml=model.name;group2=config2.yaml=model.name

The script reads each cell's 01_baseline_results.json, 02_sweep_active.json,
and 02_sweep_results.json. For each tracked axis it selects the condition with
the best mean F1 over the requested cells, then launches the selected condition
back through adaptive_experiment_controller.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = "42,1,2,3,4"
DEFAULT_MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"

BKM_AXIS_GROUPS: "OrderedDict[str, list[str]]" = OrderedDict(
    [
        ("lr", ["--lr_backbone", "--lr_head"]),
        ("warmup", ["--warmup_epochs"]),
        ("normal_ratio", ["--normal_ratio"]),
        ("per_class", ["--max_per_class"]),
        ("weight_decay", ["--weight_decay"]),
        ("smoothing", ["--smooth_window", "--smooth_method"]),
        ("label_smoothing", ["--label_smoothing"]),
        ("asl", ["--asl_gamma_neg", "--asl_gamma_pos", "--asl_clip"]),
        ("stochastic_depth", ["--stochastic_depth_rate"]),
        ("focal_gamma", ["--focal_gamma"]),
        ("abnormal_weight", ["--abnormal_weight"]),
        ("ema", ["--ema_decay"]),
        ("allow_tie_save", ["--allow_tie_save"]),
    ]
)

BASE_ARGS: dict[str, Any] = {
    "--mode": "binary",
    "--config": "dataset.yaml",
    "--epochs": 20,
    "--patience": 5,
    "--batch_size": 32,
    "--dropout": 0.0,
    "--precision": "fp16",
    "--num_workers": 0,
    "--prefetch_factor": 2,
    "--smooth_window": 1,
    "--smooth_method": "median",
    "--lr_backbone": "2e-5",
    "--lr_head": "2e-4",
    "--warmup_epochs": 5,
    "--weight_decay": 0.01,
    "--normal_ratio": 700,
    "--grad_clip": 0.0,
    "--max_per_class": 0,
    "--label_smoothing": 0.0,
    "--asl_gamma_neg": 0.0,
    "--asl_gamma_pos": 0.0,
    "--asl_clip": 0.0,
    "--stochastic_depth_rate": 0.0,
    "--focal_gamma": 0.0,
    "--abnormal_weight": 1.0,
    "--ema_decay": 0.0,
    "--allow_tie_save": False,
}


@dataclass(frozen=True)
class Cell:
    group: str
    config: str
    model: str


@dataclass
class Score:
    cell: Cell
    axis: str
    key: str
    values: dict[str, Any]
    candidate: str
    f1: float
    fn: float
    fp: float
    seeds: int


@dataclass
class Selection:
    axis: str
    key: str
    values: dict[str, Any]
    candidate: str
    mean_f1: float
    mean_fn: float
    mean_fp: float
    cells: int


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_name(run: dict[str, Any]) -> str:
    candidate = run.get("candidate")
    if candidate:
        return str(candidate)
    return re.sub(r"_s\d+$", "", str(run.get("tag", "")))


def normalize_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            return float(value)
        except ValueError:
            return value
    return value


def canonical_values(axis: str, values: dict[str, Any]) -> str:
    payload = {flag: normalize_value(values.get(flag, BASE_ARGS.get(flag))) for flag in BKM_AXIS_GROUPS[axis]}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def values_for_axis(args: dict[str, Any], axis: str) -> tuple[bool, dict[str, Any]]:
    values: dict[str, Any] = {}
    changed = False
    for flag in BKM_AXIS_GROUPS[axis]:
        value = args.get(flag, BASE_ARGS.get(flag))
        values[flag] = value
        if normalize_value(value) != normalize_value(BASE_ARGS.get(flag)):
            changed = True
    return changed, values


def one_changed_axis(args: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    changed: list[tuple[str, dict[str, Any]]] = []
    for axis in BKM_AXIS_GROUPS:
        is_changed, values = values_for_axis(args, axis)
        if is_changed:
            changed.append((axis, values))
    if len(changed) != 1:
        return None
    return changed[0]


def score_rows_by_candidate(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results.get("runs", {}).values():
        if not isinstance(row, dict) or row.get("status") != "complete":
            continue
        f1 = row.get("test_f1")
        if f1 is None:
            continue
        rows[str(row.get("candidate") or candidate_name(row))].append(row)

    out: dict[str, dict[str, Any]] = {}
    for candidate, items in rows.items():
        f1s = [float(item["test_f1"]) for item in items if item.get("test_f1") is not None]
        if not f1s:
            continue
        fns = [float(item.get("fn", 0) or 0) for item in items]
        fps = [float(item.get("fp", 0) or 0) for item in items]
        out[candidate] = {
            "f1": mean(f1s),
            "fn": mean(fns) if fns else 0.0,
            "fp": mean(fps) if fps else 0.0,
            "seeds": len(f1s),
        }
    return out


def runtime_args_from_active(active: dict[str, Any]) -> dict[str, Any]:
    for run in active.get("runs", []):
        args = run.get("args", {}) if isinstance(run, dict) else {}
        if not isinstance(args, dict):
            continue
        out: dict[str, Any] = {}
        for flag in ("--batch_size", "--num_workers", "--prefetch_factor"):
            if flag in args:
                out[flag] = args[flag]
        if out:
            return out
    return {}


def load_cell_scores(cell: Cell, validations_root: Path, require_complete: bool) -> tuple[list[Score], dict[str, Any]]:
    cell_dir = validations_root / cell.group
    active_path = cell_dir / "02_sweep_active.json"
    sweep_path = cell_dir / "02_sweep_results.json"
    baseline_path = cell_dir / "01_baseline_results.json"
    missing = [p for p in (active_path, sweep_path, baseline_path) if not p.exists()]
    if missing:
        msg = ", ".join(str(p) for p in missing)
        if require_complete:
            raise FileNotFoundError(f"missing BKM source files for {cell.group}: {msg}")
        print(f"[consensus-bkm] skip incomplete cell {cell.group}: {msg}")
        return [], {}

    active = read_json(active_path)
    sweep_results = read_json(sweep_path)
    baseline_results = read_json(baseline_path)
    runtime_args = runtime_args_from_active(active)

    cand_args: dict[str, dict[str, Any]] = {}
    for run in active.get("runs", []):
        if not isinstance(run, dict):
            continue
        args = run.get("args", {})
        if not isinstance(args, dict):
            continue
        cand_args.setdefault(candidate_name(run), args)

    cand_scores = score_rows_by_candidate(sweep_results)
    baseline_scores = score_rows_by_candidate(baseline_results)
    baseline = None
    if baseline_scores:
        baseline = max(baseline_scores.values(), key=lambda s: (s["f1"], -s["fn"], -s["fp"], s["seeds"]))

    scores: list[Score] = []
    if baseline is not None:
        for axis in BKM_AXIS_GROUPS:
            base_values = {flag: BASE_ARGS.get(flag) for flag in BKM_AXIS_GROUPS[axis]}
            scores.append(
                Score(
                    cell=cell,
                    axis=axis,
                    key=f"{axis}:{canonical_values(axis, base_values)}",
                    values=base_values,
                    candidate="baseline",
                    f1=float(baseline["f1"]),
                    fn=float(baseline["fn"]),
                    fp=float(baseline["fp"]),
                    seeds=int(baseline["seeds"]),
                )
            )

    for candidate, args in cand_args.items():
        score = cand_scores.get(candidate)
        if score is None:
            continue
        axis_values = one_changed_axis(args)
        if axis_values is None:
            continue
        axis, values = axis_values
        scores.append(
            Score(
                cell=cell,
                axis=axis,
                key=f"{axis}:{canonical_values(axis, values)}",
                values=values,
                candidate=candidate,
                f1=float(score["f1"]),
                fn=float(score["fn"]),
                fp=float(score["fp"]),
                seeds=int(score["seeds"]),
            )
        )
    if require_complete and not scores:
        raise RuntimeError(f"no complete BKM source scores for {cell.group}")
    return scores, runtime_args


def select_consensus(scores: list[Score], total_cells: int) -> dict[str, Selection]:
    by_axis_key: dict[tuple[str, str], list[Score]] = defaultdict(list)
    for score in scores:
        by_axis_key[(score.axis, score.key)].append(score)

    selected: dict[str, Selection] = {}
    for axis in BKM_AXIS_GROUPS:
        options = [(key, items) for (ax, key), items in by_axis_key.items() if ax == axis]
        if not options:
            continue
        max_coverage = max(len(items) for _, items in options)
        options = [(key, items) for key, items in options if len(items) == max_coverage]

        def rank(option: tuple[str, list[Score]]) -> tuple[float, float, float, int]:
            _, items = option
            return (
                mean(item.f1 for item in items),
                -mean(item.fn for item in items),
                -mean(item.fp for item in items),
                len(items),
            )

        key, items = max(options, key=rank)
        first = items[0]
        selected[axis] = Selection(
            axis=axis,
            key=key,
            values=first.values,
            candidate=first.candidate,
            mean_f1=mean(item.f1 for item in items),
            mean_fn=mean(item.fn for item in items),
            mean_fp=mean(item.fp for item in items),
            cells=len(items),
        )
        if len(items) < total_cells:
            print(
                f"[consensus-bkm] warn: axis {axis} selected from {len(items)}/{total_cells} cells",
                flush=True,
            )
    return selected


def parse_groups(raw: str) -> list[Cell]:
    cells: list[Cell] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split("=")
        if len(parts) == 2:
            group, config = parts
            model = DEFAULT_MODEL_NAME
        elif len(parts) == 3:
            group, config, model = parts
        else:
            raise ValueError(f"invalid group entry: {item!r}")
        cells.append(Cell(group=group, config=config, model=model))
    if not cells:
        raise ValueError("--groups did not contain any cells")
    return cells


def short_name(value: str) -> str:
    stem = Path(value).name
    stem = re.sub(r"\.ya?ml$", "", stem)
    stem = stem.split(".", 1)[0]
    return re.sub(r"[^A-Za-z0-9]+", "", stem).lower()


def apply_selection(base: dict[str, Any], selection: dict[str, Selection]) -> dict[str, Any]:
    args = dict(base)
    for entry in selection.values():
        for flag, value in entry.values.items():
            args[flag] = value

    if any(float(args.get(flag, 0.0) or 0.0) > 0 for flag in ("--asl_gamma_neg", "--asl_gamma_pos", "--asl_clip")):
        args["--label_smoothing"] = 0.0
        args["--focal_gamma"] = 0.0
    return args


def build_base_args(cell: Cell, runtime: dict[str, Any], cli: argparse.Namespace) -> dict[str, Any]:
    args = dict(BASE_ARGS)
    args["--config"] = cell.config
    args["--model_name"] = cell.model
    for flag, attr in (
        ("--batch_size", "batch_size"),
        ("--num_workers", "num_workers"),
        ("--prefetch_factor", "prefetch_factor"),
    ):
        value = getattr(cli, attr)
        if value is not None and value > 0:
            args[flag] = value
        elif flag in runtime:
            args[flag] = runtime[flag]
    return args


def normal_ratio_token(args: dict[str, Any]) -> str:
    try:
        return f"n{int(float(args.get('--normal_ratio', 700)))}"
    except (TypeError, ValueError):
        return "ncustom"


def make_runs(
    scope: str,
    cells: list[Cell],
    selection: dict[str, Selection],
    runtimes: dict[Cell, dict[str, Any]],
    seeds: list[int],
    cli: argparse.Namespace,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for cell in cells:
        args = apply_selection(build_base_args(cell, runtimes.get(cell, {}), cli), selection)
        dataset = short_name(cell.config)
        backbone = short_name(cell.model)
        candidate = f"fresh0412_v11_bkm_{scope}_{dataset}_{backbone}_{normal_ratio_token(args)}"
        for seed in seeds:
            runs.append(
                {
                    "tag": f"{candidate}_s{seed}",
                    "candidate": candidate,
                    "seed": seed,
                    "args": dict(args),
                    "reason": f"{scope} consensus BKM: axis choices selected by mean F1 over source cells",
                }
            )
    return runs


def write_selection_csv(path: Path, label: str, selection: dict[str, Selection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scope", "axis", "candidate", "values", "mean_f1", "mean_fn", "mean_fp", "cells"],
        )
        writer.writeheader()
        for axis, entry in selection.items():
            writer.writerow(
                {
                    "scope": label,
                    "axis": axis,
                    "candidate": entry.candidate,
                    "values": json.dumps(entry.values, ensure_ascii=False, sort_keys=True),
                    "mean_f1": f"{entry.mean_f1:.6f}",
                    "mean_fn": f"{entry.mean_fn:.4f}",
                    "mean_fp": f"{entry.mean_fp:.4f}",
                    "cells": entry.cells,
                }
            )


def append_selection_md(lines: list[str], title: str, selection: dict[str, Selection]) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| axis | selected candidate | values | mean F1 | mean FN | mean FP | cells |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for axis, entry in selection.items():
        values = ", ".join(f"{k}={v}" for k, v in entry.values.items())
        lines.append(
            f"| {axis} | `{entry.candidate}` | `{values}` | "
            f"{entry.mean_f1:.4f} | {entry.mean_fn:.2f} | {entry.mean_fp:.2f} | {entry.cells} |"
        )
    lines.append("")


def run_controller(
    queue_path: Path,
    summary_path: Path,
    markdown_path: Path,
    log_group: str,
    cli: argparse.Namespace,
) -> None:
    cmd = [
        cli.python,
        "scripts/adaptive_experiment_controller.py",
        "--queue",
        str(queue_path),
        "--summary",
        str(summary_path),
        "--markdown",
        str(markdown_path),
        "--target-min",
        "5",
        "--target-max",
        "15",
        "--stop-mode",
        "never",
        "--candidate-min-runs-before-skip",
        "0",
        "--completion-exit-grace",
        "15",
        "--update-live-summary",
        "--checkpoint-retention",
        cli.checkpoint_retention,
        "--checkpoint-retention-scope",
        cli.checkpoint_retention_scope,
        "--log-dir-group",
        log_group,
    ]
    if cli.force:
        cmd.append("--force")
    if cli.max_launched > 0:
        cmd.extend(["--max-launched", str(cli.max_launched)])
    if cli.dry_run:
        cmd.append("--dry-run")

    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def write_queue(path: Path, scope: str, selection: dict[str, Selection], runs: list[dict[str, Any]]) -> None:
    payload = {
        "created_by": "scripts/run_consensus_bkm.py",
        "scope": scope,
        "note": "Consensus BKM values selected from completed baseline and one-factor sweep results.",
        "selection": {
            axis: {
                "candidate": entry.candidate,
                "values": entry.values,
                "mean_f1": entry.mean_f1,
                "mean_fn": entry.mean_fn,
                "mean_fp": entry.mean_fp,
                "cells": entry.cells,
            }
            for axis, entry in selection.items()
        },
        "runs": runs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[consensus-bkm] wrote {path} runs={len(runs)}", flush=True)


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("no seeds provided")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--validations-root", type=Path, default=Path("validations"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-root-group", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--max-launched", type=int, default=0)
    parser.add_argument("--checkpoint-retention", choices=["all", "dataset-backbone-best"], default="dataset-backbone-best")
    parser.add_argument("--checkpoint-retention-scope", choices=["summary", "log-group", "logs"], default="log-group")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--scope",
        choices=["dataset", "global", "both"],
        default="both",
        help="Which consensus BKM queues to run.",
    )
    args = parser.parse_args()

    cells = parse_groups(args.groups)
    seeds = parse_seeds(args.seeds)
    validations_root = args.validations_root
    if not validations_root.is_absolute():
        validations_root = ROOT / validations_root
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_scores: list[Score] = []
    runtimes: dict[Cell, dict[str, Any]] = {}
    for cell in cells:
        scores, runtime = load_cell_scores(cell, validations_root, require_complete=not args.allow_partial)
        all_scores.extend(scores)
        runtimes[cell] = runtime

    if not all_scores:
        print("[consensus-bkm] no source scores; nothing to run", flush=True)
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_lines = [
        "# Consensus BKM",
        "",
        f"- Source cells: {len(cells)}",
        f"- Seeds per target condition: {','.join(str(s) for s in seeds)}",
        f"- Checkpoint retention: {args.checkpoint_retention} / {args.checkpoint_retention_scope}",
        "",
    ]

    groups_by_config: dict[str, list[Cell]] = defaultdict(list)
    for cell in cells:
        groups_by_config[cell.config].append(cell)

    if args.scope in {"dataset", "both"}:
        for config, dataset_cells in sorted(groups_by_config.items()):
            dataset_scores = [score for score in all_scores if score.cell in dataset_cells]
            selection = select_consensus(dataset_scores, total_cells=len(dataset_cells))
            label = f"dataset_{short_name(config)}"
            append_selection_md(md_lines, label, selection)
            ds_dir = out_dir / label
            write_selection_csv(ds_dir / "bkm_selection.csv", label, selection)
            runs = make_runs(label, dataset_cells, selection, runtimes, seeds, args)
            queue = ds_dir / "bkm_queue.json"
            summary = ds_dir / "bkm_results.json"
            markdown = ds_dir / "bkm_results.md"
            write_queue(queue, label, selection, runs)
            log_group = f"{args.log_root_group}/{ts}_bkm_{label}"
            run_controller(queue, summary, markdown, log_group, args)

    if args.scope in {"global", "both"}:
        selection = select_consensus(all_scores, total_cells=len(cells))
        label = "global"
        append_selection_md(md_lines, label, selection)
        gl_dir = out_dir / label
        write_selection_csv(gl_dir / "bkm_selection.csv", label, selection)
        runs = make_runs(label, cells, selection, runtimes, seeds, args)
        queue = gl_dir / "bkm_queue.json"
        summary = gl_dir / "bkm_results.json"
        markdown = gl_dir / "bkm_results.md"
        write_queue(queue, label, selection, runs)
        log_group = f"{args.log_root_group}/{ts}_bkm_{label}"
        run_controller(queue, summary, markdown, log_group, args)

    (out_dir / "consensus_bkm_selection.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[consensus-bkm] selection report: {out_dir / 'consensus_bkm_selection.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
