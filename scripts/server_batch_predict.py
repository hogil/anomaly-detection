"""Server batch prediction for anomaly datasets.

Scans one or more roots for dataset directories containing:
  - timeseries.csv
  - scenarios.csv or scenarios_per_member.csv

For each dataset directory, this script:
  1. renders model-input images
  2. loads a trained best model from a run directory
  3. runs prediction
  4. saves predicted display images
  5. if ground truth exists, saves FP/FN/TP/TN image buckets
  6. writes per-dataset CSV/JSON summaries and a global summary

Typical usage:
  python scripts/server_batch_predict.py ^
    --model-run logs\\260427_065155_fresh0412_v11_regls015_n700_s1_F0.9987_R0.9987 ^
    --input-root D:\\datasets\\products ^
    --output-root server_inference ^
    --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import _detect_x_col, _format_timestamp, _load_model_from_best_info  # noqa: E402
from src.data.image_renderer import ImageRenderer  # noqa: E402
from src.data.schema import highlighted_member as read_highlighted_member  # noqa: E402
from src.data.schema import legend_axis as read_legend_axis  # noqa: E402
from src.data.schema import members as read_members  # noqa: E402
from src.data.schema import target as read_target  # noqa: E402


SKIP_SCAN_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "logs",
    "weights",
    "validations",
    "preview",
    "experiments",
    "server_inference",
    "server_outputs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch render + predict + FP/FN export for dataset folders.")
    parser.add_argument(
        "--model-run",
        required=True,
        help="Run directory containing best_model.pth and best_info.json, or a direct best_model.pth path.",
    )
    parser.add_argument(
        "--input-root",
        action="append",
        default=[],
        help="Root to recursively scan for dataset folders. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=[],
        help="Specific dataset directory containing timeseries.csv and scenarios*.csv. Can be passed multiple times.",
    )
    parser.add_argument("--config", default="dataset.yaml", help="Renderer config.")
    parser.add_argument("--output-root", default="server_inference", help="Output root for all datasets.")
    parser.add_argument(
        "--prefer-per-member",
        action="store_true",
        help="Prefer scenarios_per_member.csv over scenarios.csv when both exist.",
    )
    parser.add_argument("--split", default=None, help="Optional split filter: train|val|test")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit per dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing dataset output directory first.")
    parser.add_argument("--dry-run", action="store_true", help="Only list discovered dataset folders.")
    parser.add_argument(
        "--save-model-inputs",
        action="store_true",
        help="Persist exact overlay images used as model inputs under output/model_inputs.",
    )
    return parser.parse_args()


def resolve_model_run(model_run_arg: str) -> Path:
    model_path = Path(model_run_arg)
    run_dir = model_path.parent if model_path.is_file() else model_path
    if not (run_dir / "best_model.pth").exists():
        raise FileNotFoundError(f"best_model.pth not found under run dir: {run_dir}")
    if not (run_dir / "best_info.json").exists():
        raise FileNotFoundError(f"best_info.json not found under run dir: {run_dir}")
    return run_dir


def sanitize_part(value: object) -> str:
    text = str(value).strip()
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("-")
    cleaned = "".join(safe).strip("-_")
    return cleaned or "na"


def true_binary_label(row: pd.Series) -> str | None:
    raw = row.get("true_class", row.get("class"))
    if pd.isna(raw):
        return None
    return "normal" if str(raw).strip().lower() == "normal" else "abnormal"


def choose_scenarios_file(dataset_dir: Path, prefer_per_member: bool) -> Path | None:
    per_member = dataset_dir / "scenarios_per_member.csv"
    standard = dataset_dir / "scenarios.csv"
    ordered = [per_member, standard] if prefer_per_member else [standard, per_member]
    for candidate in ordered:
        if candidate.exists():
            return candidate
    return None


def discover_dataset_dirs(input_roots: Iterable[Path], explicit_dirs: Iterable[Path], prefer_per_member: bool) -> list[tuple[Path, Path]]:
    found: dict[str, tuple[Path, Path]] = {}

    for dataset_dir in explicit_dirs:
        ts_path = dataset_dir / "timeseries.csv"
        sc_path = choose_scenarios_file(dataset_dir, prefer_per_member)
        if ts_path.exists() and sc_path:
            found[str(dataset_dir.resolve()).lower()] = (dataset_dir.resolve(), sc_path.resolve())

    for root in input_roots:
        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in SKIP_SCAN_NAMES and not d.startswith("images") and not d.startswith("display")
            ]
            if "timeseries.csv" not in filenames:
                continue
            dataset_dir = Path(current_root)
            sc_path = choose_scenarios_file(dataset_dir, prefer_per_member)
            if not sc_path:
                continue
            found[str(dataset_dir.resolve()).lower()] = (dataset_dir.resolve(), sc_path.resolve())

    return sorted(found.values(), key=lambda item: str(item[0]).lower())


def build_output_stem(row: pd.Series, title_columns: list[str], p_abnormal: float) -> str:
    highlighted_member = sanitize_part(read_highlighted_member(row) or "member")
    chart_id = sanitize_part(row.get("chart_id", "chart"))
    title_parts = [sanitize_part(row.get(col, "na")) for col in title_columns]
    ts_val = row.get("_timestamp_string", "na")
    p_part = f"p{int(round(p_abnormal * 100)):03d}"
    return "_".join([p_part, *title_parts, chart_id, highlighted_member, sanitize_part(ts_val)])


def render_display_image(
    renderer: ImageRenderer,
    fleet_data: dict[str, tuple],
    highlighted_member: str,
    display_path: Path,
    row: pd.Series,
    x_col: str,
    title_columns: list[str],
) -> None:
    true_label = true_binary_label(row)
    anomalous_ids = [highlighted_member] if true_label == "abnormal" else []
    defect_start_idx = row.get("defect_start_idx")
    if true_label != "abnormal":
        defect_start_idx = None

    title_parts = [str(row.get(col, "")) for col in title_columns if pd.notna(row.get(col))]
    title = " / ".join(title_parts) if title_parts else str(row.get("chart_id", "chart"))

    target = read_target(row)

    renderer.render_overlay_display(
        fleet_data,
        highlighted_member,
        str(display_path),
        anomalous_ids=anomalous_ids,
        defect_start_idx=defect_start_idx,
        title=title,
        x_label=x_col,
        target=target,
    )


def render_model_input(renderer: ImageRenderer, fleet_data: dict[str, tuple], highlighted_member: str, row: pd.Series, path: Path) -> None:
    renderer.render_overlay(fleet_data, highlighted_member, str(path), target=read_target(row))


def infer_dataset(
    dataset_dir: Path,
    scenarios_path: Path,
    output_dir: Path,
    model,
    classes: list[str],
    renderer: ImageRenderer,
    title_columns: list[str],
    split_filter: str | None,
    limit: int | None,
    device: torch.device,
    save_model_inputs: bool,
) -> dict[str, object]:
    ts_df = pd.read_csv(dataset_dir / "timeseries.csv")
    sc_df = pd.read_csv(scenarios_path)
    x_col = _detect_x_col(ts_df)

    if split_filter:
        sc_df = sc_df[sc_df["split"] == split_filter].reset_index(drop=True)
    if limit:
        sc_df = sc_df.head(limit).reset_index(drop=True)

    if sc_df.empty:
        return {
            "dataset": dataset_dir.name,
            "dataset_dir": str(dataset_dir),
            "scenarios_file": str(scenarios_path),
            "processed": 0,
        }

    ts_grouped = {sid: grp for sid, grp in ts_df.groupby("chart_id")}

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_root = output_dir / "predictions"
    for subdir in ["normal", "abnormal", "fp_normal", "fn_abnormal", "tp_abnormal", "tn_normal"]:
        (pred_root / subdir).mkdir(parents=True, exist_ok=True)
    if save_model_inputs:
        (output_dir / "model_inputs").mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    temp_dir = Path(tempfile.mkdtemp(prefix=f"server_pred_{dataset_dir.name}_"))
    results: list[dict[str, object]] = []

    try:
        for _, row in tqdm(sc_df.iterrows(), total=len(sc_df), desc=f"predict:{dataset_dir.name}"):
            row = row.copy()
            sid = row["chart_id"]
            highlighted_member = read_highlighted_member(row)
            legend_axis = read_legend_axis(row)
            members = read_members(row)
            sc_ts = ts_grouped.get(sid)
            if sc_ts is None or not highlighted_member or not legend_axis or not members:
                continue

            fleet_data: dict[str, tuple] = {}
            for member_id in members:
                member_ts = sc_ts[sc_ts[legend_axis].astype(str) == str(member_id)].sort_values(x_col)
                if member_ts.empty:
                    continue
                fleet_data[member_id] = (member_ts[x_col].to_numpy(), member_ts["value"].to_numpy())

            if highlighted_member not in fleet_data:
                continue

            member_x = fleet_data[highlighted_member][0]
            ts_string = _format_timestamp(member_x[0] if len(member_x) else "na")
            row["_timestamp_string"] = ts_string

            input_name = f"{sanitize_part(sid)}_{sanitize_part(highlighted_member)}.png"
            temp_input = temp_dir / input_name
            render_model_input(renderer, fleet_data, highlighted_member, row, temp_input)

            img = Image.open(temp_input).convert("RGB")
            x_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x_tensor)
                probs = F.softmax(logits, dim=1)[0]
                pred_idx = int(torch.argmax(probs).item())

            pred_class = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
            try:
                abn_idx = classes.index("abnormal")
            except ValueError:
                abn_idx = min(1, len(classes) - 1)
            p_abnormal = float(probs[abn_idx].item())

            stem = build_output_stem(row, title_columns, p_abnormal)
            pred_path = pred_root / pred_class / f"{stem}.png"
            render_display_image(renderer, fleet_data, highlighted_member, pred_path, row, x_col, title_columns)

            if save_model_inputs:
                input_dir = output_dir / "model_inputs" / (row.get("split") or "all")
                input_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(temp_input, input_dir / f"{stem}.png")

            truth = true_binary_label(row)
            bucket = None
            if truth == "normal" and pred_class == "abnormal":
                bucket = "fp_normal"
            elif truth == "abnormal" and pred_class == "normal":
                bucket = "fn_abnormal"
            elif truth == "abnormal" and pred_class == "abnormal":
                bucket = "tp_abnormal"
            elif truth == "normal" and pred_class == "normal":
                bucket = "tn_normal"

            if bucket:
                shutil.copy2(pred_path, pred_root / bucket / pred_path.name)

            result = row.to_dict()
            result.update(
                {
                    "dataset_dir": str(dataset_dir),
                    "scenarios_file": str(scenarios_path),
                    "legend_axis": legend_axis,
                    "highlighted_member": highlighted_member,
                    "target": read_target(row),
                    "predicted": pred_class,
                    "true_binary": truth,
                    "p_abnormal": round(p_abnormal, 6),
                    "prediction_image": str(pred_path.relative_to(output_dir)).replace("\\", "/"),
                    "bucket": bucket or "",
                }
            )
            results.append(result)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["p_abnormal", "chart_id"], ascending=[False, True]).reset_index(drop=True)
    results_df.to_csv(output_dir / "predictions.csv", index=False)

    if "bucket" in results_df.columns:
        for bucket_name, filename in [("fp_normal", "fp_list.csv"), ("fn_abnormal", "fn_list.csv"), ("tp_abnormal", "tp_list.csv"), ("tn_normal", "tn_list.csv")]:
            results_df[results_df["bucket"] == bucket_name].to_csv(output_dir / filename, index=False)

    bucket_counts = Counter(results_df["bucket"]) if not results_df.empty else Counter()
    pred_counts = Counter(results_df["predicted"]) if not results_df.empty else Counter()
    summary = {
        "dataset": dataset_dir.name,
        "dataset_dir": str(dataset_dir),
        "scenarios_file": str(scenarios_path),
        "processed": int(len(results_df)),
        "predicted_normal": int(pred_counts.get("normal", 0)),
        "predicted_abnormal": int(pred_counts.get("abnormal", 0)),
        "fp_normal": int(bucket_counts.get("fp_normal", 0)),
        "fn_abnormal": int(bucket_counts.get("fn_abnormal", 0)),
        "tp_abnormal": int(bucket_counts.get("tp_abnormal", 0)),
        "tn_normal": int(bucket_counts.get("tn_normal", 0)),
    }
    if "true_binary" in results_df.columns and results_df["true_binary"].notna().any():
        valid_truth = results_df["true_binary"].notna()
        summary["accuracy"] = round(float((results_df.loc[valid_truth, "predicted"] == results_df.loc[valid_truth, "true_binary"]).mean()), 6)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    renderer = ImageRenderer(config)
    title_columns = config.get("image", {}).get("title_columns", ["device", "step", "item"])

    input_roots = [Path(p).resolve() for p in args.input_root]
    explicit_dirs = [Path(p).resolve() for p in args.dataset_dir]
    if not input_roots and not explicit_dirs:
        input_roots = [ROOT]

    discovered = discover_dataset_dirs(input_roots, explicit_dirs, args.prefer_per_member)
    if not discovered:
        raise FileNotFoundError("No dataset directories found. Expected timeseries.csv + scenarios*.csv")

    print("Discovered datasets:")
    for dataset_dir, scenarios_path in discovered:
        print(f"  - {dataset_dir}  [{scenarios_path.name}]")
    if args.dry_run:
        return 0

    run_dir = resolve_model_run(args.model_run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = _load_model_from_best_info(run_dir, device)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []

    for dataset_dir, scenarios_path in discovered:
        try:
            relative = None
            for root in input_roots:
                try:
                    relative = dataset_dir.relative_to(root)
                    break
                except ValueError:
                    continue
            dataset_out = output_root / (str(relative).replace("\\", "__").replace("/", "__") if relative else dataset_dir.name)
            if dataset_out.exists() and args.overwrite:
                shutil.rmtree(dataset_out)
            dataset_out.mkdir(parents=True, exist_ok=True)

            summary = infer_dataset(
                dataset_dir=dataset_dir,
                scenarios_path=scenarios_path,
                output_dir=dataset_out,
                model=model,
                classes=classes,
                renderer=renderer,
                title_columns=title_columns,
                split_filter=args.split,
                limit=args.limit,
                device=device,
                save_model_inputs=args.save_model_inputs,
            )
            summaries.append(summary)
            print(f"[done] {dataset_dir.name}: {summary}")
        except Exception as exc:
            error_summary = {
                "dataset": dataset_dir.name,
                "dataset_dir": str(dataset_dir),
                "scenarios_file": str(scenarios_path),
                "error": str(exc),
            }
            summaries.append(error_summary)
            print(f"[error] {dataset_dir}: {exc}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_root / "summary.csv", index=False)
    with open(output_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False)
    print(f"Global summary written to {output_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
