#!/usr/bin/env python
"""Two-stage binary gate + defect-type prediction.

Stage 1 loads a binary run and decides pass/fail. Stage 2 loads an
``anomaly_type`` run and classifies only samples that Stage 1 sends to
``abnormal``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage binary + defect-type prediction.")
    parser.add_argument(
        "--binary-model-run",
        required=True,
        help="Stage-1 binary run dir containing best_model.pth and best_info.json, or direct best_model.pth path.",
    )
    parser.add_argument(
        "--type-model-run",
        required=True,
        help="Stage-2 anomaly_type/multiclass run dir containing best_model.pth and best_info.json.",
    )
    parser.add_argument("--dataset-dir", default="data", help="Directory containing timeseries.csv and scenarios.csv.")
    parser.add_argument("--scenarios-file", default=None, help="Optional scenarios CSV. Defaults to dataset-dir/scenarios.csv.")
    parser.add_argument("--config", default="dataset.yaml", help="Renderer config path.")
    parser.add_argument("--output-dir", default="two_stage_output", help="Output directory.")
    parser.add_argument("--split", default=None, help="Optional split filter: train|val|test.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of scenario rows to process.")
    parser.add_argument(
        "--normal-threshold",
        type=float,
        default=0.9,
        help="Stage-1 pass threshold. A sample is normal only when p_normal > threshold.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Inference device. Default is cpu for reproducible smoke tests.",
    )
    parser.add_argument("--chunksize", type=int, default=200_000, help="timeseries.csv chunk size.")
    parser.add_argument("--save-display", action="store_true", help="Save display images grouped by final prediction.")
    parser.add_argument("--save-model-inputs", action="store_true", help="Persist exact model input images.")
    return parser.parse_args()


def resolve_run(path_text: str) -> Path:
    path = Path(path_text)
    run_dir = path.parent if path.is_file() else path
    if not (run_dir / "best_model.pth").exists():
        raise FileNotFoundError(f"best_model.pth not found under run dir: {run_dir}")
    if not (run_dir / "best_info.json").exists():
        raise FileNotFoundError(f"best_info.json not found under run dir: {run_dir}")
    return run_dir


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


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


def truth_class(row: pd.Series) -> str | None:
    raw = row.get("true_class", row.get("class"))
    if pd.isna(raw):
        return None
    return str(raw).strip()


def truth_binary(row: pd.Series) -> str | None:
    raw = truth_class(row)
    if raw is None:
        return None
    return "normal" if raw.lower() == "normal" else "abnormal"


def safe_rate(num: float, denom: float) -> float | None:
    if denom == 0:
        return None
    return float(num / denom)


def read_scenarios(path: Path, split: str | None, limit: int | None) -> pd.DataFrame:
    sc_df = pd.read_csv(path)
    if split:
        if "split" not in sc_df.columns:
            raise ValueError("--split was provided but scenarios file has no split column.")
        sc_df = sc_df[sc_df["split"] == split].reset_index(drop=True)
    if limit is not None:
        sc_df = sc_df.head(limit).reset_index(drop=True)
    if "chart_id" not in sc_df.columns:
        raise ValueError("scenarios file must contain chart_id.")
    return sc_df


def load_selected_timeseries(path: Path, chart_ids: set[str], chunksize: int) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if "chart_id" not in chunk.columns:
            raise ValueError("timeseries.csv must contain chart_id.")
        mask = chunk["chart_id"].astype(str).isin(chart_ids)
        if mask.any():
            chunks.append(chunk.loc[mask].copy())
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def build_fleet_data(row: pd.Series, chart_ts: pd.DataFrame, x_col: str) -> tuple[dict[str, tuple], str, str]:
    legend_axis = read_legend_axis(row)
    highlighted = read_highlighted_member(row)
    members = read_members(row, chart_ts, legend_axis)
    if not legend_axis or not highlighted or not members:
        return {}, legend_axis, highlighted or ""

    fleet_data: dict[str, tuple] = {}
    for member_id in members:
        member_ts = chart_ts[chart_ts[legend_axis].astype(str) == str(member_id)].sort_values(x_col)
        if member_ts.empty:
            continue
        fleet_data[str(member_id)] = (member_ts[x_col].to_numpy(), member_ts["value"].to_numpy())
    return fleet_data, legend_axis, highlighted


def predict_probs(model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(tensor)
        return F.softmax(logits, dim=1)[0].detach().cpu()


def render_display_image(
    renderer: ImageRenderer,
    fleet_data: dict[str, tuple],
    highlighted_member: str,
    display_path: Path,
    row: pd.Series,
    x_col: str,
    title_columns: list[str],
) -> None:
    true_label = truth_binary(row)
    anomalous_ids = [highlighted_member] if true_label == "abnormal" else []
    defect_start_idx = row.get("defect_start_idx") if true_label == "abnormal" else None
    title_parts = [str(row.get(col, "")) for col in title_columns if pd.notna(row.get(col))]
    title = " / ".join(title_parts) if title_parts else str(row.get("chart_id", "chart"))

    renderer.render_overlay_display(
        fleet_data,
        highlighted_member,
        str(display_path),
        anomalous_ids=anomalous_ids,
        defect_start_idx=defect_start_idx,
        title=title,
        x_label=x_col,
        target=read_target(row),
    )


def bucket_for(true_label: str | None, pred_label: str) -> str:
    if true_label == "normal" and pred_label == "abnormal":
        return "fp_normal"
    if true_label == "abnormal" and pred_label == "normal":
        return "fn_abnormal"
    if true_label == "abnormal" and pred_label == "abnormal":
        return "tp_abnormal"
    if true_label == "normal" and pred_label == "normal":
        return "tn_normal"
    return ""


def make_summary(results_df: pd.DataFrame, binary_classes: list[str], type_classes: list[str], args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "processed": int(len(results_df)),
        "binary_model_run": str(resolve_run(args.binary_model_run)),
        "type_model_run": str(resolve_run(args.type_model_run)),
        "dataset_dir": str(Path(args.dataset_dir)),
        "scenarios_file": str(Path(args.scenarios_file) if args.scenarios_file else Path(args.dataset_dir) / "scenarios.csv"),
        "split": args.split,
        "limit": args.limit,
        "normal_threshold": args.normal_threshold,
        "binary_classes": binary_classes,
        "type_classes": type_classes,
        "type_model_contains_normal": "normal" in type_classes,
    }
    if results_df.empty:
        return summary

    pred_counts = Counter(results_df["binary_pred"])
    bucket_counts = Counter(results_df["bucket"])
    summary.update(
        {
            "stage1_predicted_normal": int(pred_counts.get("normal", 0)),
            "stage1_predicted_abnormal": int(pred_counts.get("abnormal", 0)),
            "stage2_evaluated": int(results_df["stage2_ran"].sum()),
            "tn": int(bucket_counts.get("tn_normal", 0)),
            "fn": int(bucket_counts.get("fn_abnormal", 0)),
            "fp": int(bucket_counts.get("fp_normal", 0)),
            "tp": int(bucket_counts.get("tp_abnormal", 0)),
        }
    )

    tn = summary["tn"]
    fn = summary["fn"]
    fp = summary["fp"]
    tp = summary["tp"]
    total = tn + fn + fp + tp
    if total:
        precision = safe_rate(tp, tp + fp)
        recall = safe_rate(tp, tp + fn)
        summary.update(
            {
                "binary_accuracy": safe_rate(tp + tn, total),
                "abnormal_recall": recall,
                "normal_recall": safe_rate(tn, tn + fp),
                "precision": precision,
                "f1": None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall),
            }
        )

    if "true_class" in results_df.columns:
        stage2_eval = results_df[
            (results_df["stage2_ran"])
            & (results_df["true_binary"] == "abnormal")
            & (results_df["true_class"].isin(type_classes))
        ]
        if not stage2_eval.empty:
            summary["stage2_type_accuracy_on_binary_tp"] = float(
                (stage2_eval["stage2_pred"] == stage2_eval["true_class"]).mean()
            )
            summary["stage2_type_eval_count"] = int(len(stage2_eval))

    return summary


def run_two_stage(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    binary_run = resolve_run(args.binary_model_run)
    type_run = resolve_run(args.type_model_run)

    binary_model, binary_classes = _load_model_from_best_info(binary_run, device)
    type_model, type_classes = _load_model_from_best_info(type_run, device)
    if "normal" not in binary_classes:
        raise ValueError("Stage-1 model classes must include normal.")
    if "abnormal" not in binary_classes:
        raise ValueError("Stage-1 model classes must include abnormal.")

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    renderer = ImageRenderer(config)
    title_columns = config.get("image", {}).get("title_columns", ["device", "step", "item"])

    dataset_dir = Path(args.dataset_dir)
    scenarios_path = Path(args.scenarios_file) if args.scenarios_file else dataset_dir / "scenarios.csv"
    sc_df = read_scenarios(scenarios_path, args.split, args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if sc_df.empty:
        results_df = pd.DataFrame()
        results_df.to_csv(output_dir / "two_stage_predictions.csv", index=False)
        summary = make_summary(results_df, binary_classes, type_classes, args)
        with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print("[two_stage_predict] rows=0")
        print(f"  csv: {output_dir / 'two_stage_predictions.csv'}")
        print(f"  summary: {output_dir / 'summary.json'}")
        return summary

    selected_ids = {str(chart_id) for chart_id in sc_df["chart_id"].tolist()}
    ts_df = load_selected_timeseries(dataset_dir / "timeseries.csv", selected_ids, args.chunksize)
    if ts_df.empty and not sc_df.empty:
        raise ValueError("No matching timeseries rows found for selected scenarios.")
    x_col = _detect_x_col(ts_df)
    ts_grouped = {str(chart_id): grp for chart_id, grp in ts_df.groupby("chart_id")}

    if args.save_display:
        (output_dir / "display").mkdir(parents=True, exist_ok=True)
    if args.save_model_inputs:
        (output_dir / "model_inputs").mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    normal_idx = binary_classes.index("normal")
    abnormal_idx = binary_classes.index("abnormal")
    temp_dir = Path(tempfile.mkdtemp(prefix="two_stage_pred_"))
    results: list[dict[str, Any]] = []

    try:
        for idx, row in tqdm(sc_df.iterrows(), total=len(sc_df), desc="two-stage"):
            row = row.copy()
            chart_id = str(row["chart_id"])
            chart_ts = ts_grouped.get(chart_id)
            if chart_ts is None:
                continue
            fleet_data, legend_axis, highlighted = build_fleet_data(row, chart_ts, x_col)
            if not fleet_data or highlighted not in fleet_data:
                continue

            member_x = fleet_data[highlighted][0]
            timestamp = _format_timestamp(member_x[0] if len(member_x) else "na")
            input_name = f"{sanitize_part(chart_id)}_{sanitize_part(highlighted)}.png"
            input_path = temp_dir / input_name
            renderer.render_overlay(fleet_data, highlighted, str(input_path), target=read_target(row))

            image = Image.open(input_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            binary_probs = predict_probs(binary_model, tensor)
            p_normal = float(binary_probs[normal_idx].item())
            p_abnormal = float(binary_probs[abnormal_idx].item())
            binary_pred = "normal" if p_normal > args.normal_threshold else "abnormal"

            stage2_ran = binary_pred == "abnormal"
            stage2_pred = ""
            stage2_confidence: float | None = None
            type_probs: dict[str, float] = {}
            if stage2_ran:
                probs = predict_probs(type_model, tensor)
                type_idx = int(torch.argmax(probs).item())
                stage2_pred = type_classes[type_idx] if type_idx < len(type_classes) else str(type_idx)
                stage2_confidence = float(probs[type_idx].item())
                type_probs = {
                    f"stage2_p_{sanitize_part(class_name)}": float(probs[class_idx].item())
                    for class_idx, class_name in enumerate(type_classes)
                }

            true_cls = truth_class(row)
            true_bin = truth_binary(row)
            bucket = bucket_for(true_bin, binary_pred)
            final_pred = "normal" if binary_pred == "normal" else f"abnormal/{stage2_pred or 'untyped'}"

            result = row.to_dict()
            result.update(
                {
                    "legend_axis": legend_axis,
                    "highlighted_member": highlighted,
                    "target": read_target(row),
                    "timestamp": timestamp,
                    "true_class": true_cls,
                    "true_binary": true_bin,
                    "p_normal": p_normal,
                    "p_abnormal": p_abnormal,
                    "binary_pred": binary_pred,
                    "stage2_ran": stage2_ran,
                    "stage2_pred": stage2_pred,
                    "stage2_confidence": stage2_confidence,
                    "final_pred": final_pred,
                    "bucket": bucket,
                }
            )
            result.update(type_probs)

            stem = f"{idx:05d}_{sanitize_part(chart_id)}_{sanitize_part(highlighted)}_{sanitize_part(final_pred)}"
            if args.save_model_inputs:
                model_input_dir = output_dir / "model_inputs" / binary_pred
                model_input_dir.mkdir(parents=True, exist_ok=True)
                saved_input = model_input_dir / f"{stem}.png"
                shutil.copy2(input_path, saved_input)
                result["model_input_image"] = str(saved_input.relative_to(output_dir)).replace("\\", "/")
            if args.save_display:
                display_dir = output_dir / "display" / sanitize_part(final_pred)
                display_dir.mkdir(parents=True, exist_ok=True)
                display_path = display_dir / f"{stem}.png"
                render_display_image(renderer, fleet_data, highlighted, display_path, row, x_col, title_columns)
                result["display_image"] = str(display_path.relative_to(output_dir)).replace("\\", "/")

            results.append(result)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        sort_cols = [col for col in ["p_abnormal", "chart_id"] if col in results_df.columns]
        if sort_cols:
            results_df = results_df.sort_values(sort_cols, ascending=[False, True]).reset_index(drop=True)
    results_df.to_csv(output_dir / "two_stage_predictions.csv", index=False)

    summary = make_summary(results_df, binary_classes, type_classes, args)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"[two_stage_predict] rows={summary.get('processed', 0)}")
    print(f"  csv: {output_dir / 'two_stage_predictions.csv'}")
    print(f"  summary: {output_dir / 'summary.json'}")
    return summary


def main() -> int:
    args = parse_args()
    run_two_stage(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
