#!/usr/bin/env python
"""Generate Grad-CAM overlays for false-positive and false-negative samples."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gradcam_report import (
    GradCam,
    build_transform,
    heat_stats,
    load_font,
    resolve_layer,
    save_cam_on_image,
    save_heat_only_cam,
)
from inference import _load_model_from_best_info


def load_items(scenarios_csv: Path, image_dir: Path, split: str) -> list[dict[str, str]]:
    scenarios = pd.read_csv(scenarios_csv)
    if split:
        scenarios = scenarios[scenarios["split"].astype(str) == split].reset_index(drop=True)

    items: list[dict[str, str]] = []
    for row in scenarios.to_dict(orient="records"):
        chart_id = str(row["chart_id"])
        true_class = str(row["class"])
        split_name = str(row["split"])
        image_path = image_dir / split_name / true_class / f"{chart_id}.png"
        if image_path.exists():
            items.append(
                {
                    "chart_id": chart_id,
                    "true_class": true_class,
                    "true_binary": "normal" if true_class == "normal" else "abnormal",
                    "image_path": str(image_path),
                }
            )
    if not items:
        raise SystemExit("no image items found")
    return items


def binary_pred(probs: torch.Tensor, classes: list[str]) -> tuple[str, float]:
    pred_idx = int(torch.argmax(probs).item())
    pred = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
    abn_idx = classes.index("abnormal") if "abnormal" in classes else min(1, len(classes) - 1)
    return pred, float(probs[abn_idx].item())


def matches_error(row: dict[str, Any], error_type: str) -> bool:
    is_fp = row["true_binary"] == "normal" and row["pred_class"] == "abnormal"
    is_fn = row["true_binary"] == "abnormal" and row["pred_class"] == "normal"
    if error_type == "fp":
        return is_fp
    if error_type == "fn":
        return is_fn
    return is_fp or is_fn


def write_gallery(rows: list[dict[str, Any]], gallery_out: Path) -> None:
    if not rows:
        return
    cell_w, image_h, title_h = 224, 224, 56
    cell_h = image_h + title_h
    pad = 14
    cols = min(6, len(rows))
    rows_n = (len(rows) + cols - 1) // cols
    canvas = Image.new("RGBA", (cols * cell_w + (cols + 1) * pad, rows_n * cell_h + (rows_n + 1) * pad), "WHITE")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    meta_font = load_font(14)
    for idx, row in enumerate(rows):
        x = pad + (idx % cols) * (cell_w + pad)
        y = pad + (idx // cols) * (cell_h + pad)
        label = str(row["error_type"]).upper().replace("_", " ")
        label_bbox = draw.textbbox((0, 0), label, font=title_font)
        label_x = x + max(0, (cell_w - (label_bbox[2] - label_bbox[0])) // 2)
        draw.text((label_x, y), label, fill=(0, 0, 0, 255), font=title_font)
        meta = f"p_abn={float(row['p_abnormal']):.4f}"
        meta_bbox = draw.textbbox((0, 0), meta, font=meta_font)
        meta_x = x + max(0, (cell_w - (meta_bbox[2] - meta_bbox[0])) // 2)
        draw.text((meta_x, y + 34), meta, fill=(50, 50, 50, 255), font=meta_font)
        overlay = Image.open(str(row["cam_overlay"])).convert("RGBA").resize((224, 224))
        canvas.alpha_composite(overlay, (x, y + title_h))
    gallery_out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(gallery_out)


def format_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    actual_errors = sum(1 for row in rows if row["error_type"] in {"fp", "fn"})
    hard_normal = sum(1 for row in rows if row["error_type"] == "hard_normal")
    lines = [
        "# Grad-CAM Error Report",
        "",
        f"- model_run: `{args.model_run}`",
        f"- split: `{args.split}`",
        f"- error_type: `{args.error_type}`",
        f"- selected: `{len(rows)}`",
        f"- actual_errors: `{actual_errors}`",
        f"- hard_normal_fill: `{hard_normal}`",
        "",
        "| sample | chart_id | true | pred | p_abnormal | left | mid | right | peak_x | cam |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['error_type']} | {row['chart_id']} | {row['true_class']} | {row['pred_class']} | "
            f"{float(row['p_abnormal']):.4f} | {float(row['left_mass_35']):.3f} | "
            f"{float(row['mid_mass_35']):.3f} | {float(row['right_mass_30']):.3f} | "
            f"{float(row['peak_x_norm']):.3f} | `{row['cam_overlay']}` |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- CAM target is the `abnormal` logit.",
            "- For FP, this shows what normal-looking pattern the model used as abnormal evidence.",
            "- `hard_normal` rows are true-normal samples with high `p_abnormal`, used only when actual FP rows are fewer than the requested limit.",
            "- CAM location is evidence location, not guaranteed defect location.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", required=True, type=Path)
    parser.add_argument("--scenarios", default=Path("data/scenarios.csv"), type=Path)
    parser.add_argument("--image-dir", default=Path("images"), type=Path)
    parser.add_argument("--out-dir", default=Path("validations/gradcam_errors"), type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--error-type", choices=["fp", "fn", "all"], default="fp")
    parser.add_argument("--layer", default="stages.3.blocks.2.conv_dw")
    parser.add_argument("--heat-threshold", type=float, default=0.25)
    parser.add_argument("--heat-min-alpha", type=float, default=0.0)
    parser.add_argument("--gallery-out", default=None, type=Path)
    parser.add_argument("--limit", type=int, default=0, help="maximum rows to run Grad-CAM for; 0 means all")
    parser.add_argument(
        "--fill-hard-normal",
        action="store_true",
        help="for FP reports, fill missing slots with normal samples ranked by p_abnormal",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def rank_candidates(rows: list[dict[str, Any]], error_type: str) -> list[dict[str, Any]]:
    if error_type == "fn":
        return sorted(rows, key=lambda row: float(row["p_abnormal"]))
    return sorted(rows, key=lambda row: float(row["p_abnormal"]), reverse=True)


def select_items_for_cam(predicted_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    error_rows: list[dict[str, Any]] = []
    hard_normal_rows: list[dict[str, Any]] = []
    for row in predicted_rows:
        if matches_error(row, args.error_type):
            error_type = "fp" if row["true_binary"] == "normal" else "fn"
            error_rows.append({**row, "error_type": error_type})
        elif args.error_type == "fp" and args.fill_hard_normal and row["true_binary"] == "normal":
            hard_normal_rows.append({**row, "error_type": "hard_normal"})

    selected = rank_candidates(error_rows, args.error_type)
    if args.limit > 0:
        selected = selected[: args.limit]
        remaining = args.limit - len(selected)
        if remaining > 0 and args.error_type == "fp" and args.fill_hard_normal:
            selected.extend(rank_candidates(hard_normal_rows, "fp")[:remaining])
    return selected


def main() -> int:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, classes = _load_model_from_best_info(args.model_run, device)
    abn_idx = classes.index("abnormal") if "abnormal" in classes else min(1, len(classes) - 1)
    gradcam = GradCam(model, resolve_layer(model, args.layer))
    transform = build_transform()
    items = load_items(args.scenarios, args.image_dir, args.split)

    predicted_rows: list[dict[str, Any]] = []
    for item in items:
        image_path = Path(item["image_path"])
        x = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_class, p_abn = binary_pred(F.softmax(model(x), dim=1)[0], classes)
        predicted_rows.append({**item, "pred_class": pred_class, "p_abnormal": p_abn})

    selected_rows = select_items_for_cam(predicted_rows, args)
    rows: list[dict[str, Any]] = []
    try:
        for item in selected_rows:
            image_path = Path(item["image_path"])
            x = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            cam, _ = gradcam(x, abn_idx)
            stats = heat_stats(cam)
            error_type = str(item["error_type"])
            out_name = f"{error_type}__{item['true_class']}__{item['chart_id']}.png"
            cam_overlay = args.out_dir / "cam_on_image" / out_name
            heat_only = args.out_dir / "heat_only" / out_name
            save_cam_on_image(image_path, cam, cam_overlay, args.heat_threshold, args.heat_min_alpha)
            save_heat_only_cam(cam, heat_only, args.heat_threshold)
            rows.append(
                {
                    **item,
                    "error_type": error_type,
                    "cam_overlay": str(cam_overlay),
                    "heat_only": str(heat_only),
                    **stats,
                }
            )
    finally:
        gradcam.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "gradcam_errors.csv"
    fieldnames = [
        "error_type",
        "chart_id",
        "true_class",
        "true_binary",
        "pred_class",
        "p_abnormal",
        "left_mass_35",
        "mid_mass_35",
        "right_mass_30",
        "peak_x_norm",
        "image_path",
        "cam_overlay",
        "heat_only",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.out_dir / "summary.md"
    summary_path.write_text(format_summary(rows, args), encoding="utf-8")
    if args.gallery_out is not None:
        write_gallery(rows, args.gallery_out)

    print(f"[gradcam-errors] wrote {csv_path}")
    print(f"[gradcam-errors] wrote {summary_path}")
    if args.gallery_out is not None:
        print(f"[gradcam-errors] gallery={args.gallery_out}")
    actual_errors = sum(1 for row in rows if row["error_type"] in {"fp", "fn"})
    hard_normal = sum(1 for row in rows if row["error_type"] == "hard_normal")
    print(f"[gradcam-errors] selected={len(rows)} actual_errors={actual_errors} hard_normal_fill={hard_normal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
