#!/usr/bin/env python
"""Test whether one-sided abnormal Grad-CAM can rescue full-image FN cases."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gradcam_report import GradCam, build_transform, heat_stats, resolve_layer
from inference import _load_model_from_best_info
from right_crop_postprocess_report import binary_metrics


def load_items(scenarios_csv: Path, image_dir: Path, split: str, limit: int) -> list[dict[str, str]]:
    scenarios = pd.read_csv(scenarios_csv)
    if split:
        scenarios = scenarios[scenarios["split"].astype(str) == split].reset_index(drop=True)
    if limit > 0:
        scenarios = scenarios.head(limit).reset_index(drop=True)

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


def pred_from_probs(probs: torch.Tensor, classes: list[str]) -> tuple[str, float]:
    pred_idx = int(torch.argmax(probs).item())
    pred = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
    abn_idx = classes.index("abnormal") if "abnormal" in classes else min(1, len(classes) - 1)
    return pred, float(probs[abn_idx].item())


def sweep_rescue(rows: list[dict[str, Any]], side_thresholds: list[float], gap_thresholds: list[float]) -> list[dict[str, Any]]:
    y_true = [r["true_binary"] for r in rows]
    out: list[dict[str, Any]] = []
    baseline = binary_metrics(y_true, [r["full_pred"] for r in rows])
    out.append({"policy": "full_image", "side_threshold": "", "gap_threshold": "", "rescued_fn": 0, "added_fp": 0, **baseline})

    for side_th in side_thresholds:
        for gap_th in gap_thresholds:
            preds = []
            rescued_fn = 0
            added_fp = 0
            for row in rows:
                pred = row["full_pred"]
                should_rescue = (
                    row["full_pred"] == "normal"
                    and row.get("cam_side_mass") is not None
                    and float(row["cam_side_mass"]) >= side_th
                    and float(row["cam_side_gap"]) >= gap_th
                )
                if should_rescue:
                    pred = "abnormal"
                    if row["true_binary"] == "abnormal":
                        rescued_fn += 1
                    else:
                        added_fp += 1
                preds.append(pred)
            metrics = binary_metrics(y_true, preds)
            out.append(
                {
                    "policy": "gradcam_normal_rescue",
                    "side_threshold": side_th,
                    "gap_threshold": gap_th,
                    "rescued_fn": rescued_fn,
                    "added_fp": added_fp,
                    **metrics,
                }
            )
    return out


def format_sweep_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| policy | side | gap | F1 macro | abnormal F1 | FN | FP | rescued FN | added FP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['side_threshold']} | {row['gap_threshold']} | "
            f"{float(row['f1_macro']):.4f} | {float(row['f1_abnormal']):.4f} | "
            f"{int(row['fn'])} | {int(row['fp'])} | {int(row['rescued_fn'])} | {int(row['added_fp'])} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", required=True, type=Path)
    parser.add_argument("--scenarios", default=Path("data/scenarios.csv"), type=Path)
    parser.add_argument("--image-dir", default=Path("images"), type=Path)
    parser.add_argument("--out-dir", default=Path("validations/gradcam_normal_rescue"), type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--layer", default="stages.3.blocks.2.conv_dw")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, classes = _load_model_from_best_info(args.model_run, device)
    abn_idx = classes.index("abnormal") if "abnormal" in classes else min(1, len(classes) - 1)
    layer = resolve_layer(model, args.layer)
    gradcam = GradCam(model, layer)
    transform = build_transform()
    items = load_items(args.scenarios, args.image_dir, args.split, args.limit)

    rows: list[dict[str, Any]] = []
    try:
        for idx, item in enumerate(items, 1):
            image = Image.open(item["image_path"]).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = F.softmax(model(x), dim=1)[0]
                full_pred, p_abn = pred_from_probs(probs, classes)

            row: dict[str, Any] = {
                **item,
                "full_pred": full_pred,
                "p_abnormal_full": p_abn,
                "cam_left_mass_35": "",
                "cam_mid_mass_35": "",
                "cam_right_mass_30": "",
                "cam_side": "",
                "cam_side_mass": None,
                "cam_side_gap": None,
            }
            if full_pred == "normal":
                cam, _ = gradcam(x, abn_idx)
                stats = heat_stats(cam)
                left = float(stats["left_mass_35"])
                mid = float(stats["mid_mass_35"])
                right = float(stats["right_mass_30"])
                if left >= right:
                    side = "left"
                    side_mass = left
                else:
                    side = "right"
                    side_mass = right
                row.update(
                    {
                        "cam_left_mass_35": left,
                        "cam_mid_mass_35": mid,
                        "cam_right_mass_30": right,
                        "cam_side": side,
                        "cam_side_mass": side_mass,
                        "cam_side_gap": side_mass - mid,
                    }
                )
            rows.append(row)
            if idx % 100 == 0:
                print(f"[gradcam-rescue] processed {idx}/{len(items)}", flush=True)
    finally:
        gradcam.close()

    side_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    gap_thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    sweep = sweep_rescue(rows, side_thresholds, gap_thresholds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "gradcam_normal_rows.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sweep_path = args.out_dir / "gradcam_normal_rescue_sweep.csv"
    with sweep_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)

    baseline = [r for r in sweep if r["policy"] == "full_image"]
    best_by_f1 = sorted(
        [r for r in sweep if r["policy"] != "full_image"],
        key=lambda r: (-float(r["f1_macro"]), int(r["fn"]), int(r["fp"])),
    )[:10]
    best_by_fn = sorted(
        [r for r in sweep if r["policy"] != "full_image"],
        key=lambda r: (int(r["fn"]), int(r["fp"]), -float(r["f1_macro"])),
    )[:10]
    summary_lines = [
        "# Grad-CAM Normal-Rescue Report",
        "",
        f"- model_run: `{args.model_run}`",
        f"- split: `{args.split}`",
        "- Rule tested: if full image predicts normal, flip to abnormal when abnormal-logit Grad-CAM is strongly one-sided.",
        "",
        "## Baseline",
        "",
        format_sweep_table(baseline),
        "",
        "## Best By F1",
        "",
        format_sweep_table(best_by_f1[:8]),
        "",
        "## Best FN Rescue",
        "",
        format_sweep_table(best_by_fn[:8]),
        "",
        "## Caveat",
        "",
        "Grad-CAM heat is model evidence location, not guaranteed anomaly location. This report is a threshold screen, not a production rule.",
    ]
    summary_path = args.out_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (args.out_dir / "best_thresholds.json").write_text(
        json.dumps({"best_by_f1": best_by_f1, "best_by_fn": best_by_fn}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[gradcam-rescue] wrote {rows_path}")
    print(f"[gradcam-rescue] wrote {sweep_path}")
    print(f"[gradcam-rescue] wrote {summary_path}")
    print("\n".join(summary_lines[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
