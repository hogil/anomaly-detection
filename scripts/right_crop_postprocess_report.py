#!/usr/bin/env python
"""Evaluate right-crop postprocessing on labeled image folders."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import _load_model_from_best_info


@dataclass(frozen=True)
class CropItem:
    chart_id: str
    true_class: str
    image_path: Path


class CropDataset(Dataset):
    def __init__(self, items: list[CropItem], crop_ratio: float, transform) -> None:
        self.items = items
        self.crop_ratio = crop_ratio
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        image = Image.open(item.image_path).convert("RGB")
        w, h = image.size
        left = max(0, min(w - 1, int(round(w * (1.0 - self.crop_ratio)))))
        right_crop = image.crop((left, 0, w, h))
        return {
            "full": self.transform(image),
            "crop": self.transform(right_crop),
            "chart_id": item.chart_id,
            "true_class": item.true_class,
            "image_path": str(item.image_path),
        }


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_items(scenarios_csv: Path, image_dir: Path, split: str, limit: int) -> list[CropItem]:
    scenarios = pd.read_csv(scenarios_csv)
    if split:
        scenarios = scenarios[scenarios["split"].astype(str) == split].reset_index(drop=True)
    if limit > 0:
        scenarios = scenarios.head(limit).reset_index(drop=True)

    items: list[CropItem] = []
    missing = 0
    for row in scenarios.to_dict(orient="records"):
        chart_id = str(row["chart_id"])
        true_class = str(row["class"])
        split_name = str(row["split"])
        image_path = image_dir / split_name / true_class / f"{chart_id}.png"
        if not image_path.exists():
            missing += 1
            continue
        items.append(CropItem(chart_id=chart_id, true_class=true_class, image_path=image_path))
    if missing:
        print(f"[right-crop] skipped missing images: {missing}")
    if not items:
        raise SystemExit("no image items found")
    return items


def probs_to_labels(probs: torch.Tensor, classes: list[str], abnormal_threshold: float) -> tuple[list[str], list[float]]:
    if "abnormal" in classes:
        abnormal_idx = classes.index("abnormal")
    else:
        abnormal_idx = min(1, len(classes) - 1)

    p_abnormal = probs[:, abnormal_idx].detach().cpu().numpy().astype(float).tolist()
    if abnormal_threshold >= 0:
        labels = ["abnormal" if p >= abnormal_threshold else "normal" for p in p_abnormal]
    else:
        pred_idx = torch.argmax(probs, dim=1).detach().cpu().numpy().tolist()
        labels = [classes[i] if i < len(classes) else str(i) for i in pred_idx]
    return labels, p_abnormal


def binary_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    true_abn = np.asarray([label != "normal" for label in y_true], dtype=bool)
    pred_abn = np.asarray([label != "normal" for label in y_pred], dtype=bool)
    tp = int((true_abn & pred_abn).sum())
    tn = int((~true_abn & ~pred_abn).sum())
    fp = int((~true_abn & pred_abn).sum())
    fn = int((true_abn & ~pred_abn).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    abn_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    normal_precision = tn / (tn + fn) if tn + fn else 0.0
    normal_recall = tn / (tn + fp) if tn + fp else 0.0
    normal_f1 = (
        2 * normal_precision * normal_recall / (normal_precision + normal_recall)
        if normal_precision + normal_recall
        else 0.0
    )
    return {
        "n": len(y_true),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "precision_abnormal": precision,
        "recall_abnormal": recall,
        "f1_abnormal": abn_f1,
        "f1_macro": (abn_f1 + normal_f1) / 2,
    }


def format_metrics_table(metrics: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| policy | F1 macro | abnormal F1 | FN | FP | TP | TN | accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in metrics.items():
        lines.append(
            f"| {name} | {row['f1_macro']:.4f} | {row['f1_abnormal']:.4f} | "
            f"{row['fn']} | {row['fp']} | {row['tp']} | {row['tn']} | {row['accuracy']:.4f} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", required=True, type=Path)
    parser.add_argument("--scenarios", default=Path("data/scenarios.csv"), type=Path)
    parser.add_argument("--image-dir", default=Path("images"), type=Path)
    parser.add_argument("--out-dir", default=Path("validations/right_crop_postprocess"), type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--crop-ratio", type=float, default=0.4, help="right-side crop width ratio")
    parser.add_argument(
        "--abnormal-threshold",
        type=float,
        default=-1.0,
        help="p_abnormal threshold. -1 uses argmax, 0.5 is binary argmax equivalent.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.05 <= args.crop_ratio <= 0.95:
        raise SystemExit("--crop-ratio must be between 0.05 and 0.95")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, classes = _load_model_from_best_info(args.model_run, device)
    transform = build_transform()
    items = load_items(args.scenarios, args.image_dir, args.split, args.limit)
    loader = DataLoader(
        CropDataset(items, args.crop_ratio, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            full = batch["full"].to(device)
            crop = batch["crop"].to(device)
            full_probs = F.softmax(model(full), dim=1)
            crop_probs = F.softmax(model(crop), dim=1)
            full_pred, full_p_abn = probs_to_labels(full_probs, classes, args.abnormal_threshold)
            crop_pred, crop_p_abn = probs_to_labels(crop_probs, classes, args.abnormal_threshold)

            for i, chart_id in enumerate(batch["chart_id"]):
                true_class = str(batch["true_class"][i])
                true_binary = "normal" if true_class == "normal" else "abnormal"
                final_all = full_pred[i]
                if full_pred[i] == "abnormal" and crop_pred[i] == "normal":
                    final_all = "normal"

                final_rescue_fn = full_pred[i]
                if full_pred[i] == "normal" and crop_pred[i] == "abnormal":
                    final_rescue_fn = "abnormal"

                final_bidirectional = full_pred[i]
                if crop_pred[i] != full_pred[i]:
                    final_bidirectional = crop_pred[i]

                # Analysis-only oracle: true context is not gated because context is global/fleet-relative.
                final_context_exempt = final_all
                if true_class == "context":
                    final_context_exempt = full_pred[i]

                rows.append(
                    {
                        "chart_id": str(chart_id),
                        "true_class": true_class,
                        "true_binary": true_binary,
                        "full_pred": full_pred[i],
                        "right_crop_pred": crop_pred[i],
                        "final_crop_gate_all": final_all,
                        "final_crop_rescue_fn": final_rescue_fn,
                        "final_crop_bidirectional": final_bidirectional,
                        "final_crop_gate_context_exempt": final_context_exempt,
                        "p_abnormal_full": full_p_abn[i],
                        "p_abnormal_right_crop": crop_p_abn[i],
                        "image_path": str(batch["image_path"][i]),
                    }
                )

    y_true = [r["true_binary"] for r in rows]
    metrics = {
        "full_image": binary_metrics(y_true, [r["full_pred"] for r in rows]),
        "right_crop_gate_all": binary_metrics(y_true, [r["final_crop_gate_all"] for r in rows]),
        "right_crop_rescue_fn": binary_metrics(y_true, [r["final_crop_rescue_fn"] for r in rows]),
        "right_crop_bidirectional": binary_metrics(y_true, [r["final_crop_bidirectional"] for r in rows]),
        "right_crop_gate_context_exempt_analysis": binary_metrics(
            y_true, [r["final_crop_gate_context_exempt"] for r in rows]
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "right_crop_postprocess.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = [
        "# Right-Crop Postprocess Report",
        "",
        f"- model_run: `{args.model_run}`",
        f"- split: `{args.split}`",
        f"- crop_ratio: `{args.crop_ratio}`",
        f"- abnormal_threshold: `{'argmax' if args.abnormal_threshold < 0 else args.abnormal_threshold}`",
        "",
        format_metrics_table(metrics),
        "",
        "Policy:",
        "- `full_image`: original full-image model decision.",
        "- `right_crop_gate_all`: if full image is abnormal but right crop is normal, final is changed to normal.",
        "- `right_crop_rescue_fn`: if full image is normal but right crop is abnormal, final is changed to abnormal.",
        "- `right_crop_bidirectional`: right crop wins whenever full image and right crop disagree.",
        "- `right_crop_gate_context_exempt_analysis`: same gate, but true context samples are not gated. This is analysis-only unless a class-aware model exists.",
    ]
    summary_path = args.out_dir / "summary.md"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[right-crop] wrote {csv_path}")
    print(f"[right-crop] wrote {summary_path}")
    print(format_metrics_table(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
