#!/usr/bin/env python
"""Report binary AUROC and normal-threshold operating points.

Input is an inference `predictions.csv` with:
- `p_abnormal`: model abnormal probability
- `true_class`: ground-truth class where `normal` is normal and all other
  labels are abnormal

The threshold convention matches train.py:
`normal_threshold=0.9` means predict normal only when `p_normal > 0.9`,
so the abnormal cutoff is `p_abnormal >= 0.1`.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def parse_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not 0.0 <= value <= 1.0:
            raise argparse.ArgumentTypeError(f"threshold must be in [0, 1]: {value}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    return values


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)

    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = float(ranks[labels == 1].sum())
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def confusion_for_threshold(labels: np.ndarray, scores: np.ndarray, normal_threshold: float) -> dict[str, float]:
    cutoff = min(1.0, max(0.0, round(1.0 - normal_threshold, 12)))
    preds = (scores >= cutoff).astype(np.int64)

    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    abnormal_recall = tp / (tp + fn) if (tp + fn) else 0.0
    normal_recall = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * abnormal_recall / (precision + abnormal_recall) if (precision + abnormal_recall) else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0

    return {
        "normal_threshold": normal_threshold,
        "p_abnormal_cutoff": cutoff,
        "tn": tn,
        "fn": fn,
        "fp": fp,
        "tp": tp,
        "abnormal_recall": abnormal_recall,
        "normal_recall": normal_recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
    }


def format_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path, help="Path to inference predictions.csv")
    parser.add_argument(
        "--normal-thresholds",
        default="0.5,0.9,0.99,0.999",
        help="Comma-separated normal_threshold values using train.py convention.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Output prefix without extension. Default: <predictions dir>/binary_threshold_report",
    )
    args = parser.parse_args()

    predictions = args.predictions
    if not predictions.exists():
        raise SystemExit(f"predictions.csv not found: {predictions}")

    df = pd.read_csv(predictions)
    required = {"p_abnormal", "true_class"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            f"{predictions} is missing columns: {missing}. "
            "Run inference on a labeled split so predictions.csv includes true_class."
        )

    scores = pd.to_numeric(df["p_abnormal"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(scores) & df["true_class"].notna().to_numpy()
    if not valid.any():
        raise SystemExit(f"no labeled rows with finite p_abnormal in {predictions}")

    labels = (df.loc[valid, "true_class"].astype(str).to_numpy() != "normal").astype(np.int64)
    scores = scores[valid]

    try:
        thresholds = parse_thresholds(args.normal_thresholds)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    rows = [confusion_for_threshold(labels, scores, nt) for nt in thresholds]
    auroc = binary_auroc(labels, scores)

    out_prefix = args.out_prefix or predictions.with_name("binary_threshold_report")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_csv = out_prefix.with_suffix(".csv")
    out_md = out_prefix.with_suffix(".md")

    fieldnames = [
        "normal_threshold",
        "p_abnormal_cutoff",
        "tn",
        "fn",
        "fp",
        "tp",
        "abnormal_recall",
        "normal_recall",
        "precision",
        "f1",
        "accuracy",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    pos = int(labels.sum())
    neg = int((labels == 0).sum())
    md_lines = [
        "# Binary Threshold Report",
        "",
        f"- predictions: `{predictions}`",
        f"- labeled rows: {len(labels)}",
        f"- normal: {neg}",
        f"- abnormal: {pos}",
        f"- AUROC: {format_float(auroc)}",
        "",
        "| normal_threshold | p_abnormal_cutoff | TN | FN | FP | TP | abnormal recall | normal recall | precision | F1 | accuracy |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| "
            f"{row['normal_threshold']:.4g} | {row['p_abnormal_cutoff']:.4g} | "
            f"{row['tn']} | {row['fn']} | {row['fp']} | {row['tp']} | "
            f"{row['abnormal_recall']:.4f} | {row['normal_recall']:.4f} | "
            f"{row['precision']:.4f} | {row['f1']:.4f} | {row['accuracy']:.4f} |"
        )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[binary_threshold_report] rows={len(labels)} auroc={format_float(auroc)}")
    print(f"  md : {out_md}")
    print(f"  csv: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
