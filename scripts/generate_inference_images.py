#!/usr/bin/env python
"""Render inference images from existing trend CSV files.

This is intentionally separate from generate_images.py:
- generate_images.py builds train/val/test/class folders for training.
- this script builds flat model-input images from existing trend data for inference.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.image_renderer import ImageRenderer
from src.data.schema import highlighted_member as read_highlighted_member
from src.data.schema import legend_axis as read_legend_axis
from src.data.schema import members as read_members
from src.data.schema import target as read_target

TQDM_DISABLE = not sys.stderr.isatty()


def sanitize_part(value: Any) -> str:
    text = str(value if value is not None and pd.notna(value) else "na").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "na"


def detect_x_col(ts_df: pd.DataFrame, override: str | None) -> str:
    if override:
        if override not in ts_df.columns:
            raise SystemExit(f"x column not found in timeseries: {override}")
        return override
    for candidate in ["time_index", "timestamp", "datetime", "date", "time"]:
        if candidate in ts_df.columns:
            return candidate
    raise SystemExit("no x column found; pass --x-col")


def load_render_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.render_config:
        path = Path(args.render_config)
    elif args.model_run:
        model_run = Path(args.model_run)
        path = model_run / "data_config_used.yaml"
        if not path.exists():
            path = Path("dataset.yaml")
    else:
        path = Path("dataset.yaml")
    if not path.exists():
        raise SystemExit(f"render config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "image" not in config:
        raise SystemExit(f"render config has no image section: {path}")
    return config


def build_fleet_data(
    row: pd.Series,
    ts_grouped: dict[str, pd.DataFrame],
    x_col: str,
) -> tuple[dict[str, tuple], str] | None:
    chart_id = str(row.get("chart_id", ""))
    if not chart_id:
        return None
    chart_ts = ts_grouped.get(chart_id)
    if chart_ts is None or chart_ts.empty:
        return None

    axis = read_legend_axis(row)
    if not axis or axis not in chart_ts.columns:
        raise SystemExit(f"legend_axis missing or invalid for chart {chart_id}: {axis}")

    highlighted_member = read_highlighted_member(row)
    if not highlighted_member:
        return None

    fleet_data: dict[str, tuple] = {}
    for member_id in read_members(row, chart_ts=chart_ts, axis=axis):
        member_ts = chart_ts[chart_ts[axis].astype(str) == str(member_id)].sort_values(x_col)
        if member_ts.empty:
            continue
        fleet_data[str(member_id)] = (
            member_ts[x_col].to_numpy(),
            member_ts["value"].to_numpy(),
        )
    if highlighted_member not in fleet_data:
        return None
    return fleet_data, highlighted_member


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeseries", required=True, type=Path, help="timeseries.csv path")
    parser.add_argument("--scenarios", required=True, type=Path, help="scenarios.csv or scenarios_per_member.csv path")
    parser.add_argument("--out-dir", required=True, type=Path, help="output directory")
    parser.add_argument("--render-config", default="", help="YAML used only for image rendering style")
    parser.add_argument("--model-run", default="", help="optional logs/<run> directory; uses data_config_used.yaml if present")
    parser.add_argument("--x-col", default=None)
    parser.add_argument("--split", default="", help="optional split filter")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-display", action="store_true", help="only write model_inputs/")
    args = parser.parse_args()

    config = load_render_config(args)
    renderer = ImageRenderer(config)
    img_cfg = config.get("image", {})
    title_columns = img_cfg.get("title_columns", ["device", "step", "item"])
    y_label = img_cfg.get("y_label", "Measurement Value")

    ts_df = pd.read_csv(args.timeseries)
    sc_df = pd.read_csv(args.scenarios)
    x_col = detect_x_col(ts_df, args.x_col)

    if args.split:
        if "split" not in sc_df.columns:
            raise SystemExit("--split was passed but scenarios has no split column")
        sc_df = sc_df[sc_df["split"].astype(str) == args.split].reset_index(drop=True)
    if args.limit and args.limit > 0:
        sc_df = sc_df.head(args.limit).reset_index(drop=True)

    ts_grouped = {str(chart_id): group for chart_id, group in ts_df.groupby("chart_id")}

    input_dir = args.out_dir / "model_inputs"
    display_dir = args.out_dir / "display"
    input_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_display:
        display_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    skipped = 0
    for idx, row in tqdm(sc_df.iterrows(), total=len(sc_df), desc="render-inference", disable=TQDM_DISABLE):
        built = build_fleet_data(row, ts_grouped, x_col)
        if built is None:
            skipped += 1
            continue
        fleet_data, highlighted_member = built
        chart_id = str(row.get("chart_id"))
        stem = f"{idx:06d}_{sanitize_part(chart_id)}_{sanitize_part(highlighted_member)}"
        input_path = input_dir / f"{stem}.png"
        target = read_target(row)
        renderer.render_overlay(fleet_data, highlighted_member, str(input_path), target=target)

        display_path = ""
        if not args.no_display:
            title_parts = [str(row.get(col)) for col in title_columns if col in row and pd.notna(row.get(col))]
            title = " / ".join(title_parts) if title_parts else chart_id
            display_file = display_dir / f"{stem}.png"
            renderer.render_overlay_display(
                fleet_data,
                highlighted_member,
                str(display_file),
                anomalous_ids=[],
                defect_start_idx=None,
                title=title,
                x_label=x_col,
                y_label=y_label,
                target=target,
            )
            display_path = str(display_file)

        manifest_rows.append(
            {
                "chart_id": chart_id,
                "highlighted_member": highlighted_member,
                "model_input": str(input_path),
                "display": display_path,
                "class": row.get("class", ""),
                "split": row.get("split", ""),
            }
        )

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["chart_id", "highlighted_member", "model_input", "display", "class", "split"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[inference-images] wrote {len(manifest_rows)} images, skipped={skipped}, manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
