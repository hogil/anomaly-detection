#!/usr/bin/env python
"""Render inference images from existing trend CSV files.

This is the **production entry point** for taking field timeseries.csv
(unlabeled, no scenarios.csv) and turning it into model-input + display
images plus a manifest. Synthetic data with scenarios.csv also works
(passed via --scenarios) but for production the script auto-builds the
scenarios info from timeseries columns.

This is intentionally separate from generate_images.py:
- generate_images.py builds train/val/test/class folders for training.
- this script builds flat model-input images from existing trend data for inference.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
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


def synthesize_scenarios(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal scenarios DataFrame from a timeseries-only production CSV.

    Production CSV does not ship with scenarios.csv. We auto-detect the legend
    axis column (eqp_id / chamber / recipe / legend_axis) and emit one row per
    (chart_id, member) so every member becomes a highlighted-target image.
    """
    candidates = ("eqp_id", "chamber", "recipe", "legend_axis")
    axis_col = next((c for c in candidates if c in ts_df.columns), None)
    if axis_col is None:
        raise SystemExit(
            "timeseries.csv has no legend axis column. "
            f"Expected one of {candidates}. "
            "Pass --scenarios <path> if you already have a scenarios CSV."
        )
    if "chart_id" not in ts_df.columns:
        raise SystemExit("timeseries.csv must have a chart_id column")

    rows: list[dict[str, Any]] = []
    for chart_id, grp in ts_df.groupby("chart_id"):
        member_ids = sorted(grp[axis_col].dropna().astype(str).unique())
        if not member_ids:
            continue
        target_val = float(grp["value"].mean()) if "value" in grp.columns else 0.0
        member_str = ",".join(member_ids)
        for hm in member_ids:
            rows.append(
                {
                    "chart_id": str(chart_id),
                    "legend_axis": axis_col,
                    "members": member_str,
                    "highlighted_member": hm,
                    "target": round(target_val, 4),
                }
            )
    if not rows:
        raise SystemExit("auto-build produced no scenarios; check timeseries.csv contents")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeseries", required=True, type=Path, help="timeseries.csv path")
    parser.add_argument("--scenarios", default=None, type=Path,
                        help="optional. synthetic data: pass scenarios.csv. production: omit and the script auto-builds rows from timeseries.")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="output directory base name. The script prepends a YYMMDD_HHMMSS prefix.")
    parser.add_argument("--render-config", default="", help="YAML used only for image rendering style")
    parser.add_argument("--model-run", default="", help="optional logs/<run> directory; uses data_config_used.yaml if present")
    parser.add_argument("--x-col", default=None)
    parser.add_argument("--split", default="", help="optional split filter (synthetic only)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-display", action="store_true", help="only write model_inputs/")
    args = parser.parse_args()

    config = load_render_config(args)
    renderer = ImageRenderer(config)
    img_cfg = config.get("image", {})
    title_columns = img_cfg.get("title_columns", ["device", "step", "item"])
    y_label = img_cfg.get("y_label", "Measurement Value")

    ts_df = pd.read_csv(args.timeseries)
    x_col = detect_x_col(ts_df, args.x_col)
    if args.scenarios is not None:
        sc_df = pd.read_csv(args.scenarios)
        scenarios_source = str(args.scenarios)
    else:
        print("[inference-images] no --scenarios given; auto-building from timeseries")
        sc_df = synthesize_scenarios(ts_df)
        scenarios_source = "<auto from timeseries>"

    if args.split:
        if "split" not in sc_df.columns:
            raise SystemExit("--split was passed but scenarios has no split column")
        sc_df = sc_df[sc_df["split"].astype(str) == args.split].reset_index(drop=True)
    if args.limit and args.limit > 0:
        sc_df = sc_df.head(args.limit).reset_index(drop=True)

    ts_grouped = {str(chart_id): group for chart_id, group in ts_df.groupby("chart_id")}

    # output dir gets a timestamp prefix so reruns do not overwrite older results
    raw_out = args.out_dir
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    if raw_out.parent == Path("."):
        out_dir = raw_out.with_name(f"{stamp}_{raw_out.name}")
    else:
        out_dir = raw_out.parent / f"{stamp}_{raw_out.name}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"[inference-images] output -> {out_dir} (scenarios source: {scenarios_source})")

    input_dir = out_dir / "model_inputs"
    display_dir = out_dir / "display"
    input_dir.mkdir()
    if not args.no_display:
        display_dir.mkdir()

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

    if args.scenarios is None:
        synth_path = out_dir / "synthesized_scenarios.csv"
        sc_df.to_csv(synth_path, index=False)
        print(f"[inference-images] wrote auto-scenarios: {synth_path}")

    manifest_path = out_dir / "manifest.csv"
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
