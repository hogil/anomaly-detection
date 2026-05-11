#!/usr/bin/env python
"""Render field timeseries into images for prediction or future dev training.

This script never synthesizes anomaly values. If a label column is provided,
the label is used only to route already-rendered images into normal/abnormal
folders and to write manifest metadata.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.image_renderer import ImageRenderer  # noqa: E402

TQDM_DISABLE = not sys.stderr.isatty()

DEFAULT_NORMAL_LABELS = {"normal", "good", "ok", "pass", "0", "false", "양호"}
DEFAULT_ABNORMAL_LABELS = {"abnormal", "bad", "ng", "fail", "failed", "1", "true", "불량"}


def parse_csv_set(text: str) -> set[str]:
    return {part.strip().lower() for part in text.split(",") if part.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeseries", required=True, type=Path, help="Field timeseries CSV.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--render-config", type=Path, default=None, help="YAML with the image section. Defaults to dataset.yaml.")
    parser.add_argument("--model-run", type=Path, default=None, help="Optional run dir; uses data_config_used.yaml when present.")
    parser.add_argument("--x-col", default=None, help="X axis column. Auto-detects time_index/timestamp/datetime/date/time.")
    parser.add_argument("--value-col", default="value", help="Measurement value column.")
    parser.add_argument("--chart-id-col", default="chart_id", help="Existing chart id column, if present.")
    parser.add_argument(
        "--chart-cols",
        default="device,step,item",
        help="Columns used to build chart_id when --chart-id-col is absent.",
    )
    parser.add_argument("--legend-axis", default=None, help="Member/fleet column. Auto-detects eqp_id/chamber/recipe.")
    parser.add_argument("--label-col", default=None, help="Optional field label column: good/bad, 양호/불량, normal/abnormal.")
    parser.add_argument("--normal-labels", default=",".join(sorted(DEFAULT_NORMAL_LABELS)))
    parser.add_argument("--abnormal-labels", default=",".join(sorted(DEFAULT_ABNORMAL_LABELS)))
    parser.add_argument("--target-col", default=None, help="Optional baseline/target value column.")
    parser.add_argument("--limit", type=int, default=0, help="Limit rendered scenario rows after expansion.")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-dev-folders", action="store_true", help="Do not write dev_model_inputs/{normal,abnormal}.")
    parser.add_argument("--no-timestamp", action="store_true", help="Use out-dir exactly instead of prefixing YYMMDD_HHMMSS.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it already exists.")
    return parser.parse_args()


def sanitize_part(value: Any) -> str:
    text = str(value if value is not None and pd.notna(value) else "na").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._-") or "na"


def load_render_config(args: argparse.Namespace) -> dict[str, Any]:
    model_run = None
    if args.model_run is not None:
        model_run = args.model_run.parent if args.model_run.is_file() else args.model_run
    if args.render_config is not None:
        path = args.render_config
    elif model_run is not None and (model_run / "data_config_used.yaml").exists():
        path = model_run / "data_config_used.yaml"
    else:
        path = ROOT / "dataset.yaml"
    if not path.exists():
        raise FileNotFoundError(f"render config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "image" not in config:
        raise ValueError(f"render config has no image section: {path}")
    return config


def detect_x_col(df: pd.DataFrame, override: str | None) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"x column not found: {override}")
        return override
    for candidate in ["time_index", "timestamp", "datetime", "date", "time"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("no x column found; pass --x-col")


def detect_legend_axis(df: pd.DataFrame, override: str | None) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"legend axis column not found: {override}")
        return override
    for candidate in ["eqp_id", "chamber", "recipe", "member", "member_id", "tool_id"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("no legend axis column found; pass --legend-axis")


def add_chart_id(df: pd.DataFrame, chart_id_col: str, chart_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    if chart_id_col in df.columns:
        df["_field_chart_id"] = df[chart_id_col].astype(str)
        return df
    missing = [col for col in chart_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{chart_id_col} is absent and chart columns are missing: {missing}. "
            "Pass --chart-cols with columns present in the CSV."
        )
    df["_field_chart_id"] = df[chart_cols].astype(str).agg("_".join, axis=1)
    return df


def maybe_parse_datetime(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    if pd.api.types.is_numeric_dtype(df[x_col]):
        return df
    if not any(token in x_col.lower() for token in ["time", "date", "timestamp", "datetime"]):
        return df
    converted = pd.to_datetime(df[x_col], errors="coerce")
    if converted.notna().mean() >= 0.8:
        df = df.copy()
        df[x_col] = converted
    return df


def normalized_label(value: Any, normal_labels: set[str], abnormal_labels: set[str]) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in normal_labels:
        return "normal"
    if text in abnormal_labels:
        return "abnormal"
    raise ValueError(f"unknown label value: {value!r}")


def label_for_member(
    group: pd.DataFrame,
    member_id: str,
    legend_axis: str,
    label_col: str | None,
    normal_labels: set[str],
    abnormal_labels: set[str],
) -> tuple[str, bool]:
    if not label_col:
        return "", False
    member_rows = group[group[legend_axis].astype(str) == str(member_id)]
    labels = [
        normalized_label(value, normal_labels, abnormal_labels)
        for value in member_rows[label_col].dropna().tolist()
    ]
    labels = [label for label in labels if label]
    if not labels:
        return "", False
    uniq = set(labels)
    if "abnormal" in uniq:
        return "abnormal", len(uniq) > 1
    return "normal", len(uniq) > 1


def target_for_member(group: pd.DataFrame, member_id: str, legend_axis: str, value_col: str, target_col: str | None) -> float | None:
    if target_col and target_col in group.columns:
        vals = pd.to_numeric(group[target_col], errors="coerce").dropna()
        if not vals.empty:
            return float(vals.median())

    fleet = group[group[legend_axis].astype(str) != str(member_id)]
    vals = pd.to_numeric(fleet[value_col], errors="coerce").dropna()
    if vals.empty:
        vals = pd.to_numeric(group[value_col], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.median())


def build_rows(
    df: pd.DataFrame,
    x_col: str,
    value_col: str,
    legend_axis: str,
    chart_cols: list[str],
    label_col: str | None,
    normal_labels: set[str],
    abnormal_labels: set[str],
    target_col: str | None,
) -> list[dict[str, Any]]:
    if value_col not in df.columns:
        raise ValueError(f"value column not found: {value_col}")
    if label_col and label_col not in df.columns:
        raise ValueError(f"label column not found: {label_col}")

    rows: list[dict[str, Any]] = []
    for chart_id, group in df.groupby("_field_chart_id", sort=True):
        members = sorted(group[legend_axis].dropna().astype(str).unique())
        if not members:
            continue
        member_csv = ",".join(members)
        meta = {col: group[col].dropna().iloc[0] for col in chart_cols if col in group.columns and group[col].notna().any()}
        for member_id in members:
            label, conflict = label_for_member(group, member_id, legend_axis, label_col, normal_labels, abnormal_labels)
            target = target_for_member(group, member_id, legend_axis, value_col, target_col)
            row = {
                "chart_id": str(chart_id),
                "legend_axis": legend_axis,
                "members": member_csv,
                "highlighted_member": member_id,
                "target": "" if target is None else round(target, 6),
                "class": label,
                "true_class": label,
                "label_conflict": conflict,
            }
            row.update(meta)
            rows.append(row)
    return rows


def build_fleet_data(row: dict[str, Any], chart_ts: pd.DataFrame, x_col: str, value_col: str) -> dict[str, tuple]:
    axis = row["legend_axis"]
    fleet_data = {}
    for member_id in str(row["members"]).split(","):
        member_id = member_id.strip()
        if not member_id:
            continue
        member_ts = chart_ts[chart_ts[axis].astype(str) == str(member_id)].sort_values(x_col)
        if member_ts.empty:
            continue
        fleet_data[member_id] = (
            member_ts[x_col].to_numpy(),
            pd.to_numeric(member_ts[value_col], errors="coerce").to_numpy(),
        )
    return fleet_data


def output_dir(base: Path, no_timestamp: bool, overwrite: bool) -> Path:
    out = base
    if not no_timestamp:
        stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        out = base.with_name(f"{stamp}_{base.name}") if base.parent == Path(".") else base.parent / f"{stamp}_{base.name}"
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def main() -> int:
    args = parse_args()
    config = load_render_config(args)
    renderer = ImageRenderer(config)
    img_cfg = config.get("image", {})
    title_columns = img_cfg.get("title_columns", ["device", "step", "item"])
    y_label = img_cfg.get("y_label", "Measurement Value")

    df = pd.read_csv(args.timeseries)
    x_col = detect_x_col(df, args.x_col)
    legend_axis = detect_legend_axis(df, args.legend_axis)
    chart_cols = [col.strip() for col in args.chart_cols.split(",") if col.strip()]
    df = add_chart_id(df, args.chart_id_col, chart_cols)
    df = maybe_parse_datetime(df, x_col)

    normal_labels = parse_csv_set(args.normal_labels)
    abnormal_labels = parse_csv_set(args.abnormal_labels)
    rows = build_rows(
        df,
        x_col=x_col,
        value_col=args.value_col,
        legend_axis=legend_axis,
        chart_cols=chart_cols,
        label_col=args.label_col,
        normal_labels=normal_labels,
        abnormal_labels=abnormal_labels,
        target_col=args.target_col,
    )
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("no render rows produced")

    out = output_dir(args.out_dir, args.no_timestamp, args.overwrite)
    input_dir = out / "model_inputs"
    display_dir = out / "display"
    input_dir.mkdir()
    if not args.no_display:
        display_dir.mkdir()

    write_dev = bool(args.label_col) and not args.no_dev_folders
    if write_dev:
        for cls in ["normal", "abnormal"]:
            (out / "dev_model_inputs" / cls).mkdir(parents=True, exist_ok=True)
            if not args.no_display:
                (out / "dev_display" / cls).mkdir(parents=True, exist_ok=True)

    ts_grouped = {str(chart_id): group for chart_id, group in df.groupby("_field_chart_id", sort=True)}
    manifest_rows: list[dict[str, Any]] = []
    skipped = 0
    for idx, row in tqdm(list(enumerate(rows)), total=len(rows), desc="field-images", disable=TQDM_DISABLE):
        chart_id = row["chart_id"]
        highlighted = row["highlighted_member"]
        chart_ts = ts_grouped.get(chart_id)
        if chart_ts is None:
            skipped += 1
            continue
        fleet_data = build_fleet_data(row, chart_ts, x_col, args.value_col)
        if highlighted not in fleet_data:
            skipped += 1
            continue

        stem = f"{idx:06d}_{sanitize_part(chart_id)}_{sanitize_part(highlighted)}"
        input_path = input_dir / f"{stem}.png"
        target = None if row.get("target") == "" else float(row["target"])
        renderer.render_overlay(fleet_data, highlighted, str(input_path), target=target)

        display_path = None
        if not args.no_display:
            title_parts = [str(row.get(col)) for col in title_columns if row.get(col) not in (None, "")]
            title = " / ".join(title_parts) if title_parts else chart_id
            display_path = display_dir / f"{stem}.png"
            renderer.render_overlay_display(
                fleet_data,
                highlighted,
                str(display_path),
                anomalous_ids=[],
                defect_start_idx=None,
                title=title,
                x_label=x_col,
                y_label=y_label,
                target=target,
            )

        dev_input = ""
        dev_display = ""
        label = str(row.get("class") or "")
        if write_dev and label in {"normal", "abnormal"}:
            dev_input_path = out / "dev_model_inputs" / label / f"{stem}.png"
            shutil.copy2(input_path, dev_input_path)
            dev_input = rel(dev_input_path, out)
            if display_path is not None:
                dev_display_path = out / "dev_display" / label / f"{stem}.png"
                shutil.copy2(display_path, dev_display_path)
                dev_display = rel(dev_display_path, out)

        manifest = dict(row)
        manifest.update(
            {
                "model_input": rel(input_path, out),
                "display": "" if display_path is None else rel(display_path, out),
                "dev_model_input": dev_input,
                "dev_display": dev_display,
                "x_col": x_col,
                "value_col": args.value_col,
            }
        )
        manifest_rows.append(manifest)

    scenarios_path = out / "field_scenarios.csv"
    pd.DataFrame(rows).to_csv(scenarios_path, index=False)

    fieldnames = sorted({key for row in manifest_rows for key in row.keys()})
    preferred = [
        "chart_id",
        "highlighted_member",
        "class",
        "true_class",
        "model_input",
        "display",
        "dev_model_input",
        "dev_display",
        "legend_axis",
        "members",
        "target",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(manifest_rows)

    counts = Counter(row.get("class") or "unlabeled" for row in manifest_rows)
    summary = {
        "timeseries": str(args.timeseries),
        "rendered": len(manifest_rows),
        "skipped": skipped,
        "x_col": x_col,
        "legend_axis": legend_axis,
        "label_col": args.label_col or "",
        "class_counts": dict(counts),
        "note": "No anomaly values were synthesized; labels only route images and metadata.",
    }
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        import json

        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"[field-images] rendered={len(manifest_rows)} skipped={skipped} output={out}")
    print(f"[field-images] class_counts={dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
