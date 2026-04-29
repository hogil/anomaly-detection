"""Scenario schema helpers.

Canonical scenario columns:
- legend_axis: the member axis shown in the legend, e.g. eqp_id/chamber/recipe
- members: comma-separated fleet members for the chart
- highlighted_member: member emphasized in the rendered image
- target: horizontal reference value used for the baseline line

Legacy generated CSVs used context_column/contexts/target/target_value.  Readers
keep fallback support so old datasets can still be rendered and evaluated.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(value != value)
    except Exception:
        return False


def first_present(row: Any, *keys: str) -> Any:
    for key in keys:
        try:
            value = row.get(key)
        except AttributeError:
            value = row[key] if key in row else None
        if not is_missing(value) and str(value).strip() != "":
            return value
    return None


def optional_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def legend_axis(row: Any) -> str:
    value = first_present(row, "legend_axis", "context_column")
    return "" if value is None else str(value)


def members(row: Any, chart_ts: Any = None, axis: str | None = None) -> list[str]:
    raw = first_present(row, "members", "contexts")
    if raw is not None:
        if isinstance(raw, str):
            return [part.strip() for part in raw.split(",") if part.strip()]
        if isinstance(raw, Iterable):
            return [str(part) for part in raw if not is_missing(part) and str(part).strip()]

    if chart_ts is not None and axis:
        return [str(x) for x in chart_ts[axis].dropna().unique().tolist()]
    return []


def highlighted_member(row: Any) -> str | None:
    value = first_present(row, "highlighted_member", "target_member")
    if value is not None:
        return str(value)

    legacy_target = first_present(row, "target")
    if optional_float(legacy_target) is None and legacy_target is not None:
        return str(legacy_target)
    return None


def target(row: Any) -> float | None:
    value = first_present(row, "target")
    parsed = optional_float(value)
    if parsed is not None:
        return parsed
    return optional_float(first_present(row, "target_value"))
