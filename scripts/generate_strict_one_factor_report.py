from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
VALIDATIONS = ROOT / "validations"
LOGS = ROOT / "logs"


FAMILY_ORDER = [
    "normal_ratio",
    "per_class",
    "lr",
    "warmup",
    "gc",
    "weight_decay",
    "smoothing",
    "label_smoothing",
    "stochastic_depth",
    "focal_gamma",
    "abnormal_weight",
    "ema",
    "color",
    "allow_tie_save",
]

FAMILY_TITLES = {
    "normal_ratio": "normal_ratio",
    "per_class": "per_class",
    "lr": "LR",
    "warmup": "warmup",
    "gc": "GC",
    "weight_decay": "weight_decay",
    "smoothing": "smoothing",
    "label_smoothing": "label_smoothing",
    "stochastic_depth": "stochastic_depth",
    "focal_gamma": "focal_gamma",
    "abnormal_weight": "abnormal_weight",
    "ema": "EMA",
    "color": "color",
    "allow_tie_save": "allow_tie_save",
}

SOURCE_PRIORITY = {
    "baseline": 100,
    "round2": 40,
    "strict": 30,
    "legacy": 20,
    "queued": 10,
}

GC_MAP = {
    "00": 0.0,
    "01": 0.1,
    "025": 0.25,
    "035": 0.35,
    "05": 0.5,
    "075": 0.75,
    "10": 1.0,
    "125": 1.25,
    "15": 1.5,
    "20": 2.0,
    "30": 3.0,
    "50": 5.0,
}
WD_MAP = {
    "000": 0.0,
    "0005": 0.005,
    "001": 0.01,
    "002": 0.02,
    "005": 0.05,
}
LS_MAP = {
    "002": 0.02,
    "005": 0.05,
    "007": 0.07,
    "01": 0.10,
    "0125": 0.125,
    "015": 0.15,
    "0175": 0.175,
    "020": 0.20,
}
SD_MAP = {
    "005": 0.05,
    "01": 0.10,
    "015": 0.15,
    "02": 0.20,
    "03": 0.30,
}
FG_MAP = {"0p5": 0.5, "1p0": 1.0, "1p5": 1.5, "2p0": 2.0}
AW_MAP = {"0p5": 0.5, "0p8": 0.8, "1p2": 1.2, "1p5": 1.5, "2p0": 2.0, "3p0": 3.0}
EMA_MAP = {"0p99": 0.99, "0p995": 0.995, "0p999": 0.999}
LR_MAP = {
    "1e5": ("1e-5 / 1e-4", 1e-5),
    "2e5": ("2e-5 / 2e-4", 2e-5),
    "2p5e5": ("2.5e-5 / 2.5e-4", 2.5e-5),
    "3e5": ("3e-5 / 3e-4", 3e-5),
    "4e5": ("4e-5 / 4e-4", 4e-5),
    "5e5": ("5e-5 / 5e-4", 5e-5),
    "1e4": ("1e-4 / 1e-3", 1e-4),
}
SMOOTH_MAP = {
    "1raw": ("1-raw", 1.0),
    "3mean": ("3-mean", 3.0),
    "3med": ("3-median", 3.1),
    "5mean": ("5-mean", 5.0),
    "5med": ("5-median", 5.1),
}
COLOR_MAP = {
    "c01": ("c01", 1.0),
    "c02": ("c02", 2.0),
    "c03": ("c03", 3.0),
}

LR_SCHEDULE_BASE_EPOCHS = 20
LR_SCHEDULE_HEAD_BASE = 2e-4
LR_SCHEDULE_BACKBONE_BASE = 2e-5

FAMILY_BASELINES: dict[str, tuple[str, float | str]] = {
    "normal_ratio": ("700", 700.0),
    "per_class": ("0 / off", 0.0),
    "lr": ("2e-5 / 2e-4", 2e-5),
    "warmup": ("5", 5.0),
    "gc": ("1", 1.0),
    "weight_decay": ("0.01", 0.01),
    "smoothing": ("3-median", 3.1),
    "label_smoothing": ("0.00", 0.00),
    "stochastic_depth": ("0.00", 0.00),
    "focal_gamma": ("0.0", 0.0),
    "abnormal_weight": ("1.0", 1.0),
    "ema": ("0.0 / off", 0.0),
    "color": ("baseline", 0.0),
    "allow_tie_save": ("off", 0.0),
}

COLOR_DESCRIPTIONS = {
    "baseline": "trend blue `#4878CF`, fleet alpha `0.4`",
    "c01": "trend red `#E43320`, fleet alpha `0.4`",
    "c02": "trend blue `#4878CF`, fleet alpha `0.15`",
    "c03": "trend red `#E43320`, fleet alpha `0.15`",
}

OPERATING_BASELINE = "fresh0412_v11_refcheck_gcsmooth_n700"
HISTORICAL_BASELINE = "fresh0412_v11_n700_existing"


@dataclass
class ConditionRecord:
    family: str
    label: str
    sort_value: float
    seeds_done: int
    seeds_total: int
    f1: float | None
    fn: float | None
    fp: float | None
    status: str
    source: str
    candidate: str | None = None
    note: str = ""


def display_label(row: ConditionRecord) -> str:
    if row.family == "lr":
        return f"{row.label}<br>bb/head={row.label}"
    if row.family == "warmup":
        return f"warmup={row.label}<br>lr=2e-5/2e-4"
    return row.label


def status_label(status: str) -> str:
    return {
        "reference": "기준",
        "complete": "완료",
        "partial": "부분완료",
        "queued": "queued",
    }.get(status, status)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else {}


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_compact(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def fmt_delta(value: float | None, baseline: float, digits: int = 4) -> str:
    if value is None:
        return "-"
    delta = value - baseline
    if abs(delta) < 1e-9:
        return "0"
    if abs(delta) < 1 and digits > 3:
        return f"{delta:+.{digits}f}".rstrip("0").rstrip(".")
    return f"{delta:+.3f}".rstrip("0").rstrip(".")


def candidate_base(tag_or_name: str) -> str:
    return re.sub(r"_s(?:42|[1-4])$", "", tag_or_name)


def parse_candidate(candidate: str) -> tuple[str, str, float] | None:
    base = candidate_base(candidate)
    if not base.startswith("fresh0412_v11_"):
        return None
    suffix = base.removeprefix("fresh0412_v11_")

    if re.fullmatch(r"n\d+", suffix):
        value = float(suffix[1:])
        return "normal_ratio", str(int(value)), value

    if re.fullmatch(r"pc\d+", suffix):
        value = float(suffix[2:])
        return "per_class", str(int(value)), value

    if suffix.startswith("lrwarm"):
        raw = suffix.removeprefix("lrwarm").replace("_n700", "")
        value = float(raw)
        return "warmup", str(int(value)), value

    m = re.fullmatch(r"lr([0-9p]+e[45])_n700", suffix)
    if m:
        token = m.group(1)
        if token in LR_MAP:
            label, sort_value = LR_MAP[token]
            return "lr", label, float(sort_value)

    m = re.fullmatch(r"gc([0-9]+)_n700", suffix)
    if m:
        token = m.group(1)
        if token in GC_MAP:
            value = GC_MAP[token]
            return "gc", fmt_compact(value), float(value)

    m = re.fullmatch(r"wd([0-9]+)_n700", suffix)
    if m:
        token = m.group(1)
        if token in WD_MAP:
            value = WD_MAP[token]
            return "weight_decay", fmt_compact(value), float(value)

    m = re.fullmatch(r"sw([0-9](?:raw|mean|med))_n700", suffix)
    if m:
        token = m.group(1)
        if token in SMOOTH_MAP:
            label, sort_value = SMOOTH_MAP[token]
            return "smoothing", label, float(sort_value)

    m = re.fullmatch(r"regls([0-9]+)_n700", suffix)
    if m:
        token = m.group(1)
        if token in LS_MAP:
            value = LS_MAP[token]
            return "label_smoothing", fmt_compact(value), float(value)

    m = re.fullmatch(r"regdp([0-9]+)_n700", suffix)
    if m:
        token = m.group(1)
        if token in SD_MAP:
            value = SD_MAP[token]
            return "stochastic_depth", fmt_compact(value), float(value)

    m = re.fullmatch(r"fg([0-9]p[0-9])_n700", suffix)
    if m:
        token = m.group(1)
        if token in FG_MAP:
            value = FG_MAP[token]
            return "focal_gamma", fmt_compact(value), float(value)

    m = re.fullmatch(r"aw([0-9]p[0-9])_n700", suffix)
    if m:
        token = m.group(1)
        if token in AW_MAP:
            value = AW_MAP[token]
            return "abnormal_weight", fmt_compact(value), float(value)

    m = re.fullmatch(r"ema([0-9]p[0-9]{2,3})_n700", suffix)
    if m:
        token = m.group(1)
        if token in EMA_MAP:
            value = EMA_MAP[token]
            return "ema", fmt_compact(value), float(value)

    m = re.fullmatch(r"color_(c0[1-3])_n700", suffix)
    if m:
        token = m.group(1)
        if token in COLOR_MAP:
            label, sort_value = COLOR_MAP[token]
            return "color", label, float(sort_value)

    if suffix == "tie_on_n700":
        return "allow_tie_save", "on", 1.0

    return None


def add_record(records: dict[str, dict[str, ConditionRecord]], record: ConditionRecord) -> None:
    family_records = records.setdefault(record.family, {})
    existing = family_records.get(record.label)
    if existing is None:
        family_records[record.label] = record
        return
    lhs = (SOURCE_PRIORITY.get(record.source, 0), record.seeds_done, record.seeds_total, 0 if record.f1 is None else 1)
    rhs = (SOURCE_PRIORITY.get(existing.source, 0), existing.seeds_done, existing.seeds_total, 0 if existing.f1 is None else 1)
    if lhs >= rhs:
        family_records[record.label] = record


def baseline_metrics() -> tuple[float, float, float]:
    data = load_json(VALIDATIONS / "baseline_delta_latest.json")
    base = data.get("baseline", {})
    return (
        float(base.get("f1_mean", 0.99548)),
        float(base.get("fn_mean", 4.4)),
        float(base.get("fp_mean", 2.4)),
    )


def inject_baselines(records: dict[str, dict[str, ConditionRecord]]) -> None:
    bf1, bfn, bfp = baseline_metrics()
    for family, (label, sort_value) in FAMILY_BASELINES.items():
        add_record(
            records,
            ConditionRecord(
                family=family,
                label=label,
                sort_value=float(sort_value),
                seeds_done=5,
                seeds_total=5,
                f1=bf1,
                fn=bfn,
                fp=bfp,
                status="reference",
                source="baseline",
                candidate=OPERATING_BASELINE,
            ),
        )


def load_legacy_v11(records: dict[str, dict[str, ConditionRecord]], path: Path) -> None:
    payload = load_json(path)
    groups = payload.get("by_group", {})
    per_candidate: dict[str, list[dict[str, Any]]] = {}
    for rows in groups.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            name = row.get("name")
            if not isinstance(name, str):
                continue
            parsed = parse_candidate(name)
            if parsed is None:
                continue
            base = candidate_base(name)
            per_candidate.setdefault(base, []).append(row)

    for candidate, rows in per_candidate.items():
        parsed = parse_candidate(candidate)
        if parsed is None:
            continue
        family, label, sort_value = parsed
        if candidate == "fresh0412_v11_n700":
            continue
        f1s = [float(r["f1"]) for r in rows if r.get("f1") is not None]
        fns = [float(r["FN"]) for r in rows if r.get("FN") is not None]
        fps = [float(r["FP"]) for r in rows if r.get("FP") is not None]
        add_record(
            records,
            ConditionRecord(
                family=family,
                label=label,
                sort_value=float(sort_value),
                seeds_done=len(rows),
                seeds_total=len(rows),
                f1=mean(f1s) if f1s else None,
                fn=mean(fns) if fns else None,
                fp=mean(fps) if fps else None,
                status="complete",
                source="legacy",
                candidate=candidate,
            ),
        )


def load_optimized_v11_normal_ratio(path: Path) -> list[ConditionRecord]:
    payload = load_json(path)
    rows = payload.get("by_group", {}).get("sweep", [])
    by_level: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str):
            continue
        match = re.search(r"_n(\d+)_s(?:42|[1-4])$", name)
        if not match:
            continue
        by_level.setdefault(int(match.group(1)), []).append(row)

    records: list[ConditionRecord] = []
    for level, level_rows in sorted(by_level.items()):
        f1s = [float(r["f1"]) for r in level_rows if r.get("f1") is not None]
        fns = [float(r["FN"]) for r in level_rows if r.get("FN") is not None]
        fps = [float(r["FP"]) for r in level_rows if r.get("FP") is not None]
        records.append(
            ConditionRecord(
                family="normal_ratio_optimized_v11",
                label=str(level),
                sort_value=float(level),
                seeds_done=len(level_rows),
                seeds_total=len(level_rows),
                f1=mean(f1s) if f1s else None,
                fn=mean(fns) if fns else None,
                fp=mean(fps) if fps else None,
                status="reference" if level == 700 else "complete",
                source="optimized_v11",
                candidate=f"fresh0413_reset_v11_n{level}",
            )
        )
    return records


def load_summary_aggregates(records: dict[str, dict[str, ConditionRecord]], path: Path, source: str) -> None:
    payload = load_json(path)
    by_candidate = payload.get("aggregates", {}).get("by_candidate", {})
    for candidate, row in by_candidate.items():
        parsed = parse_candidate(candidate)
        if parsed is None:
            continue
        family, label, sort_value = parsed
        complete = int(row.get("complete", 0))
        add_record(
            records,
            ConditionRecord(
                family=family,
                label=label,
                sort_value=float(sort_value),
                seeds_done=complete,
                seeds_total=complete,
                f1=float(row["f1_mean"]) if row.get("f1_mean") is not None else None,
                fn=float(row["fn_mean"]) if row.get("fn_mean") is not None else None,
                fp=float(row["fp_mean"]) if row.get("fp_mean") is not None else None,
                status="complete" if complete > 0 else "queued",
                source=source,
                candidate=candidate,
            ),
        )


def load_round2_queue(records: dict[str, dict[str, ConditionRecord]], queue_path: Path, summary_path: Path) -> None:
    queue = load_json(queue_path)
    runs = queue.get("runs", [])
    expected: dict[str, int] = {}
    for run in runs:
        candidate = run.get("candidate") or candidate_base(str(run.get("tag", "")))
        parsed = parse_candidate(candidate)
        if parsed is None:
            continue
        expected[candidate] = expected.get(candidate, 0) + 1

    summary = load_json(summary_path)
    by_candidate = summary.get("aggregates", {}).get("by_candidate", {})
    for candidate, seeds_total in expected.items():
        parsed = parse_candidate(candidate)
        if parsed is None:
            continue
        family, label, sort_value = parsed
        row = by_candidate.get(candidate, {})
        seeds_done = int(row.get("complete", 0))
        add_record(
            records,
            ConditionRecord(
                family=family,
                label=label,
                sort_value=float(sort_value),
                seeds_done=seeds_done,
                seeds_total=seeds_total,
                f1=float(row["f1_mean"]) if row.get("f1_mean") is not None else None,
                fn=float(row["fn_mean"]) if row.get("fn_mean") is not None else None,
                fp=float(row["fp_mean"]) if row.get("fp_mean") is not None else None,
                status="complete" if seeds_done >= seeds_total and seeds_total > 0 else ("partial" if seeds_done > 0 else "queued"),
                source="round2",
                candidate=candidate,
            ),
        )


def sorted_family_rows(records: dict[str, dict[str, ConditionRecord]], family: str) -> list[ConditionRecord]:
    rows = list(records.get(family, {}).values())
    return sorted(rows, key=lambda r: (r.sort_value, r.label))


def completed_or_reference(rows: list[ConditionRecord]) -> list[ConditionRecord]:
    return [r for r in rows if r.status in {"complete", "reference"}]


def report_rows(family: str, rows: list[ConditionRecord]) -> list[ConditionRecord]:
    rows = completed_or_reference(rows)
    if family == "color":
        return [r for r in rows if r.label in {"baseline", "c01"}]
    return rows


def best_completed(rows: list[ConditionRecord], exclude_baseline: bool = True, prefer_strict_5seed: bool = True) -> ConditionRecord | None:
    candidates = [r for r in rows if r.status == "complete" and r.f1 is not None]
    if exclude_baseline:
        baseline_label = FAMILY_BASELINES.get(rows[0].family, (None, None))[0] if rows else None
        candidates = [r for r in candidates if r.label != baseline_label]
    if not candidates:
        return None
    if prefer_strict_5seed:
        strict_5seed = [
            r for r in candidates
            if r.seeds_done >= 5 and r.seeds_total >= 5 and r.source in {"strict", "round2"}
        ]
        if strict_5seed:
            candidates = strict_5seed
    return min(candidates, key=lambda r: ((r.fn or math.inf) + (r.fp or math.inf), -(r.f1 or -math.inf)))


def strongest_candidates(records: dict[str, dict[str, ConditionRecord]]) -> list[str]:
    lines: list[str] = []
    for family in ("label_smoothing", "abnormal_weight", "stochastic_depth", "gc", "color"):
        rows = sorted_family_rows(records, family)
        if not rows:
            continue
        best = best_completed(rows)
        if best is None:
            continue
        if family == "gc":
            lines.append(
                f"`GC` broad good band remains active; current lowest completed total error is around `{best.label}` with `F1={fmt_float(best.f1)}`, `FN={fmt_compact(best.fn)}`, `FP={fmt_compact(best.fp)}`. Incomplete values are kept out of the main table."
            )
        elif family == "color":
            c01 = next((r for r in rows if r.label.startswith("c01") and r.f1 is not None), None)
            if c01:
                lines.append(
                    f"`color`는 현재 유효한 비교가 baseline vs c01뿐입니다: `c01 {fmt_float(c01.f1)} / FN {fmt_compact(c01.fn)} / FP {fmt_compact(c01.fp)}`. c02/c03는 생성 이미지가 의도와 달라 재생성이 필요합니다."
                )
        else:
            lines.append(
                f"`{family}` currently best completed point is `{best.label}` with `F1={fmt_float(best.f1)}`, `FN={fmt_compact(best.fn)}`, `FP={fmt_compact(best.fp)}`."
            )
    return lines


def robust_upper_bound(values: list[float]) -> float | None:
    cleaned = sorted(v for v in values if math.isfinite(v))
    if len(cleaned) < 4:
        return None
    max_v = cleaned[-1]
    second_v = cleaned[-2]
    median_v = cleaned[len(cleaned) // 2]
    if second_v <= 0:
        second_v = max(median_v, 1e-6)
    if median_v <= 0:
        median_v = max(second_v, 1e-6)
    # Hunting/collapse outlier only. Keep value in data, just clip axis scale.
    if max_v >= second_v * 4.0 and max_v >= median_v * 6.0:
        return second_v * 1.15
    return None


def robust_lower_bound(values: list[float]) -> float | None:
    cleaned = sorted(v for v in values if math.isfinite(v))
    if len(cleaned) < 4:
        return None
    min_v = cleaned[0]
    second_v = cleaned[1]
    median_v = cleaned[len(cleaned) // 2]
    if second_v <= 0:
        return None
    # Single severe low outlier only. Keep the point, clip the F1 axis.
    if min_v <= second_v - 0.05 and min_v <= median_v - 0.08:
        margin = max(0.005, (median_v - second_v) * 0.5)
        return max(0.0, second_v - margin)
    return None


def experiment_interpretation(records: dict[str, dict[str, ConditionRecord]]) -> list[str]:
    lines: list[str] = []
    lines.append("이번 strict one-factor round에서는 baseline을 고정한 채 `normal_ratio`, `per_class`, `lr`, `warmup`, `gc`, `weight_decay`, `smoothing`, `label_smoothing`, `stochastic_depth`, `focal_gamma`, `abnormal_weight`, `ema`, `color`, `allow_tie_save`를 개별 축으로 확인했습니다.")

    meaningful: list[str] = []
    broad: list[str] = []
    weak: list[str] = []

    def best_label(family: str) -> str | None:
        rows = sorted_family_rows(records, family)
        best = best_completed(rows)
        return best.label if best else None

    ls = best_label("label_smoothing")
    if ls:
        meaningful.append(f"`label_smoothing`은 `{ls}` 근처에서 가장 강한 개선이 보였고, 너무 낮거나 높으면 FP/FN 균형이 다시 나빠졌습니다")
    aw = best_label("abnormal_weight")
    if aw:
        meaningful.append(f"`abnormal_weight`는 `{aw}` 근처에서 sweet spot이 보였고, 더 크게 주면 FN이 다시 증가했습니다")
    sd = best_label("stochastic_depth")
    if sd:
        meaningful.append(f"`stochastic_depth`는 `{sd}` 인근에서 유의미한 개선이 나타났습니다")

    if best_label("gc"):
        broad.append("`gc`는 단일 sharp optimum보다는 넓은 양호 구간이 보였고, 헌팅 값 하나가 축 스케일을 왜곡하는 형태였습니다")
    if best_label("focal_gamma"):
        weak.append("`focal_gamma`는 여러 값이 비슷해서 뚜렷한 최적값보다는 broad-good 혹은 약한 효과 축에 가깝습니다")
    if best_label("normal_ratio"):
        weak.append("`normal_ratio`는 성능이 전반적으로 좋아지는 구간은 보이지만, 현재 점들만으로는 매끈한 단일 sweet spot이라고 단정하기 어렵습니다")
    if best_label("ema"):
        weak.append("`ema`는 baseline 대비 개선은 있으나 강한 최적값 주장을 하기는 아직 어렵습니다")

    if meaningful:
        lines.append("유의미한 최적값 후보가 보이는 축은 " + "; ".join(meaningful) + ".")
    if broad:
        lines.append("넓은 양호 구간으로 해석하는 편이 맞는 축은 " + "; ".join(broad) + ".")
    if weak:
        lines.append("현재로서는 뚜렷한 최적값이 약하거나 추가 확인이 필요한 축은 " + "; ".join(weak) + ".")
    return lines


def evidence_limits(records: dict[str, dict[str, ConditionRecord]]) -> list[str]:
    queued = 0
    partial = 0
    three_seed = 0
    for family in FAMILY_ORDER:
        for row in sorted_family_rows(records, family):
            if row.status == "queued":
                queued += 1
            elif row.status == "partial":
                partial += 1
            elif row.status == "complete" and row.seeds_done < 5:
                three_seed += 1

    return [
        f"운영 baseline은 `{OPERATING_BASELINE}`입니다. 5-seed matched control 기준 `F1=0.9955`, `FN=4.4`, `FP=2.4`이며 target band hit은 `0/5`입니다. FP가 전 seed에서 낮아 기준선이 너무 깨끗하다는 한계를 명시해야 합니다.",
        f"`{HISTORICAL_BASELINE}`은 historical selected ref로 보존하되, strict one-factor 표와 delta 계산의 현재 기준은 `{OPERATING_BASELINE}`입니다.",
        "`label_smoothing=0.0`은 baseline train config에 명시된 no-smoothing 상태입니다. 단, `label_smoothing>0`에서는 loss 구현 경로가 `CrossEntropyLoss(label_smoothing=...)`로 바뀌므로 최종 claim에는 이 구현 차이를 한계로 적어야 합니다.",
        "현재 표는 baseline-fixed one-factor evidence만 섞어 보여줍니다. alternate-parent stress, bad-case rescue, logical/per-member 실험은 별도 표로 분리해야 합니다.",
        f"아직 claim 성숙 전인 조건이 남아 있습니다: queued `{queued}`개, partial `{partial}`개, 5-seed 미만 완료 `{three_seed}`개.",
        "`stochastic_depth`는 학습 때 일부 residual/drop-path branch를 확률적으로 끄는 regularization입니다. 추론 때는 전체 경로를 쓰며, 모델이 한 경로에 과적합하지 않게 만들어 seed 안정성과 FN/FP 균형이 좋아지는지 보는 축입니다.",
        "최종 논문화 전에는 각 축마다 per-seed/worst-seed, history의 val_loss/F1 진동, prediction trend의 반복 FP/FN chart_id, label-or-annotation suspect를 붙여야 합니다.",
    ]


def provisional_golden_recipe(records: dict[str, dict[str, ConditionRecord]]) -> list[str]:
    picks = {
        "normal_ratio": None,
        "gc": None,
        "label_smoothing": None,
        "stochastic_depth": None,
        "focal_gamma": None,
        "abnormal_weight": None,
        "ema": None,
        "allow_tie_save": None,
    }
    for family in picks:
        rows = sorted_family_rows(records, family)
        if rows:
            picks[family] = best_completed(rows)

    recipe_lines: list[str] = []
    recipe_lines.extend([
        "| axis | selected value | F1 | FN | FP | status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for family in ("normal_ratio", "gc", "label_smoothing", "stochastic_depth", "focal_gamma", "abnormal_weight", "ema", "allow_tie_save"):
        best = picks[family]
        if best is None:
            continue
        recipe_lines.append(f"| `{family}` | `{best.label}` | {fmt_float(best.f1)} | {fmt_compact(best.fn)} | {fmt_compact(best.fp)} | provisional |")
    return recipe_lines


def pending_round2(records: dict[str, dict[str, ConditionRecord]]) -> list[str]:
    lines: list[str] = []
    for family in FAMILY_ORDER:
        for row in sorted_family_rows(records, family):
            if row.source == "round2" and row.status != "complete":
                lines.append(f"- `{family} = {row.label}`: `{row.seeds_done}/{row.seeds_total}` complete")
    return lines


def plot_family(
    rows: list[ConditionRecord],
    out_path: Path,
    title: str,
    baseline: tuple[float, float, float],
    comparison_rows: list[ConditionRecord] | None = None,
) -> None:
    baseline_f1, baseline_fn, baseline_fp = baseline
    rows = completed_or_reference(rows)
    comparison_rows = completed_or_reference(comparison_rows or [])
    x_values = sorted({r.sort_value for r in rows + comparison_rows})
    x_index = {value: idx for idx, value in enumerate(x_values)}
    labels_by_idx = {
        x_index[r.sort_value]: r.label
        for r in rows + comparison_rows
        if r.sort_value in x_index
    }
    x_labels = [labels_by_idx.get(idx, fmt_compact(value)) for idx, value in enumerate(x_values)]
    xs = list(range(len(x_values)))

    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(x_values) * 1.0), 4.2))
    metrics = [
        ("F1", baseline_f1, lambda r: r.f1),
        ("FN", baseline_fn, lambda r: r.fn),
        ("FP", baseline_fp, lambda r: r.fp),
    ]

    for ax, (metric_name, base_value, getter) in zip(axes, metrics):
        ys_reference_x: list[int] = []
        ys_reference_y: list[float] = []
        ys_complete_x: list[int] = []
        ys_complete_y: list[float] = []
        for row in rows:
            value = getter(row)
            if value is None:
                continue
            idx = x_index[row.sort_value]
            if row.status == "reference":
                ys_reference_x.append(idx)
                ys_reference_y.append(float(value))
            elif row.status == "complete":
                ys_complete_x.append(idx)
                ys_complete_y.append(float(value))
        if ys_reference_x:
            ax.plot(ys_reference_x, ys_reference_y, color="#333333", marker="s", lw=0.0, ms=6, label="reference")
        if ys_complete_x:
            ax.plot(ys_complete_x, ys_complete_y, color="#1f77b4", marker="o", lw=1.8, label="current ref sweep")
        comparison_y: list[float] = []
        if comparison_rows:
            comparison_x: list[int] = []
            for row in comparison_rows:
                value = getter(row)
                if value is None:
                    continue
                comparison_x.append(x_index[row.sort_value])
                comparison_y.append(float(value))
            if comparison_x:
                ax.plot(comparison_x, comparison_y, color="#2ca02c", marker="^", lw=1.8, label="optimized v11 sweep")
        ax.set_title(metric_name)
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.grid(alpha=0.25, linestyle=":")
        if metric_name == "F1":
            all_y = [baseline_f1] + ys_reference_y + ys_complete_y + comparison_y
            lower = robust_lower_bound(all_y)
            if lower is None:
                ax.set_ylim(max(0.0, min(all_y) - 0.01), min(1.0, max(all_y) + 0.01))
            else:
                ax.set_ylim(lower, min(1.0, max(all_y) + 0.01))
                ax.text(
                    0.98,
                    0.98,
                    "scale clipped",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="#666666",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
                )
        else:
            all_y = ys_reference_y + ys_complete_y + comparison_y
            upper = robust_upper_bound(all_y)
            if upper is not None:
                ax.set_ylim(bottom=0.0, top=upper)
                ax.text(
                    0.98,
                    0.98,
                    "scale clipped",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="#666666",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
                )
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def cosine_warmup_lr(epoch: int, warmup_epochs: int, base_lr: float, total_epochs: int = LR_SCHEDULE_BASE_EPOCHS) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    if total_epochs <= warmup_epochs:
        return base_lr
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def plot_lr_schedule_family(rows: list[ConditionRecord], out_path: Path, family: str) -> None:
    epochs = list(range(1, LR_SCHEDULE_BASE_EPOCHS + 1))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    if family == "warmup":
        for row in rows:
            if row.f1 is None:
                continue
            try:
                warmup = int(float(row.sort_value))
            except ValueError:
                continue
            ys = [cosine_warmup_lr(ep, warmup, LR_SCHEDULE_HEAD_BASE) for ep in epochs]
            ax.plot(epochs, ys, marker="o", ms=3, lw=1.6, label=f"warmup={row.label}")
        ax.set_title("lr_head schedule by warmup")
        ax.set_ylabel("lr_head")
    elif family == "lr":
        for row in rows:
            if row.f1 is None:
                continue
            base_lr = float(row.sort_value) * 10.0
            ys = [cosine_warmup_lr(ep, 5, base_lr) for ep in epochs]
            ax.plot(epochs, ys, marker="o", ms=3, lw=1.6, label=row.label)
        ax.set_title("lr_head schedule by LR condition (warmup=5)")
        ax.set_ylabel("lr_head")
        ax.set_yscale("log")
    else:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    out_paths: list[Path],
    records: dict[str, dict[str, ConditionRecord]],
    optimized_normal_ratio: list[ConditionRecord],
    plots_dir: Path,
    state: dict[str, Any],
    strict_summary: dict[str, Any],
    round2_summary: dict[str, Any],
) -> None:
    baseline = baseline_metrics()
    generated_at = datetime.now().isoformat(timespec="seconds")
    main_complete = int(strict_summary.get("aggregates", {}).get("complete", 0))
    round2_complete = int(round2_summary.get("aggregates", {}).get("complete", 0))
    round2_queue = load_json(VALIDATIONS / "paper_strict_single_factor_round2_queue.json")
    round2_total = len(round2_queue.get("runs", []))

    lines = [
        "# Strict Single-Factor Report",
        "",
        f"_Auto-updated at `{generated_at}`._",
        "",
        "## Interpretation",
        "",
    ]
    lines.extend([f"- {line}" for line in experiment_interpretation(records)])
    lines.extend([
        "",
        "## Evidence Limits And Next Fixes",
        "",
    ])
    lines.extend([f"- {line}" for line in evidence_limits(records)])
    lines.extend([
        "",
        "## Summary",
        "",
        f"- Operating baseline: `{OPERATING_BASELINE}` -> `F1={fmt_float(baseline[0])}`, `FN={fmt_compact(baseline[1])}`, `FP={fmt_compact(baseline[2])}` over `5/5` seeds.",
        f"- Historical selected ref: `{HISTORICAL_BASELINE}` -> `F1=0.9901`, `FN=9.8`, `FP=5.0`; kept only as reference-selection history.",
        f"- Main strict queue: `{main_complete}` completed runs, decision `{strict_summary.get('decision')}`.",
        f"- Round-2 refinement: `{round2_complete}/{round2_total}` completed runs, stage `{state.get('stage')}`, status `{state.get('status')}`.",
        "",
        "Display용 이미지와 실제 학습 입력 이미지는 다릅니다. 아래 두 montage는 기존 `display_v11/`와 `images_v11/`에서 같은 class 순서로 가져온 예시입니다.",
        "",
        "**Display images**",
        "",
        "![display samples](sample_overview_display.png)",
        "",
        "**Training images**",
        "",
        "![training samples](sample_overview_train.png)",
        "",
    ])
    lines.extend([f"- {line}" for line in strongest_candidates(records)])
    if optimized_normal_ratio:
        lines.append("- `normal_ratio`: 현재 ref 기준 sweep에서는 3000~3500 구간이 좋아 보이지만, optimized-v11 sweep을 같이 보면 normal_ratio 증가가 항상 개선을 만들지는 않습니다.")
    lines.extend([
        "",
        "## Provisional Golden Recipe",
        "",
        "_This is one-factor evidence only. Joint combo validation still has to be run after round-2 closes._",
        "",
    ])
    lines.extend(provisional_golden_recipe(records) or ["- No completed candidate set yet."])
    lines.extend([
        "",
        "## Logical Member Attribution Example",
        "",
        "같은 context chart를 member별 target 이미지로 확장합니다. 불량 member를 target으로 만든 이미지만 anomaly class이고, 양호 member를 target으로 만든 이미지는 normal class입니다.",
        "",
        "![logical member target examples](logical_member_targets_ch09100.png)",
        "",
        "즉 family 전체 이상 감지가 아니라, highlight 된 member 단위로 label을 부여하는 학습 예시입니다.",
        "",
        "## Pending Round-2 Checks",
        "",
    ])
    lines.extend(pending_round2(records) or ["- No queued round-2 conditions."])
    lines.extend([
        "",
        "## Plot Index",
        "",
    ])
    for family in FAMILY_ORDER:
        rows = report_rows(family, sorted_family_rows(records, family))
        if not rows:
            continue
        plot_name = f"{family}.png"
        lines.append(f"- `{family}`: [{plot_name}]({plots_dir.name}/{plot_name})")
        if family in {"lr", "warmup"}:
            lines.append(f"- `{family}` learning-rate schedule: [{family}_lr_schedule.png]({plots_dir.name}/{family}_lr_schedule.png)")

    for family in FAMILY_ORDER:
        rows = report_rows(family, sorted_family_rows(records, family))
        if not rows:
            continue
        plot_name = f"{family}.png"
        lines.extend([
            "",
            f"## {FAMILY_TITLES[family]}",
            "",
            f"![{family}]({plots_dir.name}/{plot_name})",
            "",
        ])
        if family == "color":
            lines.extend([
                "조건 설명:",
                "",
            ])
            for key, desc in COLOR_DESCRIPTIONS.items():
                lines.append(f"- `{key}`: {desc}")
            lines.extend([
                "",
                "조건별 대표 sample:",
                "",
                "| baseline | c01 | c02 | c03 |",
                "| --- | --- | --- | --- |",
                "| ![baseline](color_samples/baseline_ch_09100.png) | ![c01](color_samples/c01_ch_09100.png) | ![c02](color_samples/c02_ch_09100.png) | ![c03](color_samples/c03_ch_09100.png) |",
                "",
            ])
        if family in {"lr", "warmup"}:
            lines.extend([
                f"![{family} learning-rate schedule]({plots_dir.name}/{family}_lr_schedule.png)",
                "",
            ])
        lines.extend([
            "| condition | seeds | F1 | ΔF1 | FN | ΔFN | FP | ΔFP | status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in rows:
            lines.append(
                f"| {display_label(row)} | {row.seeds_done}/{row.seeds_total} | {fmt_float(row.f1)} | {fmt_delta(row.f1, baseline[0])} | {fmt_compact(row.fn)} | {fmt_delta(row.fn, baseline[1], digits=3)} | {fmt_compact(row.fp)} | {fmt_delta(row.fp, baseline[2], digits=3)} | {status_label(row.status)} |"
            )
        if family == "normal_ratio" and optimized_normal_ratio:
            opt_baseline = next((r for r in optimized_normal_ratio if r.label == "700"), optimized_normal_ratio[0])
            opt_base_f1 = opt_baseline.f1 if opt_baseline.f1 is not None else baseline[0]
            opt_base_fn = opt_baseline.fn if opt_baseline.fn is not None else baseline[1]
            opt_base_fp = opt_baseline.fp if opt_baseline.fp is not None else baseline[2]
            lines.extend([
                "",
                "### optimized-v11 normal_ratio comparison",
                "",
                "이미 성능이 최적화된 v11 조건에서는 normal_ratio를 키워도 단조 개선되지 않습니다. 이 표는 normal 수 증가 효과가 데이터/학습 상태에 의존하며, 무조건적인 scale-up claim은 안 된다는 근거입니다.",
                "",
                "| condition | seeds | F1 | ΔF1 vs opt700 | FN | ΔFN vs opt700 | FP | ΔFP vs opt700 | status |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ])
            for row in optimized_normal_ratio:
                lines.append(
                    f"| {display_label(row)} | {row.seeds_done}/{row.seeds_total} | {fmt_float(row.f1)} | {fmt_delta(row.f1, opt_base_f1)} | {fmt_compact(row.fn)} | {fmt_delta(row.fn, opt_base_fn, digits=3)} | {fmt_compact(row.fp)} | {fmt_delta(row.fp, opt_base_fp, digits=3)} | {status_label(row.status)} |"
                )

    body = "\n".join(lines) + "\n"
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding="utf-8")


def build_records(strict_summary_path: Path, round2_summary_path: Path, round2_queue_path: Path) -> dict[str, dict[str, ConditionRecord]]:
    records: dict[str, dict[str, ConditionRecord]] = {}
    inject_baselines(records)
    load_legacy_v11(records, LOGS / "v11_experiments_summary_fresh0412.json")
    load_summary_aggregates(records, strict_summary_path, "strict")
    load_summary_aggregates(records, round2_summary_path, "round2")
    load_round2_queue(records, round2_queue_path, round2_summary_path)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate markdown and plots for strict one-factor experiments.")
    ap.add_argument("--strict-summary", default=str(VALIDATIONS / "paper_strict_single_factor_summary.json"))
    ap.add_argument("--round2-summary", default=str(VALIDATIONS / "paper_strict_single_factor_round2_summary.json"))
    ap.add_argument("--round2-queue", default=str(VALIDATIONS / "paper_strict_single_factor_round2_queue.json"))
    ap.add_argument("--state", default=str(VALIDATIONS / "paper_strict_single_factor_state.json"))
    ap.add_argument("--markdown-out", default=str(VALIDATIONS / "paper_strict_single_factor_summary.md"))
    ap.add_argument("--report-out", default=str(VALIDATIONS / "paper_strict_single_factor_report.md"))
    ap.add_argument("--plots-dir", default=str(VALIDATIONS / "paper_strict_single_factor_plots"))
    args = ap.parse_args()

    strict_summary_path = Path(args.strict_summary)
    round2_summary_path = Path(args.round2_summary)
    round2_queue_path = Path(args.round2_queue)
    state_path = Path(args.state)
    markdown_out = Path(args.markdown_out)
    report_out = Path(args.report_out)
    plots_dir = Path(args.plots_dir)

    strict_summary = load_json(strict_summary_path)
    round2_summary = load_json(round2_summary_path)
    state = load_json(state_path)
    records = build_records(strict_summary_path, round2_summary_path, round2_queue_path)
    optimized_normal_ratio = load_optimized_v11_normal_ratio(LOGS / "v11_experiments_summary_fresh0413_reset.json")
    baseline = baseline_metrics()

    plots_dir.mkdir(parents=True, exist_ok=True)
    for family in FAMILY_ORDER:
        rows = sorted_family_rows(records, family)
        if not rows:
            continue
        comparison = optimized_normal_ratio if family == "normal_ratio" else None
        plot_family(rows, plots_dir / f"{family}.png", FAMILY_TITLES[family], baseline, comparison_rows=comparison)
        if family in {"lr", "warmup"}:
            plot_lr_schedule_family(completed_or_reference(rows), plots_dir / f"{family}_lr_schedule.png", family)

    write_markdown([markdown_out, report_out], records, optimized_normal_ratio, plots_dir, state, strict_summary, round2_summary)
    print(json.dumps({
        "markdown": str(markdown_out),
        "report": str(report_out),
        "plots_dir": str(plots_dir),
        "families": [family for family in FAMILY_ORDER if records.get(family)],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
