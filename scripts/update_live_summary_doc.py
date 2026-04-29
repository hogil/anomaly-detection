#!/usr/bin/env python
"""Update the live status sections in docs/summary.md from controller artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt_num(value: Any, digits: int = 4) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return ""


def latest_run(summary: dict[str, Any]) -> dict[str, Any] | None:
    runs = [r for r in summary.get("runs", {}).values() if r.get("status") == "complete"]
    if not runs:
        return None
    return max(runs, key=lambda r: str(r.get("completed_at", "")))


def active_log(logs_dir: Path) -> str:
    if not logs_dir.exists():
        return ""
    candidates = []
    for path in logs_dir.glob("**/*"):
        if not path.is_dir() or path == logs_dir:
            continue
        name = path.name
        if "_rawbase_" not in name and "_lossfilter_" not in name:
            continue
        if re.search(r"_F\d", name):
            continue
        candidates.append(path)
    if not candidates:
        return ""
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.name


def candidate_lines(rawbase: dict[str, Any]) -> list[str]:
    by_candidate = rawbase.get("aggregates", {}).get("by_candidate", {})
    if not isinstance(by_candidate, dict) or not by_candidate:
        return ["- rawbase completed candidates: none yet."]
    lines = ["- rawbase 현재 완료 후보:"]
    for name, row in sorted(by_candidate.items()):
        lines.append(
            f"  - `{name}`: `{row.get('complete', 0)}/5`, "
            f"F1 `{fmt_num(row.get('f1_mean'))}`, FN `{fmt_num(row.get('fn_mean'), 3)}`, FP `{fmt_num(row.get('fp_mean'), 3)}`"
        )
    return lines


def delta(value: Any, baseline: Any, digits: int = 4) -> str:
    if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)):
        return ""
    diff = value - baseline
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.{digits}f}".rstrip("0").rstrip(".")


def copy_live_plots(args: argparse.Namespace) -> list[tuple[str, str]]:
    args.docs_plots_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("candidate F1", "log_history_report_rawbase_candidate_f1.png", "rawbase_live_candidate_f1.png"),
        ("val F1 curves", "log_history_report_rawbase_val_f1_curves.png", "rawbase_live_val_f1_curves.png"),
        ("grad p99 curves", "log_history_report_rawbase_grad_p99_curves.png", "rawbase_live_grad_p99_curves.png"),
    ]
    out: list[tuple[str, str]] = []
    for label, src_name, dst_name in plots:
        src = args.validations_dir / src_name
        if not src.exists():
            continue
        dst = args.docs_plots_dir / dst_name
        shutil.copy2(src, dst)
        out.append((label, f"plots/{dst_name}"))
    return out


def rawbase_live_sections(args: argparse.Namespace) -> list[str]:
    raw_ref = read_json(args.raw_ref_summary)
    rawbase = read_json(args.rawbase_summary)
    candidate_csv = read_csv(args.live_candidates_csv)
    run_csv = read_csv(args.live_runs_csv)
    raw_agg = raw_ref.get("aggregates", {})
    raw_f1 = raw_agg.get("f1_mean")
    raw_fn = raw_agg.get("fn_mean")
    raw_fp = raw_agg.get("fp_mean")

    lines = [
        "## Rawbase Live Tables And Plots",
        "",
        "이 아래 표와 plot은 active raw baseline `fresh0412_v11_refcheck_raw_n700`을 ref/baseline으로 둡니다. 예전 gcsmooth 기준 상세 축 표는 current performance 표에서 제거했습니다.",
        "",
    ]
    plots = copy_live_plots(args)
    if plots:
        lines.extend(["### Live Plots", ""])
        for label, rel in plots:
            lines.append(f"- `{label}`: [{Path(rel).name}]({rel})")
        lines.append("")
        for label, rel in plots:
            lines.append(f"![{label}]({rel})")
            lines.append("")
    else:
        lines.extend([
            "### Live Plots",
            "",
            "- 아직 rawbase live plot artifact가 없습니다. 먼저 `python scripts/generate_log_history_report.py --logs-dir logs --out-prefix validations/log_history_report_rawbase --contains rawbase --top-k 30`를 실행합니다.",
            "",
        ])

    lines.extend([
        "### Candidate Table",
        "",
        "| candidate | seeds | F1 | dF1 vs raw ref | FN | dFN vs raw ref | FP | dFP vs raw ref | source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    if raw_agg:
        lines.append(
            f"| `fresh0412_v11_refcheck_raw_n700` | {raw_agg.get('complete', 0)}/5 | {fmt_num(raw_f1)} | 0 | {fmt_num(raw_fn, 3)} | 0 | {fmt_num(raw_fp, 3)} | 0 | raw ref |"
        )
    if candidate_csv:
        for row in candidate_csv:
            name = row.get("candidate", "")
            f1 = float(row["f1_mean"]) if row.get("f1_mean") else None
            fn = float(row["fn_mean"]) if row.get("fn_mean") else None
            fp = float(row["fp_mean"]) if row.get("fp_mean") else None
            complete = row.get("runs", "0")
            lines.append(
                f"| `{name}` | {complete}/5 | {fmt_num(f1)} | {delta(f1, raw_f1)} | {fmt_num(fn, 3)} | {delta(fn, raw_fn, 3)} | {fmt_num(fp, 3)} | {delta(fp, raw_fp, 3)} | rawbase history |"
            )
    else:
        by_candidate = rawbase.get("aggregates", {}).get("by_candidate", {})
        for name, row in sorted(by_candidate.items()):
            f1 = row.get("f1_mean")
            fn = row.get("fn_mean")
            fp = row.get("fp_mean")
            complete = row.get("complete", 0)
            lines.append(
                f"| `{name}` | {complete}/5 | {fmt_num(f1)} | {delta(f1, raw_f1)} | {fmt_num(fn, 3)} | {delta(fn, raw_fn, 3)} | {fmt_num(fp, 3)} | {delta(fp, raw_fp, 3)} | rawbase live |"
            )
    lines.append("")

    rows = [r for r in rawbase.get("runs", {}).values() if r.get("status") == "complete"]
    rows = sorted(rows, key=lambda r: str(r.get("completed_at", "")), reverse=True)
    lines.extend([
        "### Recent Runs",
        "",
        "| tag | seed | F1 | FN | FP | epoch | run dir |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    if run_csv:
        for row in reversed(run_csv[-20:]):
            tag = Path(row.get("run_dir", "")).name
            f1 = float(row["best_info_f1"]) if row.get("best_info_f1") else None
            lines.append(
                f"| `{tag}` | {row.get('seed', '')} | {fmt_num(f1)} | {row.get('fn', '')} | {row.get('fp', '')} | {row.get('best_epoch', '')} | `{row.get('run_dir', '')}` |"
            )
    else:
        for row in rows[:20]:
            lines.append(
                f"| `{row.get('tag')}` | {row.get('seed', '')} | {fmt_num(row.get('test_f1'))} | {row.get('fn', '')} | {row.get('fp', '')} | {row.get('epoch', '')} | `{row.get('run_dir', '')}` |"
            )
    lines.extend([
        "",
        "### Historical Tables",
        "",
        "예전 `docs/plots/*.png`와 dense one-factor 표는 gcsmooth matched control 기준이므로 current raw performance 표로 쓰지 않습니다. 필요하면 historical appendix로 별도 분리합니다.",
    ])
    return lines


def build_status(args: argparse.Namespace) -> list[str]:
    raw_ref = read_json(args.raw_ref_summary)
    rawbase = read_json(args.rawbase_summary)
    sample_skip = read_json(args.sample_skip_summary)
    candidate_csv = read_csv(args.live_candidates_csv)
    run_csv = read_csv(args.live_runs_csv)

    raw_agg = raw_ref.get("aggregates", {})
    rawbase_agg = rawbase.get("aggregates", {})
    rawbase_latest = latest_run(rawbase)
    sample_latest = latest_run(sample_skip)
    if candidate_csv:
        complete = sum(int(float(r.get("runs") or 0)) for r in candidate_csv)
        f1_vals = [float(r["f1_mean"]) for r in candidate_csv if r.get("f1_mean")]
        fn_vals = [float(r["fn_mean"]) for r in candidate_csv if r.get("fn_mean")]
        fp_vals = [float(r["fp_mean"]) for r in candidate_csv if r.get("fp_mean")]
        rawbase_agg = {
            "complete": complete,
            "f1_mean": sum(f1_vals) / len(f1_vals) if f1_vals else None,
            "fn_mean": sum(fn_vals) / len(fn_vals) if fn_vals else None,
            "fp_mean": sum(fp_vals) / len(fp_vals) if fp_vals else None,
        }

    lines = [
        "",
        "- 이 블록은 `scripts/update_live_summary_doc.py`가 controller artifact에서 갱신합니다.",
        "- 서버 rawbase queue 정책: core 축을 먼저 실행하고 후속 평가는 `color`, `sample_skip`, `logical_train` 순서로 분리한 뒤 `gc`를 마지막에 실행합니다. GC는 5조건만 유지합니다.",
    ]
    if raw_agg:
        lines.append(
            "- raw baseline refcheck: "
            f"`{raw_agg.get('complete', 0)}/5`, F1 `{fmt_num(raw_agg.get('f1_mean'))}`, "
            f"FN `{fmt_num(raw_agg.get('fn_mean'), 3)}`, FP `{fmt_num(raw_agg.get('fp_mean'), 3)}`."
        )
    if rawbase_agg:
        lines.append(
            "- rawbase round1 aggregate: "
            f"`{rawbase_agg.get('complete', 0)}` complete runs, F1 `{fmt_num(rawbase_agg.get('f1_mean'))}`, "
            f"FN `{fmt_num(rawbase_agg.get('fn_mean'), 3)}`, FP `{fmt_num(rawbase_agg.get('fp_mean'), 3)}`, "
            f"decision `{rawbase.get('decision', '')}`."
        )
    if candidate_csv:
        lines.append("- rawbase 현재 완료 후보:")
        for row in candidate_csv:
            lines.append(
                f"  - `{row.get('candidate')}`: `{row.get('runs')}/5`, F1 `{fmt_num(float(row.get('f1_mean') or 0))}`, FN `{fmt_num(float(row.get('fn_mean') or 0), 3)}`, FP `{fmt_num(float(row.get('fp_mean') or 0), 3)}`"
            )
    else:
        lines.extend(candidate_lines(rawbase))
    if run_csv:
        latest_csv = run_csv[-1]
        lines.append(
            "- latest completed rawbase run: "
            f"`{Path(latest_csv.get('run_dir', '')).name}` -> F1 `{fmt_num(float(latest_csv.get('best_info_f1') or 0))}`, "
            f"FN `{latest_csv.get('fn')}`, FP `{latest_csv.get('fp')}`, epoch `{latest_csv.get('best_epoch')}`."
        )
    elif rawbase_latest:
        lines.append(
            "- latest completed rawbase run: "
            f"`{rawbase_latest.get('tag')}` -> F1 `{fmt_num(rawbase_latest.get('test_f1'))}`, "
            f"FN `{rawbase_latest.get('fn')}`, FP `{rawbase_latest.get('fp')}`, epoch `{rawbase_latest.get('epoch')}`."
        )
    if sample_latest:
        lines.append(
            "- sample-skip result: "
            f"`{sample_latest.get('tag')}` -> F1 `{fmt_num(sample_latest.get('test_f1'))}`, "
            f"FN `{sample_latest.get('fn')}`, FP `{sample_latest.get('fp')}`."
        )
    else:
        lines.append("- sample-skip/logical-train: `bash scripts/sweeps_server/06_sample_skip.sh`, `bash scripts/sweeps_server/50_logical_train.sh` 순서로 별도 실행합니다.")
    lines.extend(
        [
            "- NT 평가는 selected threshold만 남겼고 reporting default는 `NT=0.9`입니다.",
            "- controller/all.sh 로그에서는 tqdm progress bar가 자동 비활성화됩니다.",
            "",
        ]
    )
    return lines


def build_overview(args: argparse.Namespace) -> list[str]:
    raw_ref = read_json(args.raw_ref_summary)
    rawbase = read_json(args.rawbase_summary)
    candidate_csv = read_csv(args.live_candidates_csv)
    raw_agg = raw_ref.get("aggregates", {})
    rawbase_agg = rawbase.get("aggregates", {})
    if candidate_csv:
        complete = sum(int(float(r.get("runs") or 0)) for r in candidate_csv)
        f1_vals = [float(r["f1_mean"]) for r in candidate_csv if r.get("f1_mean")]
        fn_vals = [float(r["fn_mean"]) for r in candidate_csv if r.get("fn_mean")]
        fp_vals = [float(r["fp_mean"]) for r in candidate_csv if r.get("fp_mean")]
        rawbase_agg = {
            "complete": complete,
            "f1_mean": sum(f1_vals) / len(f1_vals) if f1_vals else None,
            "fn_mean": sum(fn_vals) / len(fn_vals) if fn_vals else None,
            "fp_mean": sum(fp_vals) / len(fp_vals) if fp_vals else None,
        }
    lines = [""]
    if raw_agg:
        lines.append(
            "- Active raw baseline: `fresh0412_v11_refcheck_raw_n700` -> "
            f"`{raw_agg.get('complete', 0)}/5`, F1 `{fmt_num(raw_agg.get('f1_mean'))}`, "
            f"FN `{fmt_num(raw_agg.get('fn_mean'), 3)}`, FP `{fmt_num(raw_agg.get('fp_mean'), 3)}`."
        )
    lines.append("- Server queue policy: rawbase follow-up runs `color`, `sample_skip`, `logical_train`, then the final 5-condition GC block.")
    if rawbase_agg:
        lines.append(
            "- Live rawbase artifact: "
            f"`{rawbase_agg.get('complete', 0)}` complete runs, decision `{rawbase.get('decision', '')}`, "
            f"F1 `{fmt_num(rawbase_agg.get('f1_mean'))}`, FN `{fmt_num(rawbase_agg.get('fn_mean'), 3)}`, FP `{fmt_num(rawbase_agg.get('fp_mean'), 3)}`."
        )
    lines.extend(
        [
            "- Last completed matched control: `fresh0412_v11_refcheck_gcsmooth_n700` -> `F1=0.9955`, `FN=4.4`, `FP=2.4` over `5/5` seeds.",
            "- Historical selected ref: `fresh0412_v11_n700_existing` -> `F1=0.9901`, `FN=9.8`, `FP=5.0`; kept only as reference-selection history.",
            "- 기존 gcsmooth round2 결과는 보존하되, raw claim용 round2는 rawbase round1 결과 뒤 새로 선정합니다.",
            "",
        ]
    )
    return lines


def build_performance_summary(args: argparse.Namespace) -> list[str]:
    raw_ref = read_json(args.raw_ref_summary)
    rawbase = read_json(args.rawbase_summary)
    sample_skip = read_json(args.sample_skip_summary)

    raw_agg = raw_ref.get("aggregates", {})
    rawbase_agg = rawbase.get("aggregates", {})
    sample_latest = latest_run(sample_skip)

    lines = [
        "",
        "| experiment | basis | seeds/runs | F1 | FN | FP | note |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    if raw_agg:
        lines.append(
            f"| `fresh0412_v11_refcheck_raw_n700` | raw server baseline | {raw_agg.get('complete', 0)}/5 | "
            f"{fmt_num(raw_agg.get('f1_mean'))} | {fmt_num(raw_agg.get('fn_mean'), 3)} | {fmt_num(raw_agg.get('fp_mean'), 3)} | 현재 서버 기준선 |"
        )
    else:
        lines.append("| `fresh0412_v11_refcheck_raw_n700` | raw server baseline | 5/5 | 0.99746 | 1.6 | 2.2 | 현재 서버 기준선 |")
    lines.extend([
        "| `fresh0412_v11_refcheck_gcsmooth_n700` | matched control | 5/5 | 0.9955 | 4.4 | 2.4 | 아래 기존 strict 표의 delta 기준 |",
        "| `fresh0412_v11_n700_existing` | historical selected ref | 5/5 | 0.9901 | 9.8 | 5.0 | 과거 reference 선택 기록 |",
    ])
    if rawbase_agg:
        lines.append(
            f"| rawbase round1 live | rawbase history | {rawbase_agg.get('complete', 0)} runs | "
            f"{fmt_num(rawbase_agg.get('f1_mean'))} | {fmt_num(rawbase_agg.get('fn_mean'), 3)} | {fmt_num(rawbase_agg.get('fp_mean'), 3)} | 중간 집계, 최종 claim 아님 |"
        )
    if sample_latest:
        lines.append(
            f"| sample-skip pilot | separate 1-run | 1/1 | {fmt_num(sample_latest.get('test_f1'))} | "
            f"{sample_latest.get('fn')} | {sample_latest.get('fp')} | main sweep과 분리 |"
        )
    lines.append("")
    return lines


def replace_section(text: str, heading: str, new_lines: list[str]) -> str:
    pattern = re.compile(rf"({re.escape(heading)}\n)(.*?)(\n## )", re.S)
    replacement = r"\1" + "\n".join(new_lines) + r"\3"
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise SystemExit(f"heading section not found: {heading}")
    return text


def replace_tail(text: str, heading: str, new_lines: list[str]) -> str:
    idx = text.find(heading)
    if idx < 0:
        return text.rstrip() + "\n\n" + "\n".join(new_lines) + "\n"
    return text[:idx].rstrip() + "\n\n" + "\n".join(new_lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("docs/summary.md"))
    parser.add_argument("--raw-ref-summary", type=Path, default=Path("validations/01_baseline_results.json"))
    parser.add_argument("--rawbase-summary", type=Path, default=Path("validations/02_sweep_results.json"))
    parser.add_argument("--sample-skip-summary", type=Path, default=Path("validations/03_sample_skip_results.json"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--validations-dir", type=Path, default=Path("validations"))
    parser.add_argument("--docs-plots-dir", type=Path, default=Path("docs/plots"))
    parser.add_argument("--live-candidates-csv", type=Path, default=Path("validations/log_history_report_rawbase_candidates.csv"))
    parser.add_argument("--live-runs-csv", type=Path, default=Path("validations/log_history_report_rawbase_runs.csv"))
    args = parser.parse_args()

    text = args.docs.read_text(encoding="utf-8")
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    text = re.sub(r"_자동 갱신 시각: `[^`]+`._", f"_자동 갱신 시각: `{now}`._", text, count=1)
    if "## 성능 요약" in text:
        text = replace_section(text, "## 성능 요약", build_performance_summary(args))
    elif "## 현재 진행 상태" in text:
        text = replace_section(text, "## 현재 진행 상태", build_status(args))
    if "## 요약" in text:
        text = replace_section(text, "## 요약", build_overview(args))
    if "## Rawbase Live Tables And Plots" in text:
        live_lines = rawbase_live_sections(args)
        text = replace_tail(text, "## Rawbase Live Tables And Plots", live_lines)
    args.docs.write_text(text, encoding="utf-8")
    print(f"[live-summary] updated {args.docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
