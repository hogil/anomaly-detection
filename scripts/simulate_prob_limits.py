#!/usr/bin/env python
"""Simulate per-chart-type prob limits against labeled predictions.

inference.py 출력 predictions.csv (true_class 포함)를 받아 chart 종류별로
"limit을 X로 하면 FN/FP가 몇 개인지" 표를 만든다. limit 최종 결정은 수동
(차트 종류마다 운영 지식이 필요) — 이 스크립트는 시뮬레이션 근거만 제공한다.

판정 규칙: p_abnormal >= limit → abnormal.
  FN = 실제 abnormal 인데 normal 판정 (놓침)
  FP = 실제 normal 인데 abnormal 판정 (오검)

Usage:
  python scripts/simulate_prob_limits.py \
    --predictions <inference_output>/predictions.csv \
    --group-by item \
    --out-prefix validations/prob_limit_sim
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_limits(text: str) -> list[float]:
    """'0.05:0.95:0.05' grid 또는 '0.1,0.3,0.5' 목록."""
    if ":" in text:
        start, stop, step = (float(x) for x in text.split(":"))
        vals = []
        v = start
        while v <= stop + 1e-9:
            vals.append(round(v, 4))
            v += step
        return vals
    return [float(x) for x in text.split(",") if x.strip()]


def binary_truth(df: pd.DataFrame) -> tuple[pd.Series, str]:
    for col in ("true_class", "binary_class", "class"):
        if col in df.columns:
            truth = df[col].astype(str).map(lambda c: "normal" if c == "normal" else "abnormal")
            return truth, col
    raise SystemExit("label 컬럼이 없습니다 (true_class/binary_class/class 중 하나 필요)")


def sweep_group(p_abn: pd.Series, truth: pd.Series, limits: list[float]) -> pd.DataFrame:
    is_abn = truth == "abnormal"
    n_abn = int(is_abn.sum())
    rows = []
    for limit in limits:
        pred_abn = p_abn >= limit
        fn = int((is_abn & ~pred_abn).sum())
        fp = int((~is_abn & pred_abn).sum())
        tp = n_abn - fn
        recall = tp / n_abn if n_abn else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)
        rows.append({"limit": limit, "FN": fn, "FP": fp,
                     "recall_abn": round(recall, 4), "f1": round(f1, 4)})
    return pd.DataFrame(rows)


def current_metrics(sub: pd.DataFrame, truth: pd.Series) -> dict:
    """predicted 컬럼 기준 현재 판정 성능 (FN/FP/recall/precision/F1)."""
    is_abn = truth == "abnormal"
    pred_abn = sub["predicted"].astype(str) == "abnormal"
    n_abn = int(is_abn.sum())
    fn = int((is_abn & ~pred_abn).sum())
    fp = int((~is_abn & pred_abn).sum())
    tp = n_abn - fn
    recall = tp / n_abn if n_abn else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"n": len(sub), "abnormal": n_abn, "FN": fn, "FP": fp,
            "recall_abn": round(recall, 4), "precision_abn": round(precision, 4),
            "f1": round(f1, 4)}


def pick_markers(sweep: pd.DataFrame) -> dict:
    """참고용 마커 2개 — 최종 선택은 수동.
    fn0: FN=0을 유지하는 최대 limit (val 기준 하나도 안 놓치면서 FP 최소)
    best_f1: F1 최대 limit
    """
    marks = {}
    zero_fn = sweep[sweep["FN"] == 0]
    if not zero_fn.empty:
        marks["fn0"] = float(zero_fn["limit"].max())
    best = sweep.sort_values(["f1", "limit"], ascending=[False, True]).iloc[0]
    marks["best_f1"] = float(best["limit"])
    return marks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path,
                        help="inference.py 출력 predictions.csv (라벨 컬럼 포함)")
    parser.add_argument("--group-by", default="item",
                        help="차트 종류 묶음 컬럼 (콤마 구분, 예: item 또는 device,step,item)")
    parser.add_argument("--limits", default="0.05:0.95:0.05",
                        help="limit 후보. 'start:stop:step' grid 또는 콤마 목록")
    parser.add_argument("--out-prefix", default="validations/prob_limit_sim",
                        help="출력 prefix — <prefix>.md / <prefix>.csv 생성")
    parser.add_argument("--emit-csv", type=Path, default=None,
                        help="fn0 마커 값으로 채운 prob_limits 초안 CSV (수동 편집용 시작점)")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    if "p_abnormal" not in df.columns:
        raise SystemExit("p_abnormal 컬럼이 없습니다")
    truth, label_col = binary_truth(df)
    group_cols = [c.strip() for c in args.group_by.split(",") if c.strip()]
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"group 컬럼이 predictions에 없습니다: {missing}")
    limits = parse_limits(args.limits)

    md_lines = [
        "# Prob Limit Simulation",
        "",
        f"- source: `{args.predictions}`",
        f"- label: `{label_col}` / rows: {len(df)} "
        f"(abnormal={int((truth == 'abnormal').sum())}, normal={int((truth == 'normal').sum())})",
        "- 판정: p_abnormal >= limit → abnormal. FN=놓침, FP=오검",
        "- 마커는 참고용 — 최종 limit은 차트 종류별 운영 지식으로 수동 결정",
        "",
    ]
    all_rows = []
    draft_rows = []

    groups = [("__ALL__", df)]
    for key, sub in df.groupby(group_cols):
        name = "/".join(str(v) for v in (key if isinstance(key, tuple) else (key,)))
        groups.append((name, sub))

    # 현재 판정 기준 성능 (predicted 컬럼이 있을 때만)
    if "predicted" in df.columns:
        md_lines.append("## 현재 판정 기준 성능 (predicted 컬럼)")
        md_lines.append("")
        md_lines.append(f"| {args.group_by} | n | abnormal | FN | FP | recall_abn | precision_abn | f1 |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        current_rows = []
        for name, sub in groups:
            m = current_metrics(sub, truth.loc[sub.index])
            md_lines.append(
                f"| {name} | {m['n']} | {m['abnormal']} | {m['FN']} | {m['FP']} "
                f"| {m['recall_abn']:.4f} | {m['precision_abn']:.4f} | {m['f1']:.4f} |"
            )
            current_rows.append({"group": name, **m})
        md_lines.append("")
        Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(current_rows).to_csv(
            Path(args.out_prefix).parent / (Path(args.out_prefix).name + "_current.csv"),
            index=False)

    for name, sub in groups:
        sub_p = sub["p_abnormal"].astype(float)
        sub_t = truth.loc[sub.index]
        sweep = sweep_group(sub_p, sub_t, limits)
        marks = pick_markers(sweep)
        n_abn = int((sub_t == "abnormal").sum())
        md_lines.append(f"## {args.group_by}={name} (n={len(sub)}, abnormal={n_abn})")
        md_lines.append("참고 마커: " + ", ".join(f"{k}={v}" for k, v in marks.items()))
        md_lines.append("")
        md_lines.append("| limit | FN | FP | recall_abn | f1 | 마커 |")
        md_lines.append("|---:|---:|---:|---:|---:|---|")
        for _, r in sweep.iterrows():
            tags = [k for k, v in marks.items() if abs(v - r["limit"]) < 1e-9]
            md_lines.append(
                f"| {r['limit']:.2f} | {int(r['FN'])} | {int(r['FP'])} "
                f"| {r['recall_abn']:.4f} | {r['f1']:.4f} | {' '.join(tags)} |"
            )
        md_lines.append("")
        sweep.insert(0, "group", name)
        all_rows.append(sweep)
        if name != "__ALL__" and "fn0" in marks:
            draft = dict(zip(group_cols, name.split("/")))
            draft["prob_limit"] = marks["fn0"]
            draft_rows.append(draft)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    md_path = out_prefix.with_suffix(".md")
    csv_path = out_prefix.with_suffix(".csv")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    pd.concat(all_rows, ignore_index=True).to_csv(csv_path, index=False)
    print(f"[simulate-prob-limits] wrote {md_path} / {csv_path}")

    if args.emit_csv is not None:
        if not draft_rows:
            print("[simulate-prob-limits] emit-csv 생략 — fn0 마커가 있는 group이 없음")
        else:
            args.emit_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(draft_rows).to_csv(args.emit_csv, index=False)
            print(f"[simulate-prob-limits] draft prob_limits CSV: {args.emit_csv} (수동 검토 필수)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
