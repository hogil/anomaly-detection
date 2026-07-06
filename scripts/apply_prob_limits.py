#!/usr/bin/env python
"""Re-judge an existing inference output with a new prob-limit CSV.

모델 재실행 없이 predictions.csv의 p_abnormal에 새 limit만 다시 적용한다.
UI에서 limit을 바꾼 뒤 호출하는 용도:
  - 판정이 바뀐 chart의 display 이미지를 normal/ <-> abnormal/ 간 이동
  - display가 없던 chart(--display-filter abnormal/none 으로 생략된 것)는 새로 렌더링
  - predictions.csv / predictions.txt / abnormal_list.txt / normal_list.txt 갱신
    (기존 predictions.csv는 predictions_prev_<ts>.csv 로 백업)

limit CSV에 매칭되지 않는 chart는 기존 판정을 유지한다.

Usage:
  python scripts/apply_prob_limits.py \
    --run-dir <inference_output_dir> \
    --prob-limit-csv configs/prob_limits.csv \
    --data-dir data
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import (  # noqa: E402
    _detect_x_col, _init_render_worker, _render_display_task, _worker_cache,
    load_prob_limits, resolve_prob_limit,
)
from src.data.image_renderer import ImageRenderer  # noqa: E402

TQDM_DISABLE = not sys.stderr.isatty()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="inference.py 출력 폴더 (predictions.csv 포함)")
    parser.add_argument("--prob-limit-csv", required=True,
                        help="차트 종류별 판정 한계 CSV")
    parser.add_argument("--data-dir", default="data",
                        help="timeseries.csv + scenarios.csv 폴더 (누락 display 렌더링용)")
    parser.add_argument("--scenarios", default=None,
                        help="scenarios CSV (default: {data_dir}/scenarios.csv)")
    parser.add_argument("--config", default="dataset.yaml",
                        help="렌더링 설정 yaml")
    parser.add_argument("--display-dir", default="display",
                        help="run-dir 안의 display sub-folder 이름")
    parser.add_argument("--render", choices=["abnormal", "all", "none"], default="abnormal",
                        help="display 없는 chart 중 새로 렌더링할 대상 (default: abnormal 판정분)")
    parser.add_argument("--workers", type=int, default=0,
                        help="렌더링 병렬 worker (0=auto)")
    return parser.parse_args()


def norm_cell(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def display_filename(row: dict, title_columns: list[str]) -> str:
    """inference.py와 동일한 파일명 규칙: p{pct}_{parts}_{target}.png"""
    p_pct = f"p{int(round(float(row['p_abnormal']) * 100)):03d}"
    name_parts = [norm_cell(row.get(c)) or "unk" for c in title_columns]
    target = row.get("target")
    if isinstance(target, (int, float)) and not pd.isna(target):
        tgt_str = f"t{target:+.3f}".replace(".", "p").replace("+", "p").replace("-", "n")
    else:
        tgt_str = "tna"
    return f"{p_pct}_{'_'.join(name_parts)}_{tgt_str}.png"


def unique_path(path: Path, assigned: set) -> Path:
    suffix = 1
    base = path.name[:-len(".png")]
    while str(path) in assigned or path.exists():
        path = path.parent / f"{base}_{suffix}.png"
        suffix += 1
    assigned.add(str(path))
    return path


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        raise SystemExit(f"predictions.csv not found: {pred_path}")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    title_columns = cfg.get("image", {}).get("title_columns", ["device", "step", "item"])

    rules = load_prob_limits(args.prob_limit_csv, title_columns)
    preds = pd.read_csv(pred_path)
    for col in ("chart_id", "p_abnormal", "predicted"):
        if col not in preds.columns:
            raise SystemExit(f"predictions.csv에 {col} 컬럼이 없습니다")
    rows = preds.to_dict(orient="records")

    display_root = run_dir / args.display_dir
    for cls in ("normal", "abnormal"):
        (display_root / cls).mkdir(parents=True, exist_ok=True)

    # 1) 재판정 + 파일 disposition 결정
    flips_to_abn = flips_to_nor = moved = 0
    render_rows = []          # display 렌더링 필요한 row index
    assigned_paths: set = set()
    for row in rows:
        old_pred = norm_cell(row.get("predicted"))
        limit = resolve_prob_limit(row, rules)
        if limit is None:
            new_pred = old_pred  # 매칭 없음 → 기존 판정 유지
            row["prob_limit"] = ""
        else:
            new_pred = "abnormal" if float(row["p_abnormal"]) >= limit else "normal"
            row["prob_limit"] = limit
        row["predicted"] = new_pred
        flipped = new_pred != old_pred
        if flipped:
            if new_pred == "abnormal":
                flips_to_abn += 1
            else:
                flips_to_nor += 1

        img_rel = norm_cell(row.get("image_file"))
        img_abs = run_dir / img_rel if img_rel else None
        has_file = img_abs is not None and img_abs.exists()

        if has_file and flipped:
            dest = unique_path(display_root / new_pred / img_abs.name, assigned_paths)
            shutil.move(str(img_abs), str(dest))
            row["image_file"] = str(dest.relative_to(run_dir)).replace("\\", "/")
            moved += 1
            # model input 이미지도 같이 이동 (있을 때만)
            mi_rel = norm_cell(row.get("model_input_file"))
            mi_abs = run_dir / mi_rel if mi_rel else None
            if mi_abs is not None and mi_abs.exists():
                mi_dest = mi_abs.parent.parent / new_pred / mi_abs.name
                mi_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(mi_abs), str(mi_dest))
                row["model_input_file"] = str(mi_dest.relative_to(run_dir)).replace("\\", "/")
        elif has_file:
            assigned_paths.add(str(img_abs))
        elif args.render == "all" or (args.render == "abnormal" and new_pred == "abnormal"):
            render_rows.append(row)

    # 2) 누락 display 렌더링 (scenarios + timeseries 필요)
    rendered = 0
    if render_rows:
        data_dir = Path(args.data_dir)
        sc_file = Path(args.scenarios) if args.scenarios else data_dir / "scenarios.csv"
        sc_df = pd.read_csv(sc_file)
        ts_df = pd.read_csv(data_dir / "timeseries.csv")
        x_col = _detect_x_col(ts_df)
        sc_map = {}
        for sc_row in sc_df.to_dict(orient="records"):
            sc_map.setdefault(sc_row["chart_id"], sc_row)

        tasks = []
        for row in render_rows:
            sc_row = sc_map.get(row["chart_id"])
            if sc_row is None:
                print(f"  [warn] scenarios에 없는 chart_id — 렌더링 생략: {row['chart_id']}")
                continue
            disp_path = unique_path(
                display_root / row["predicted"] / display_filename(row, title_columns),
                assigned_paths,
            )
            title_parts = [norm_cell(row.get(c)) for c in title_columns if norm_cell(row.get(c))]
            tasks.append({
                "idx": len(tasks),
                "row": sc_row,
                "disp_path": str(disp_path),
                "title": " / ".join(title_parts) if title_parts else str(row["chart_id"]),
            })
            row["image_file"] = str(disp_path.relative_to(run_dir)).replace("\\", "/")

        if tasks:
            needed = {task["row"]["chart_id"] for task in tasks}
            ts_grouped = {sid: grp for sid, grp in ts_df.groupby("chart_id") if sid in needed}
            workers = args.workers if args.workers > 0 else min(max(1, cpu_count() - 1),
                                                                max(1, len(tasks) // 25))
            if workers == 1:
                _worker_cache["ts_grouped"] = ts_grouped
                _worker_cache["renderer"] = ImageRenderer(cfg)
                _worker_cache["x_col"] = x_col
                for task in tqdm(tasks, desc="render display", disable=TQDM_DISABLE):
                    _render_display_task(task)
                    rendered += 1
            else:
                import pickle
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as handle:
                    pickle.dump(ts_grouped, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    ts_pickle_path = handle.name
                try:
                    with Pool(processes=workers, initializer=_init_render_worker,
                              initargs=(ts_pickle_path, cfg, x_col)) as pool:
                        for _ in tqdm(pool.imap_unordered(_render_display_task, tasks, chunksize=4),
                                      total=len(tasks), desc="render display", disable=TQDM_DISABLE):
                            rendered += 1
                finally:
                    os.unlink(ts_pickle_path)

    # 3) predictions.csv 백업 후 갱신 + 리스트 재작성 (inference.py와 동일 포맷)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    backup = run_dir / f"predictions_prev_{stamp}.csv"
    shutil.copy2(pred_path, backup)

    results_df = pd.DataFrame(rows).sort_values("p_abnormal", ascending=False).reset_index(drop=True)
    results_df.to_csv(pred_path, index=False)

    abn_df = results_df[results_df["predicted"] == "abnormal"]
    nor_df = results_df[results_df["predicted"] == "normal"]
    list_columns = [c for c in ("device", "step", "item", "target", "p_abnormal", "prob_limit",
                                "chart_id", "image_file") if c in results_df.columns]
    abn_df[list_columns].to_csv(run_dir / "abnormal_list.txt", index=False)
    nor_df[list_columns].to_csv(run_dir / "normal_list.txt", index=False)
    with open(run_dir / "predictions.txt", "w", encoding="utf-8") as f:
        f.write(f"# ABNORMAL ({len(abn_df)}) — p_abnormal desc\n")
        f.write(",".join(list_columns) + "\n")
        for _, r in abn_df.iterrows():
            f.write(",".join(str(r[c]) for c in list_columns) + "\n")
        f.write(f"\n# NORMAL ({len(nor_df)}) — p_abnormal desc\n")
        f.write(",".join(list_columns) + "\n")
        for _, r in nor_df.iterrows():
            f.write(",".join(str(r[c]) for c in list_columns) + "\n")

    print(f"[apply-prob-limits] total={len(rows)} "
          f"normal→abnormal={flips_to_abn} abnormal→normal={flips_to_nor} "
          f"moved={moved} rendered={rendered}")
    print(f"[apply-prob-limits] now: abnormal={len(abn_df)} normal={len(nor_df)}")
    print(f"[apply-prob-limits] backup: {backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
