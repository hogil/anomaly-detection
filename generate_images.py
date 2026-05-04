"""
이미지 생성 에이전트 (병렬 처리)

data/timeseries.csv + data/scenarios.csv → 이미지 렌더링.
모든 클래스가 overlay 포맷 (highlighted_member 하이라이트 + fleet).

병렬 처리: multiprocessing.Pool 사용 (CPU 코어 수만큼 worker)

images/: highlighted_member=하이라이트 + fleet=회색
display/: 전체 멤버 색상 구분
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from src.data.image_renderer import ImageRenderer
from src.data.schema import highlighted_member as read_highlighted_member
from src.data.schema import legend_axis as read_legend_axis
from src.data.schema import members as read_members
from src.data.schema import target as read_target

DEFAULT_DATASET_CONFIG = "dataset.yaml"
TQDM_DISABLE = not sys.stderr.isatty()


# worker별 ts_grouped 캐시 (process 단위)
_worker_cache = {}


def _init_worker(ts_pickle_path: str, config: dict, x_col: str = "time_index"):
    """worker process 초기화 — 데이터 로드"""
    import pickle
    with open(ts_pickle_path, "rb") as f:
        _worker_cache["ts_grouped"] = pickle.load(f)
    _worker_cache["renderer"] = ImageRenderer(config)
    _worker_cache["images_dir"] = Path(config["output"]["image_dir"])
    _worker_cache["display_dir"] = Path(config["output"]["display_dir"])
    _worker_cache["x_col"] = x_col

    # display 메타: 제목 컬럼 + 축 라벨
    img_cfg = config.get("image", {})
    _worker_cache["title_columns"] = img_cfg.get("title_columns", ["device", "step", "item"])
    _worker_cache["x_label"] = img_cfg.get("x_label") or x_col
    _worker_cache["y_label"] = img_cfg.get("y_label", "Measurement Value (nm)")
    # Stage 14 color ablation: "always" (default) or "context_only"
    _worker_cache["fleet_mode"] = img_cfg.get("fleet_mode", "always")


def _render_one(row_dict: dict):
    """1개 시나리오 렌더링 (worker 내부 호출)"""
    ts_grouped = _worker_cache["ts_grouped"]
    renderer = _worker_cache["renderer"]
    images_dir = _worker_cache["images_dir"]
    display_dir = _worker_cache["display_dir"]
    x_col = _worker_cache.get("x_col", "time_index")
    title_columns = _worker_cache.get("title_columns", ["device", "step", "item"])
    x_label = _worker_cache.get("x_label", x_col)
    y_label = _worker_cache.get("y_label", "Measurement Value (nm)")

    sid = row_dict["chart_id"]
    cls = row_dict["class"]
    split = row_dict["split"]
    highlighted_member = read_highlighted_member(row_dict)
    members = read_members(row_dict)
    legend_axis = read_legend_axis(row_dict)
    if not highlighted_member or not members or not legend_axis:
        return None

    # defect_start_idx: 연속 x 좌표 (numeric or datetime). NaN/None/-1 → 없음
    raw_ds = row_dict.get("defect_start_idx", None)
    defect_start = None
    if raw_ds is not None:
        try:
            if isinstance(raw_ds, (int, float)):
                if not (np.isnan(raw_ds) or raw_ds < 0):
                    defect_start = float(raw_ds)
            else:
                # datetime, str timestamp 등 — 그대로 통과
                defect_start = raw_ds
        except (TypeError, ValueError):
            defect_start = None

    sc_ts = ts_grouped.get(sid)
    if sc_ts is None:
        return None

    # fleet 구성: 멤버별 (x_vals, y_vals) pair
    # x_col은 numeric, datetime, 또는 다른 연속 좌표 가능
    fleet_data = {}
    for mid in members:
        member_ts = sc_ts[sc_ts[legend_axis].astype(str) == str(mid)].sort_values(x_col)
        if member_ts.empty:
            continue
        x_vals = member_ts[x_col].to_numpy()
        y_vals = member_ts["value"].to_numpy()
        if len(x_vals) == 0:
            continue
        fleet_data[mid] = (x_vals, y_vals)

    if not fleet_data:
        return None

    # 제목: title_columns에서 값을 추출하여 " / "로 연결
    title_parts = []
    for col in title_columns:
        if col in row_dict and row_dict[col] is not None:
            title_parts.append(str(row_dict[col]))
    chart_title = " / ".join(title_parts) if title_parts else sid

    filename = f"{sid}.png"
    train_path = images_dir / split / cls / filename
    disp_path = display_dir / split / cls / filename

    target = read_target(row_dict)

    # Stage 14: fleet_mode="context_only" → non-context class 에서는 fleet 제거 (highlighted member만)
    fleet_mode = _worker_cache.get("fleet_mode", "always")
    if fleet_mode == "context_only" and cls != "context":
        train_fleet_data = {highlighted_member: fleet_data[highlighted_member]} if highlighted_member in fleet_data else fleet_data
    else:
        train_fleet_data = fleet_data
    renderer.render_overlay(train_fleet_data, highlighted_member, str(train_path),
                            target=target)

    anomalous_ids = [highlighted_member] if cls != "normal" else []
    disp_defect_start = defect_start if cls not in ("normal", "context") else None
    renderer.render_overlay_display(
        fleet_data, highlighted_member, str(disp_path),
        anomalous_ids=anomalous_ids,
        defect_start_idx=disp_defect_start,
        title=chart_title,
        x_label=x_label,
        y_label=y_label,
        target=target,
    )
    return sid


def _log(message: str) -> None:
    print(message, flush=True)


def render_all(
    config: dict,
    num_workers: int = 0,
    x_col: str = None,
    chunksize: int | None = None,
    maxtasksperchild: int = 0,
):
    out_cfg = config["output"]
    data_dir = Path(out_cfg["data_dir"])
    started = time.perf_counter()

    _log("데이터 로드 중...")
    ts_df = pd.read_csv(data_dir / "timeseries.csv")
    sc_df = pd.read_csv(data_dir / "scenarios.csv")
    _log(f"  CSV 로드 완료: timeseries={len(ts_df):,} rows, scenarios={len(sc_df):,} rows")

    _log("  chart_id별 timeseries group 생성 중...")
    ts_grouped = {sid: grp for sid, grp in ts_df.groupby("chart_id")}
    _log(f"  group 생성 완료: {len(ts_grouped):,} charts")

    # x 컬럼 자동 감지: time_index → timestamp → date 순서
    if x_col is None:
        for cand in ["time_index", "timestamp", "datetime", "date", "time"]:
            if cand in ts_df.columns:
                x_col = cand
                break
        if x_col is None:
            x_col = "time_index"  # fallback
    _log(f"X 컬럼: {x_col} (dtype={ts_df[x_col].dtype})")

    if num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)
    if chunksize is None:
        chunksize = 8 if num_workers > 1 else 1
    _log(
        f"이미지 생성: {len(sc_df):,}개 "
        f"(workers={num_workers}, chunksize={chunksize}, maxtasksperchild={maxtasksperchild or 'off'})"
    )

    for split, cls in sc_df[["split", "class"]].drop_duplicates().itertuples(index=False):
        (Path(out_cfg["image_dir"]) / str(split) / str(cls)).mkdir(parents=True, exist_ok=True)
        (Path(out_cfg["display_dir"]) / str(split) / str(cls)).mkdir(parents=True, exist_ok=True)

    rows = sc_df.to_dict(orient="records")

    if num_workers == 1:
        _worker_cache["ts_grouped"] = ts_grouped
        _worker_cache["renderer"] = ImageRenderer(config)
        _worker_cache["images_dir"] = Path(config["output"]["image_dir"])
        _worker_cache["display_dir"] = Path(config["output"]["display_dir"])
        _worker_cache["x_col"] = x_col
        img_cfg = config.get("image", {})
        _worker_cache["title_columns"] = img_cfg.get("title_columns", ["device", "step", "item"])
        _worker_cache["x_label"] = img_cfg.get("x_label") or x_col
        _worker_cache["y_label"] = img_cfg.get("y_label", "Measurement Value (nm)")
        _worker_cache["fleet_mode"] = img_cfg.get("fleet_mode", "always")
        rendered = 0
        for result in tqdm(rows, total=len(rows), desc="이미지 생성", disable=TQDM_DISABLE):
            if _render_one(result) is not None:
                rendered += 1
    else:
        # ts_grouped를 worker 간 공유 (pickle to temp file)
        import pickle, tempfile

        _log("  worker 초기화용 timeseries pickle 생성 중...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(ts_grouped, f, protocol=pickle.HIGHEST_PROTOCOL)
            ts_pickle_path = f.name
        _log(f"  pickle 생성 완료: {os.path.getsize(ts_pickle_path) / (1024*1024):.1f} MB")

        pool_kwargs = {
            "processes": num_workers,
            "initializer": _init_worker,
            "initargs": (ts_pickle_path, config, x_col),
        }
        if maxtasksperchild > 0:
            pool_kwargs["maxtasksperchild"] = maxtasksperchild

        rendered = 0
        try:
            _log("  worker 시작 중...")
            with Pool(**pool_kwargs) as pool:
                iterator = pool.imap_unordered(_render_one, rows, chunksize=chunksize)
                for result in tqdm(iterator, total=len(rows), desc="이미지 생성", disable=TQDM_DISABLE):
                    if result is not None:
                        rendered += 1
        finally:
            os.unlink(ts_pickle_path)

    for split in ["train", "val", "test"]:
        n = len(sc_df[sc_df["split"] == split])
        _log(f"  {split}: {n}개")
    _log(f"  rendered: {rendered:,}/{len(sc_df):,}")
    _log(f"  elapsed: {time.perf_counter() - started:.1f}s")
    _log(f"\n  학습용: {out_cfg['image_dir']}/")
    _log(f"  유저용: {out_cfg['display_dir']}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--workers", type=int, default=0,
                        help="병렬 worker 수 (0=auto, CPU 코어 수-1)")
    parser.add_argument("--x_col", type=str, default=None,
                        help="x축 컬럼명 (자동 감지: time_index, timestamp, ...)")
    parser.add_argument("--chunksize", type=int, default=0,
                        help="Pool chunksize (0=auto)")
    parser.add_argument("--maxtasksperchild", type=int, default=0,
                        help="worker recycle interval. 0=off; old behavior used 100 and can cause long pauses.")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    render_all(
        config,
        num_workers=args.workers,
        x_col=args.x_col,
        chunksize=args.chunksize or None,
        maxtasksperchild=args.maxtasksperchild,
    )


if __name__ == "__main__":
    main()
