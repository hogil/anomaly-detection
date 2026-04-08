"""
이미지 생성 에이전트 (병렬 처리)

data/timeseries.csv + data/scenarios.csv → 이미지 렌더링.
모든 클래스가 overlay 포맷 (target 하이라이트 + fleet).

병렬 처리: multiprocessing.Pool 사용 (CPU 코어 수만큼 worker)

images/: target=하이라이트 + fleet=회색
display/: 전체 멤버 색상 구분
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from src.data.image_renderer import ImageRenderer


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
    target_id = row_dict["target"]
    contexts = row_dict["contexts"].split(",")
    ctx_col = row_dict["context_column"]

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
    for mid in contexts:
        member_ts = sc_ts[sc_ts[ctx_col] == mid].sort_values(x_col)
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

    renderer.render_overlay(fleet_data, target_id, str(train_path))

    anomalous_ids = [target_id] if cls != "normal" else []
    disp_defect_start = defect_start if cls not in ("normal", "context") else None
    renderer.render_overlay_display(
        fleet_data, target_id, str(disp_path),
        anomalous_ids=anomalous_ids,
        defect_start_idx=disp_defect_start,
        title=chart_title,
        x_label=x_label,
        y_label=y_label,
    )
    return sid


def render_all(config: dict, num_workers: int = 0, x_col: str = None):
    out_cfg = config["output"]
    data_dir = Path(out_cfg["data_dir"])

    print(f"데이터 로드 중...")
    ts_df = pd.read_csv(data_dir / "timeseries.csv")
    sc_df = pd.read_csv(data_dir / "scenarios.csv")
    ts_grouped = {sid: grp for sid, grp in ts_df.groupby("chart_id")}

    # x 컬럼 자동 감지: time_index → timestamp → date 순서
    if x_col is None:
        for cand in ["time_index", "timestamp", "datetime", "date", "time"]:
            if cand in ts_df.columns:
                x_col = cand
                break
        if x_col is None:
            x_col = "time_index"  # fallback
    print(f"X 컬럼: {x_col} (dtype={ts_df[x_col].dtype})")

    if num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)
    print(f"이미지 생성: {len(sc_df)}개 (workers={num_workers})")

    # ts_grouped를 worker 간 공유 (pickle to temp file)
    import pickle, tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        pickle.dump(ts_grouped, f)
        ts_pickle_path = f.name

    rows = sc_df.to_dict(orient="records")

    try:
        with Pool(processes=num_workers, initializer=_init_worker,
                  initargs=(ts_pickle_path, config, x_col)) as pool:
            results = list(tqdm(
                pool.imap_unordered(_render_one, rows, chunksize=20),
                total=len(rows), desc="이미지 생성"
            ))
    finally:
        os.unlink(ts_pickle_path)

    for split in ["train", "val", "test"]:
        n = len(sc_df[sc_df["split"] == split])
        print(f"  {split}: {n}개")
    print(f"\n  학습용: {out_cfg['image_dir']}/")
    print(f"  유저용: {out_cfg['display_dir']}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--workers", type=int, default=0,
                        help="병렬 worker 수 (0=auto, CPU 코어 수-1)")
    parser.add_argument("--x_col", type=str, default=None,
                        help="x축 컬럼명 (자동 감지: time_index, timestamp, ...)")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    render_all(config, num_workers=args.workers, x_col=args.x_col)


if __name__ == "__main__":
    main()
