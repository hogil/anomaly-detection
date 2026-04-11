"""
데이터 생성 에이전트 (Tabular)

Chart = device + step + item (고정)
Context = eqp_id, chamber, recipe (fleet 비교)

출력:
    data/timeseries.csv   (chart_id, time_index, device, step, item, eqp_id, chamber, recipe, value)
    data/scenarios.csv    (chart_id, class, device, step, item, context_column, target, contexts, ...)

병렬 처리:
    --workers 1   : 순차 (default, 노트북 기본)
    --workers 0   : auto (cpu_count - 1, H200 서버 권장)
    --workers N>1 : N개 process 병렬

병렬 모드에서는 (seed, scenario_id) 로부터 SeedSequence로 per-task RNG 유도.
순차 모드 결과와는 달라지지만, 같은 (config, seed, workers) 조합은 100% 재현.
"""

import argparse
import copy
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from src.data.scenario_generator import ScenarioGenerator


def stratified_train_val_split(df: pd.DataFrame, val_ratio: float,
                                stratify_col: str, random_state: int):
    """sklearn.model_selection.train_test_split(stratify=...) 대체.

    폐쇄망 Ubuntu 24 + numpy 2.x + 구버전 scipy/sklearn ABI 충돌을 피하기 위해
    numpy만 사용. sklearn과 동일하게 class별 indices를 shuffle 후 val_ratio 만큼 분리.
    random_state 고정 시 deterministic.
    """
    rng = np.random.default_rng(random_state)
    train_idx = []
    val_idx = []
    for cls in sorted(df[stratify_col].unique()):
        cls_idx = df.index[df[stratify_col] == cls].to_numpy()
        perm = rng.permutation(len(cls_idx))
        cls_idx = cls_idx[perm]
        n_val = max(1, int(round(len(cls_idx) * val_ratio)))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    return df.loc[train_idx].copy(), df.loc[val_idx].copy()


def scale_config(config: dict, scale: float) -> dict:
    cfg = copy.deepcopy(config)
    d = cfg["defect"]

    r = d["mean_shift"]["shift_sigma_range"]
    d["mean_shift"]["shift_sigma_range"] = [r[0]*scale, r[1]*scale]
    r = d["standard_deviation"]["scale_range"]
    d["standard_deviation"]["scale_range"] = [1+(r[0]-1)*scale, 1+(r[1]-1)*scale]
    r = d["spike"]["magnitude_sigma_range"]
    d["spike"]["magnitude_sigma_range"] = [r[0]*scale, r[1]*scale]
    r = d["drift"]["slope_sigma_range"]
    d["drift"]["slope_sigma_range"] = [r[0]*scale, r[1]*scale]
    if "min_max_drift_sigma" in d["drift"]:
        d["drift"]["min_max_drift_sigma"] = d["drift"]["min_max_drift_sigma"] * scale

    # enforcement floor도 함께 scale (test가 더 약하게 보강되도록)
    if "enforcement" in d:
        e = d["enforcement"]
        for key in ["mean_shift_floor_sigma", "spike_floor_sigma",
                    "drift_floor_sigma", "context_floor_sigma"]:
            if key in e:
                e[key] = e[key] * scale
        # std_floor_ratio: 1을 기준으로 (1.0이 정상)
        if "std_floor_ratio" in e:
            e["std_floor_ratio"] = 1 + (e["std_floor_ratio"] - 1) * scale
        # normal floor (상한)는 scale하지 않음 — normal은 항상 깨끗해야 함

    c = cfg["context"]["target_deviation"]
    r = c["mean_sigma_range"]
    c["mean_sigma_range"] = [r[0]*scale, r[1]*scale]
    r = c["std_range"]
    c["std_range"] = [1+(r[0]-1)*scale, 1+(r[1]-1)*scale]

    return cfg


def _compute_target_value(result, sid: str, rng: np.random.Generator, cls: str) -> float:
    """수평 기준선 위치 계산 — fleet(정상 멤버) median.

    anomaly 를 제외한 정상 데이터의 중심을 보여준다.
    모든 클래스 공통: fleet (target 제외) 의 median 사용.
    """
    fleet_vals = [
        r["value"]
        for r in result.timeseries_rows
        if r["chart_id"] == sid and r.get(result.context_column) != result.target
    ]
    if fleet_vals:
        return float(np.median(fleet_vals))

    # fleet 없으면 전체 median fallback
    all_vals = [r["value"] for r in result.timeseries_rows if r["chart_id"] == sid]
    if all_vals:
        return float(np.median(all_vals))

    return 0.0


def generate_batch(config, classes, count_per_class, rng, start_id, label):
    """순차 생성 (기본). count_per_class: int 또는 dict {class: count}."""
    gen = ScenarioGenerator(config, rng=rng)

    all_ts_rows = []
    all_sc_rows = []
    sc_id = start_id

    for cls in classes:
        n = count_per_class[cls] if isinstance(count_per_class, dict) else count_per_class
        for i in tqdm(range(n), desc=f"{cls}({label})"):
            sid = f"ch_{sc_id:05d}"
            result = gen.generate(sid, cls)

            all_ts_rows.extend(result.timeseries_rows)
            tv = _compute_target_value(result, sid, rng, cls)

            all_sc_rows.append({
                "chart_id": sid,
                "class": cls,
                "device": result.device,
                "step": result.step,
                "item": result.item,
                "context_column": result.context_column,
                "target": result.target,
                "contexts": ",".join(result.contexts),
                "defect_start_idx": result.defect_start_idx,
                "defect_params": json.dumps(result.defect_params),
                "target_value": round(float(tv), 6),
            })
            sc_id += 1

    return all_ts_rows, all_sc_rows, sc_id


# ============================================================================
# 병렬 생성 (multiprocessing) — H200 서버 32 코어 활용
# ============================================================================

# Worker process별 캐시 (config는 init 1회만 pickle)
_w_cache = {}


def _init_worker(config_dict: dict, base_seed: int):
    """worker process 초기화 — config 캐시"""
    _w_cache["cfg"] = config_dict
    _w_cache["base_seed"] = base_seed


def _gen_one(task: tuple):
    """단일 시나리오 생성 (worker 내부 호출).

    task = (sid_int, cls, batch_idx)
      - sid_int:   chart_id 정수 (e.g. 5 → "ch_00005")
      - cls:       class name
      - batch_idx: 0=trainval, 1=test (RNG 격리용)

    재현성: SeedSequence([base_seed, sid_int, batch_idx]) → 같은 입력은 항상 같은 결과.
    sequential generate_batch와는 RNG 순서가 달라 결과 다름.
    """
    sid_int, cls, batch_idx = task
    cfg = _w_cache["cfg"]
    base = _w_cache["base_seed"]

    # Per-task deterministic RNG (worker order에 무관)
    ss = np.random.SeedSequence([int(base), int(sid_int), int(batch_idx)])
    rng = np.random.default_rng(ss)

    # Fresh ScenarioGenerator per task — child generators(BaselineGen/DefectSynth)도
    # 같은 rng 받음. ScenarioGenerator __init__는 가벼움 (config ref + child 2개).
    gen = ScenarioGenerator(cfg, rng=rng)

    sid = f"ch_{sid_int:05d}"
    result = gen.generate(sid, cls)
    target_value = _compute_target_value(result, sid, rng, cls)

    sc_row = {
        "chart_id": sid,
        "class": cls,
        "device": result.device,
        "step": result.step,
        "item": result.item,
        "context_column": result.context_column,
        "target": result.target,
        "contexts": ",".join(result.contexts),
        "defect_start_idx": result.defect_start_idx,
        "defect_params": json.dumps(result.defect_params),
        "target_value": round(float(target_value), 6),
    }
    return sid_int, sc_row, result.timeseries_rows


def generate_batch_parallel(config, classes, count_per_class, base_seed,
                             start_id, label, num_workers, batch_idx):
    """병렬 생성. num_workers process pool 사용.

    재현성: (base_seed, sid_int, batch_idx) → SeedSequence → per-task RNG.
    같은 인자는 worker 수와 무관하게 항상 같은 결과.
    """
    # Build task list
    tasks = []
    sc_id = start_id
    for cls in classes:
        n = count_per_class[cls] if isinstance(count_per_class, dict) else count_per_class
        for _ in range(n):
            tasks.append((sc_id, cls, batch_idx))
            sc_id += 1

    print(f"  [parallel] {len(tasks)} 시나리오, {num_workers} workers, batch_idx={batch_idx}")

    with Pool(processes=num_workers, initializer=_init_worker,
              initargs=(config, base_seed)) as pool:
        # imap_unordered로 worker 부하 자동 분산
        results = list(tqdm(
            pool.imap_unordered(_gen_one, tasks, chunksize=8),
            total=len(tasks), desc=f"{label}({num_workers}w)"
        ))

    # sid_int 순으로 정렬하여 deterministic 순서 복원
    results.sort(key=lambda r: r[0])

    all_ts_rows = []
    all_sc_rows = []
    for _, sc_row, ts_rows in results:
        all_sc_rows.append(sc_row)
        all_ts_rows.extend(ts_rows)

    return all_ts_rows, all_sc_rows, sc_id


def generate(config: dict, num_workers: int = 1):
    """전체 데이터 생성.

    num_workers:
      1   : 순차 (default, 노트북 호환)
      0   : auto (cpu_count - 1)
      N>1 : N process 병렬
    """
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)

    dataset_cfg = config["dataset"]
    split_cfg = dataset_cfg["split"]
    samples_per_class = dataset_cfg["samples_per_class"]
    classes = dataset_cfg["classes"]
    test_scale = dataset_cfg.get("test_difficulty_scale", 1.0)

    out_dir = Path(config["output"]["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # samples_per_class: int → 모든 클래스 동일, dict → 클래스별
    if isinstance(samples_per_class, dict):
        spc_dict = {cls: int(samples_per_class[cls]) for cls in classes}
    else:
        spc_dict = {cls: int(samples_per_class) for cls in classes}

    test_ratio = split_cfg["test"]
    n_test_dict = {cls: max(1, round(spc_dict[cls] * test_ratio)) for cls in classes}
    n_trainval_dict = {cls: spc_dict[cls] - n_test_dict[cls] for cls in classes}

    # 병렬 worker 수 결정
    if num_workers == 0:
        nw = max(1, cpu_count() - 1)
    else:
        nw = max(1, num_workers)
    use_parallel = nw > 1

    print(f"\n  Workers: {nw} ({'parallel' if use_parallel else 'sequential'})")

    print(f"\n=== Train+Val ===")
    for cls in classes:
        print(f"  {cls}: {n_trainval_dict[cls]}")
    if use_parallel:
        tv_ts, tv_sc, next_id = generate_batch_parallel(
            config, classes, n_trainval_dict, seed, 0, "train+val",
            num_workers=nw, batch_idx=0,
        )
    else:
        tv_ts, tv_sc, next_id = generate_batch(config, classes, n_trainval_dict, rng, 0, "train+val")

    test_config = scale_config(config, test_scale)
    print(f"\n=== Test (scale={test_scale}) ===")
    for cls in classes:
        print(f"  {cls}: {n_test_dict[cls]}")
    if use_parallel:
        te_ts, te_sc, _ = generate_batch_parallel(
            test_config, classes, n_test_dict, seed, next_id, "test",
            num_workers=nw, batch_idx=1,
        )
    else:
        te_ts, te_sc, _ = generate_batch(test_config, classes, n_test_dict, rng, next_id, "test")

    tv_df = pd.DataFrame(tv_sc)
    te_df = pd.DataFrame(te_sc)

    val_ratio = split_cfg["val"] / (split_cfg["train"] + split_cfg["val"])
    train_df, val_df = stratified_train_val_split(
        tv_df, val_ratio=val_ratio, stratify_col="class", random_state=seed,
    )
    train_df["split"] = "train"
    val_df["split"] = "val"
    te_df["split"] = "test"

    scenarios_df = pd.concat([train_df, val_df, te_df]).reset_index(drop=True)

    ts_df = pd.DataFrame(tv_ts + te_ts)
    ts_df.to_csv(out_dir / "timeseries.csv", index=False)
    scenarios_df.to_csv(out_dir / "scenarios.csv", index=False)

    print(f"\n{'='*50}")
    print(f"  데이터 생성 완료")
    print(f"{'='*50}")
    print(f"  시나리오: {len(scenarios_df)}")
    print(f"  시계열 행: {len(ts_df):,}")
    tr = len(scenarios_df[scenarios_df["split"]=="train"])
    va = len(scenarios_df[scenarios_df["split"]=="val"])
    te = len(scenarios_df[scenarios_df["split"]=="test"])
    print(f"  train={tr}, val={va}, test={te}")
    for cls in classes:
        c = len(scenarios_df[scenarios_df["class"]==cls])
        print(f"    {cls}: {c}")
    print(f"\n  저장: {out_dir}/")


def _snapshot_config(config: dict, source_path: Path) -> None:
    """데이터 생성 시 config 을 configs/datasets/ 로 snapshot 저장.

    - 파일명: {version}_{YYYYMMDD_HHMMSS}.yaml
    - version 은 dataset.version 또는 최근 v9midN naming 에서 추론
    - 기존 latest.yaml symlink (Windows 는 copy) 도 함께 갱신
    """
    import datetime as _dt
    import shutil as _sh
    out_dir = Path("configs/datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 버전 태그 결정 (dataset.version > timestamp)
    ver = None
    if isinstance(config.get("dataset"), dict):
        ver = config["dataset"].get("version")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{ver}_{ts}" if ver else ts

    snapshot_path = out_dir / f"dataset_{tag}.yaml"
    # 원본 yaml 을 그대로 복사 (주석 보존)
    _sh.copy2(source_path, snapshot_path)

    # latest 포인터 (symlink 불가능할 때 copy)
    latest_path = out_dir / "latest.yaml"
    try:
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        _sh.copy2(source_path, latest_path)
    except OSError:
        pass

    print(f"  [config snapshot] → {snapshot_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--workers", type=int, default=1,
                        help="병렬 worker 수 (1=순차 default, 0=auto cpu_count-1, N>1=N process)")
    parser.add_argument("--no_snapshot", action="store_true",
                        help="configs/datasets/ 자동 snapshot 비활성")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not args.no_snapshot:
        _snapshot_config(config, Path(args.config))

    generate(config, num_workers=args.workers)


if __name__ == "__main__":
    main()
