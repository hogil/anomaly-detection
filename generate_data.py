"""
데이터 생성 에이전트 (Tabular)

Chart = device + step + item (고정)
Context = eqp_id, chamber, recipe (fleet 비교)

출력:
    data/timeseries.csv   (chart_id, time_index, device, step, item, eqp_id, chamber, recipe, value)
    data/scenarios.csv    (chart_id, class, device, step, item, context_column, target, contexts, ...)
"""

import argparse
import copy
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.data.scenario_generator import ScenarioGenerator


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


def generate_batch(config, classes, count_per_class, rng, start_id, label):
    """count_per_class: int (모든 클래스 동일) 또는 dict {class: count}"""
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
            })
            sc_id += 1

    return all_ts_rows, all_sc_rows, sc_id


def generate(config: dict):
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

    print(f"\n=== Train+Val ===")
    for cls in classes:
        print(f"  {cls}: {n_trainval_dict[cls]}")
    tv_ts, tv_sc, next_id = generate_batch(config, classes, n_trainval_dict, rng, 0, "train+val")

    test_config = scale_config(config, test_scale)
    print(f"\n=== Test (scale={test_scale}) ===")
    for cls in classes:
        print(f"  {cls}: {n_test_dict[cls]}")
    te_ts, te_sc, _ = generate_batch(test_config, classes, n_test_dict, rng, next_id, "test")

    tv_df = pd.DataFrame(tv_sc)
    te_df = pd.DataFrame(te_sc)

    val_ratio = split_cfg["val"] / (split_cfg["train"] + split_cfg["val"])
    train_df, val_df = train_test_split(
        tv_df, test_size=val_ratio, stratify=tv_df["class"], random_state=seed
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    generate(config)


if __name__ == "__main__":
    main()
