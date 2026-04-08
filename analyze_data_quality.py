"""
데이터 품질 분석 스크립트

목표:
1. Normal인데 fleet 대비 이탈도가 abnormal 수준인 케이스 검출
2. Abnormal인데 defect 강도가 normal과 구분 안 되는 케이스 검출

접근:
- 각 chart마다 target/fleet 분리
- 좌측(0~70%)을 정상영역, 우측(70~100%)을 잠재적 defect영역으로 가정
- 지표:
    A) target_mean_dev   = |target.mean - fleet.mean| / fleet.std
    B) right_shift       = (target.right.mean - target.left.mean) / fleet.std
       (mean_shift/drift 강도)
    C) right_std_ratio   = target.right.std / target.left.std
       (standard_deviation 강도)
    D) right_max_dev     = max(|target.right - fleet.right.mean|) / fleet.std
       (spike 강도)
    E) right_dev_to_fleet = |target.right.mean - fleet.right.mean| / fleet.std
       (mean_shift visible to model)
    F) ctx_global_dev    = |target.mean - fleet.mean| / fleet.std
       (context 강도)

추출:
- normal class에서 right_dev_to_fleet, target_mean_dev 큰 상위 N개 → "abnormal처럼 보이는 normal"
- abnormal class에서 강도 지표 작은 하위 N개 → "normal처럼 보이는 abnormal"
- test split만 (test_difficulty_scale=0.7 적용된 것이라 가장 어려움)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("data_analysis")
OUT_DIR.mkdir(exist_ok=True)


def compute_metrics(target_vals, fleet_vals_list, target_idx, fleet_idx_list):
    """1개 chart에 대한 정량 지표 계산"""
    # fleet 통합 (모든 멤버 합침)
    fleet_all = np.concatenate(fleet_vals_list) if fleet_vals_list else np.array([])
    fleet_idx_all = np.concatenate(fleet_idx_list) if fleet_idx_list else np.array([])

    if len(fleet_all) < 5 or len(target_vals) < 5:
        return None

    fleet_mean = float(np.mean(fleet_all))
    fleet_std = max(float(np.std(fleet_all)), 1e-6)

    target_mean = float(np.mean(target_vals))
    target_std = float(np.std(target_vals))

    # 좌/우 영역 (time_index 70% 기준)
    if len(target_idx) == 0 or len(fleet_idx_all) == 0:
        return None

    t_max = max(target_idx.max(), fleet_idx_all.max())
    t_min = min(target_idx.min(), fleet_idx_all.min())
    split_idx = t_min + (t_max - t_min) * 0.7

    t_left_mask = target_idx < split_idx
    t_right_mask = target_idx >= split_idx
    f_right_mask = fleet_idx_all >= split_idx

    t_left_vals = target_vals[t_left_mask]
    t_right_vals = target_vals[t_right_mask]
    f_right_vals = fleet_all[f_right_mask]

    if len(t_left_vals) < 3 or len(t_right_vals) < 3 or len(f_right_vals) < 3:
        return None

    t_left_mean = float(np.mean(t_left_vals))
    t_left_std = max(float(np.std(t_left_vals)), 1e-6)
    t_right_mean = float(np.mean(t_right_vals))
    t_right_std = float(np.std(t_right_vals))
    f_right_mean = float(np.mean(f_right_vals))

    # 지표
    metrics = {
        # A) target 전체 평균이 fleet 평균에서 얼마나 떨어졌나
        "ctx_dev_sigma": abs(target_mean - fleet_mean) / fleet_std,
        # B) target 좌→우 평균 변화 (mean_shift / drift)
        "right_shift_sigma": abs(t_right_mean - t_left_mean) / fleet_std,
        # C) target 좌 vs 우 산포 비율 (standard_deviation)
        "right_std_ratio": t_right_std / t_left_std,
        # D) 우측 영역 최대 이탈 (spike)
        "right_max_dev_sigma": float(np.max(np.abs(t_right_vals - f_right_mean))) / fleet_std,
        # E) 우측 영역 평균이 fleet 우측 평균에서 얼마나 떨어졌나 (가장 직관적)
        "right_dev_to_fleet_sigma": abs(t_right_mean - f_right_mean) / fleet_std,
        # 부가
        "fleet_std": fleet_std,
        "target_std": target_std,
        "t_left_std": t_left_std,
        "t_right_std": t_right_std,
        "n_target": len(target_vals),
        "n_fleet": len(fleet_all),
        "n_target_right": len(t_right_vals),
    }
    return metrics


def main():
    print("Loading scenarios.csv ...")
    sc = pd.read_csv(DATA_DIR / "scenarios.csv")
    print(f"  {len(sc)} scenarios")

    print("Loading timeseries.csv ...")
    ts = pd.read_csv(DATA_DIR / "timeseries.csv")
    print(f"  {len(ts)} rows")

    # chart_id별 그룹화
    print("Grouping by chart_id ...")
    ts_grouped = dict(tuple(ts.groupby("chart_id", sort=False)))
    print(f"  {len(ts_grouped)} charts")

    # 분석 (test split만)
    test_sc = sc[sc["split"] == "test"].copy()
    print(f"Analyzing {len(test_sc)} test charts ...")

    records = []
    for _, row in test_sc.iterrows():
        chart_id = row["chart_id"]
        cls = row["class"]
        ctx_col = row["context_column"]
        target = row["target"]

        if chart_id not in ts_grouped:
            continue
        df = ts_grouped[chart_id]

        # target / fleet 분리
        target_df = df[df[ctx_col] == target]
        fleet_df = df[df[ctx_col] != target]

        if len(target_df) == 0 or len(fleet_df) == 0:
            continue

        target_vals = target_df["value"].to_numpy()
        target_idx = target_df["time_index"].to_numpy()

        # fleet은 멤버별로 분리
        fleet_vals_list = []
        fleet_idx_list = []
        for mid, mdf in fleet_df.groupby(ctx_col):
            fleet_vals_list.append(mdf["value"].to_numpy())
            fleet_idx_list.append(mdf["time_index"].to_numpy())

        m = compute_metrics(target_vals, fleet_vals_list, target_idx, fleet_idx_list)
        if m is None:
            continue

        m["chart_id"] = chart_id
        m["class"] = cls
        m["context_column"] = ctx_col
        records.append(m)

    df = pd.DataFrame(records)
    print(f"\n=== Computed metrics for {len(df)} test charts ===")

    # 클래스별 통계
    print("\n=== Class-level metric stats (median) ===")
    metric_cols = [
        "ctx_dev_sigma",
        "right_shift_sigma",
        "right_std_ratio",
        "right_max_dev_sigma",
        "right_dev_to_fleet_sigma",
    ]
    print(df.groupby("class")[metric_cols].median().round(3).to_string())

    print("\n=== Class-level metric stats (10th percentile, abnormal weakness) ===")
    print(df.groupby("class")[metric_cols].quantile(0.10).round(3).to_string())

    print("\n=== Normal class metric stats (90th percentile, normal strength) ===")
    norm = df[df["class"] == "normal"]
    print(norm[metric_cols].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).round(3).to_string())

    # === 케이스 1: Normal-like-abnormal ===
    # right_dev_to_fleet_sigma 또는 right_shift_sigma 또는 ctx_dev_sigma가 큰 normal
    normal = df[df["class"] == "normal"].copy()
    normal["suspicion_score"] = (
        normal["right_dev_to_fleet_sigma"]
        + normal["right_shift_sigma"]
        + normal["ctx_dev_sigma"]
    )
    bad_normal = normal.sort_values("suspicion_score", ascending=False).head(20)
    print("\n=== TOP 20: Normal cases that LOOK abnormal ===")
    print(bad_normal[["chart_id", "ctx_dev_sigma", "right_shift_sigma",
                      "right_dev_to_fleet_sigma", "right_std_ratio",
                      "suspicion_score"]].round(2).to_string(index=False))

    # === 케이스 2: Abnormal-like-normal ===
    print("\n=== BOTTOM 10 per abnormal class: cases that LOOK normal ===")
    abnormal_classes = ["mean_shift", "standard_deviation", "spike", "drift", "context"]

    weak_records = []
    for ac in abnormal_classes:
        sub = df[df["class"] == ac].copy()
        if len(sub) == 0:
            continue
        # 클래스별 핵심 지표
        if ac == "mean_shift":
            sub["weakness"] = sub["right_dev_to_fleet_sigma"]  # 작을수록 안 보임
        elif ac == "standard_deviation":
            sub["weakness"] = sub["right_std_ratio"]  # 1에 가까울수록 안 보임
        elif ac == "spike":
            sub["weakness"] = sub["right_max_dev_sigma"]
        elif ac == "drift":
            sub["weakness"] = sub["right_shift_sigma"]
        elif ac == "context":
            sub["weakness"] = sub["ctx_dev_sigma"]

        weak = sub.sort_values("weakness", ascending=True).head(10)
        weak["which_class"] = ac
        weak_records.append(weak)
        print(f"\n--- {ac} (key metric below) ---")
        cols = ["chart_id", "weakness", "ctx_dev_sigma", "right_shift_sigma",
                "right_dev_to_fleet_sigma", "right_std_ratio", "right_max_dev_sigma"]
        print(weak[cols].round(2).to_string(index=False))

    weak_all = pd.concat(weak_records, ignore_index=True) if weak_records else pd.DataFrame()

    # 저장
    df.to_csv(OUT_DIR / "metrics_test.csv", index=False)
    bad_normal.to_csv(OUT_DIR / "bad_normal_top20.csv", index=False)
    if len(weak_all) > 0:
        weak_all.to_csv(OUT_DIR / "weak_abnormal_per_class.csv", index=False)
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
