# Tabular Data Pipeline Redesign

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 데이터 생성을 tabular CSV 기반으로 전면 재설계. 모든 클래스(normal 포함)가 overlay 이미지 포맷으로 통일.

**Architecture:** 시나리오 단위로 tabular 시계열 생성 → CSV 저장 → CSV에서 이미지 렌더링. 학습/추론 동일 파이프라인. 모든 이미지는 overlay(하이라이트+fleet) 포맷.

**Tech Stack:** Python, numpy, pandas, matplotlib, PyYAML, scikit-learn

---

## 핵심 변경사항

### 이전 (폐기)
```
generate_data.py → series.npy + masks.npy + labels.csv + fleet/*.npz/*.json
generate_images.py → npy/npz 로드 → 이미지 렌더링
```

### 이후 (신규)
```
generate_data.py → data/timeseries.csv + data/scenarios.csv
generate_images.py → CSV 로드 → 그룹핑 → overlay 이미지 렌더링
```

### 데이터 포맷

**data/timeseries.csv** — 원시 시계열 (Fab 데이터와 동일 포맷)
```
scenario_id,time_index,eqp_id,chamber,recipe,value
sc_00000,0,EQP_A,CH01,RCP_001,0.05
sc_00000,1,EQP_A,CH01,RCP_001,0.03
sc_00000,2,EQP_A,CH01,RCP_001,NaN
...
```

**data/scenarios.csv** — 시나리오 메타데이터
```
scenario_id,class,grouping_column,target_member,deviation_type,defect_start_idx,split,defect_params
sc_00000,mean_shift,eqp_id,EQP_A,mean_shift,230,train,{...}
sc_00001,context,recipe,RCP_003,mean,,train,{...}
sc_00002,normal,eqp_id,EQP_B,none,,train,{}
```

### 이미지 포맷 통일 (ALL overlay)

**모든 클래스**가 동일한 overlay 포맷:
- images/: target=하이라이트색 + fleet=회색, legend "Fleet" 1개
- display/: target=빨강(불량) 또는 파랑(정상) + fleet=각각 연한색, legend 개별 ID

**시나리오 = 1개 그룹핑 컬럼의 fleet**:
- normal: 정상 target + 정상 fleet → label: normal
- mean_shift: mean_shift target + 정상 fleet → label: mean_shift
- context: context deviation target + 정상 fleet → label: context

---

## File Structure

```
src/
├── data/
│   ├── baseline_generator.py   # 유지 (에피소드 기반 시계열)
│   ├── defect_synthesizer.py   # 유지 (불량 주입)
│   ├── scenario_generator.py   # 신규: 시나리오 생성 오케스트레이션
│   └── image_renderer.py       # 수정: overlay 통일
│
generate_data.py                # 전면 재작성: tabular CSV 출력
generate_images.py              # 전면 재작성: CSV → 이미지
config.yaml                     # 수정: 새 구조 반영
```

- `fleet_generator.py` → **삭제** (scenario_generator.py로 통합)
- `scenario_generator.py` — 모든 클래스의 시나리오를 통합 생성. 1 시나리오 = 1 그룹핑 컬럼 + N 멤버 + target 1개

---

## Task 1: config.yaml 재설계

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: config.yaml 새 구조 작성**

```yaml
# =============================================================================
# Semiconductor Metrology Anomaly Detection - Configuration
# =============================================================================
seed: 42

# =============================================================================
# 1. 에피소드 기반 시계열 생성
# =============================================================================
episode:
  count_range: [8, 18]
  length_range: [15, 35]
  region_weights:
    dense: 0.45
    sparse: 0.35
    missing: 0.20
  region:
    dense:
      density_range: [0.85, 1.0]
    sparse:
      density_range: [0.30, 0.55]
    missing:
      density: 0.0
  noise:
    gaussian:
      sigma_range: [0.02, 0.06]
    laplacian:
      b_range: [0.01, 0.04]
    correlated:
      rho_range: [0.6, 0.8]
      sigma_range: [0.02, 0.05]

# =============================================================================
# 2. 베이스라인 신호
# =============================================================================
baseline:
  value_range: [-0.3, 0.3]
  random_walk_step: [0.003, 0.010]

# =============================================================================
# 3. 시나리오 공통 (모든 클래스에 fleet overlay 적용)
# =============================================================================
scenario:
  grouping:
    columns: ["eqp_id", "chamber", "recipe"]
    num_columns_range: [1, 3]          # context만 다중 컬럼, 나머지는 1개 사용
    eqp_id_count_range: [3, 7]
    chamber_count_range: [3, 5]
    recipe_count_range: [3, 5]
  fleet_variation:
    mean_range: [0.0, 0.01]
    std_range: [0.0, 0.01]

# =============================================================================
# 4. 불량 합성 (baseline_std 배수)
# =============================================================================
defect:
  region_ratio_range: [0.10, 0.35]
  mean_shift:
    shift_sigma_range: [3.0, 4.5]
    noise_boost_range: [1.2, 1.6]
  standard_deviation:
    scale_range: [2.2, 3.0]
  spike:
    magnitude_sigma_range: [6.0, 10.0]
    spike_ratio_range: [0.15, 0.35]
  drift:
    slope_sigma_range: [0.06, 0.10]

# =============================================================================
# 5. Context 불량
# =============================================================================
context:
  anomaly_count_range: [1, 2]
  target_deviation:
    mean_sigma_range: [3.5, 5.0]
    std_range: [2.8, 3.5]

# =============================================================================
# 6. 메타데이터 풀
# =============================================================================
metadata:
  equipment:
    ids: ["EQP_A","EQP_B","EQP_C","EQP_D","EQP_E","EQP_F","EQP_G","EQP_H","EQP_I","EQP_J"]
  chamber:
    ids: ["CH01","CH02","CH03","CH04","CH05"]
  recipe:
    ids: ["RCP_001","RCP_002","RCP_003","RCP_004","RCP_005","RCP_006","RCP_007","RCP_008"]

# =============================================================================
# 7. 이미지 렌더링
# =============================================================================
image:
  width: 224
  height: 224
  dpi: 100
  background: "white"
  overlay:
    fleet_color: "#B0B0B0"
    fleet_alpha: 0.40
    fleet_marker_size: 14
    target_color: "#4878CF"
    target_alpha: 0.75
    target_marker_size: 16

# =============================================================================
# 8. 데이터셋
# =============================================================================
dataset:
  samples_per_class: 200
  classes: ["normal","mean_shift","standard_deviation","spike","drift","context"]
  split:
    train: 0.70
    val: 0.15
    test: 0.15
  test_difficulty_scale: 0.6

# =============================================================================
# 9. 출력 경로
# =============================================================================
output:
  data_dir: "data"
  image_dir: "images"
  display_dir: "display"
```

- [ ] **Step 2: Commit**

```bash
git add config.yaml
git commit -m "refactor: redesign config for tabular data pipeline"
```

---

## Task 2: ScenarioGenerator 구현

**Files:**
- Create: `src/data/scenario_generator.py`
- Delete: `src/data/fleet_generator.py`

ScenarioGenerator는 **모든 클래스**의 시나리오를 생성한다.
1 시나리오 = 1 그룹핑 컬럼 + N 멤버(독립 베이스라인) + target 1개.

- normal: target에 불량 없음
- mean_shift/std/spike/drift: target의 시계열 오른쪽 끝에 불량 주입
- context: target의 전체 시계열이 fleet 대비 유의차

- [ ] **Step 1: scenario_generator.py 작성**

```python
"""
시나리오 생성기

모든 클래스를 통합 처리:
- 1 시나리오 = 1 그룹핑 컬럼 + N 멤버 + target 1개
- 출력: tabular 형식 (scenario_id, time_index, member_id, value)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .baseline_generator import BaselineGenerator
from .defect_synthesizer import DefectSynthesizer


@dataclass
class ScenarioResult:
    """시나리오 생성 결과"""
    scenario_id: str
    cls: str                        # normal, mean_shift, ..., context
    grouping_column: str            # eqp_id, chamber, recipe
    member_ids: List[str]           # fleet 전체 멤버
    target_id: str                  # 하이라이트 대상
    defect_start_idx: int           # 불량 시작 인덱스 (-1이면 없음)
    defect_params: dict             # 불량 파라미터
    timeseries_rows: List[dict]     # [{scenario_id, time_index, member_id, value}, ...]


class ScenarioGenerator:

    def __init__(self, config: dict, rng: np.random.Generator = None):
        self.cfg = config
        self.rng = rng or np.random.default_rng()
        self.baseline_gen = BaselineGenerator(config, rng=self.rng)
        self.defect_synth = DefectSynthesizer(config, rng=self.rng)
        self.meta = config["metadata"]
        self.scenario_cfg = config["scenario"]

        self._pool = {
            "eqp_id": self.meta["equipment"]["ids"],
            "chamber": self.meta["chamber"]["ids"],
            "recipe": self.meta["recipe"]["ids"],
        }
        grp = self.scenario_cfg["grouping"]
        self._count_range = {
            "eqp_id": grp["eqp_id_count_range"],
            "chamber": grp["chamber_count_range"],
            "recipe": grp["recipe_count_range"],
        }

    def generate(self, scenario_id: str, cls: str) -> ScenarioResult:
        """1개 시나리오 생성"""
        # 1) 그룹핑 컬럼 1개 랜덤 선택
        columns = self.scenario_cfg["grouping"]["columns"]
        col = str(self.rng.choice(columns))

        # 2) 멤버 수 랜덤 + ID 선택
        pool = self._pool[col]
        lo, hi = self._count_range[col]
        count = self.rng.integers(lo, min(hi, len(pool)) + 1)
        member_ids = [str(m) for m in self.rng.choice(pool, size=count, replace=False)]

        # 3) target 선택
        target_id = str(self.rng.choice(member_ids))

        # 4) 각 멤버 독립 베이스라인 생성
        member_data = {}  # {member_id: (values, mask)}
        fleet_means = []

        for mid in member_ids:
            values, mask, _ = self.baseline_gen.generate()
            member_data[mid] = (values, mask)
            valid = np.where(mask)[0]
            if len(valid) > 0:
                fleet_means.append(np.nanmean(values[valid]))

        # 5) Fleet 평균 정렬 (타이트한 밴드)
        fleet_center = np.mean(fleet_means) if fleet_means else 0.0
        fleet_var = self.scenario_cfg["fleet_variation"]

        for mid in member_ids:
            values, mask = member_data[mid]
            valid = np.where(mask)[0]
            if len(valid) > 0:
                cur = np.nanmean(values[valid])
                tgt = fleet_center + self.rng.uniform(*fleet_var["mean_range"]) * self.rng.choice([-1, 1])
                values[valid] += (tgt - cur)
                member_data[mid] = (values, mask)

        # 6) Target에 불량 주입
        defect_start_idx = -1
        defect_params = {}

        if cls == "normal":
            pass  # 불량 없음

        elif cls == "context":
            # Context: target 전체 시계열이 fleet 대비 유의차
            defect_params = self._inject_context(member_data, member_ids, target_id)

        else:
            # mean_shift, std, spike, drift: target 시계열 오른쪽 끝에 주입
            values, mask = member_data[target_id]
            new_values, info = self.defect_synth.inject(values, mask, cls)
            member_data[target_id] = (new_values, mask)
            defect_start_idx = info.start_idx
            defect_params = info.parameters

        # 7) Tabular rows 변환
        rows = []
        for mid in member_ids:
            values, mask = member_data[mid]
            for t in range(len(values)):
                if mask[t]:
                    rows.append({
                        "scenario_id": scenario_id,
                        "time_index": t,
                        "member_id": mid,
                        "value": float(values[t]),
                    })

        return ScenarioResult(
            scenario_id=scenario_id,
            cls=cls,
            grouping_column=col,
            member_ids=member_ids,
            target_id=target_id,
            defect_start_idx=defect_start_idx,
            defect_params=defect_params,
            timeseries_rows=rows,
        )

    def _inject_context(self, member_data, member_ids, target_id):
        """Context deviation: target의 전체 시계열을 fleet 대비 이동"""
        ctx_cfg = self.cfg["context"]
        dev_cfg = ctx_cfg["target_deviation"]

        values, mask = member_data[target_id]
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return {}

        # fleet std 측정
        fleet_stds = []
        for mid in member_ids:
            if mid == target_id:
                continue
            fv, fm = member_data[mid]
            fvalid = np.where(fm)[0]
            if len(fvalid) > 0:
                fleet_stds.append(np.nanstd(fv[fvalid]))
        fleet_std = np.mean(fleet_stds) if fleet_stds else 0.05
        fleet_std = max(fleet_std, 0.01)

        deviation_type = str(self.rng.choice(["mean", "std", "both"]))
        mean_shift = 0.0
        std_scale = 1.0

        if deviation_type in ("mean", "both"):
            factor = float(self.rng.uniform(*dev_cfg["mean_sigma_range"]))
            mean_shift = fleet_std * factor * self.rng.choice([-1, 1])
            values[valid] += mean_shift

        if deviation_type in ("std", "both"):
            std_scale = float(self.rng.uniform(*dev_cfg["std_range"]))
            center = np.nanmean(values[valid])
            values[valid] = center + (values[valid] - center) * std_scale

        member_data[target_id] = (values, mask)
        return {
            "deviation_type": deviation_type,
            "mean_shift": float(mean_shift),
            "std_scale": float(std_scale),
            "fleet_std": float(fleet_std),
        }
```

- [ ] **Step 2: fleet_generator.py 삭제**

```bash
rm src/data/fleet_generator.py
```

- [ ] **Step 3: Commit**

```bash
git add src/data/scenario_generator.py
git rm src/data/fleet_generator.py
git commit -m "refactor: replace fleet_generator with unified scenario_generator"
```

---

## Task 3: generate_data.py 전면 재작성

**Files:**
- Rewrite: `generate_data.py`

시나리오 기반으로 tabular CSV 2개 생성:
- `data/timeseries.csv` — 원시 시계열
- `data/scenarios.csv` — 시나리오 메타데이터

Context는 다중 컬럼 시 여러 시나리오 생성 (컬럼당 1개).

- [ ] **Step 1: generate_data.py 작성**

```python
"""
데이터 생성 에이전트 (Tabular)

시나리오 단위로 tabular 시계열 생성 → CSV 저장.
모든 클래스가 overlay 포맷 (1 시나리오 = 1 컬럼 그룹핑 + N 멤버).

Usage:
    python generate_data.py
    python generate_data.py --config config.yaml

출력:
    data/timeseries.csv   (scenario_id, time_index, member_id, value)
    data/scenarios.csv    (scenario_id, class, grouping_column, target_member, ...)
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
    """test용 불량 강도 축소"""
    cfg = copy.deepcopy(config)
    d = cfg["defect"]

    r = d["mean_shift"]["shift_sigma_range"]
    d["mean_shift"]["shift_sigma_range"] = [r[0] * scale, r[1] * scale]
    r = d["mean_shift"]["noise_boost_range"]
    d["mean_shift"]["noise_boost_range"] = [1 + (r[0]-1)*scale, 1 + (r[1]-1)*scale]
    r = d["standard_deviation"]["scale_range"]
    d["standard_deviation"]["scale_range"] = [1 + (r[0]-1)*scale, 1 + (r[1]-1)*scale]
    r = d["spike"]["magnitude_sigma_range"]
    d["spike"]["magnitude_sigma_range"] = [r[0]*scale, r[1]*scale]
    r = d["drift"]["slope_sigma_range"]
    d["drift"]["slope_sigma_range"] = [r[0]*scale, r[1]*scale]

    c = cfg["context"]["target_deviation"]
    r = c["mean_sigma_range"]
    c["mean_sigma_range"] = [r[0]*scale, r[1]*scale]
    r = c["std_range"]
    c["std_range"] = [1 + (r[0]-1)*scale, 1 + (r[1]-1)*scale]

    return cfg


def generate_batch(config, classes, count_per_class, rng, start_id, label):
    """시나리오 배치 생성"""
    gen = ScenarioGenerator(config, rng=rng)
    grp_cfg = config["scenario"]["grouping"]
    ctx_cfg = config.get("context", {})

    all_ts_rows = []
    all_sc_rows = []
    sc_id = start_id

    for cls in classes:
        if cls == "context":
            # Context: 다중 컬럼 시나리오 → 불량 컬럼만 카운트
            generated = 0
            pbar = tqdm(total=count_per_class, desc=f"{cls}({label})")
            while generated < count_per_class:
                # 컬럼 수 결정
                num_cols = rng.integers(
                    grp_cfg["num_columns_range"][0],
                    grp_cfg["num_columns_range"][1] + 1
                )
                num_cols = min(num_cols, len(grp_cfg["columns"]))
                chosen = list(rng.choice(grp_cfg["columns"], size=num_cols, replace=False))

                # 불량 수 결정
                ano_lo, ano_hi = ctx_cfg.get("anomaly_count_range", [1, 1])
                num_ano = rng.integers(ano_lo, min(ano_hi, num_cols) + 1)
                ano_cols = list(rng.choice(chosen, size=num_ano, replace=False))

                for col in chosen:
                    if generated >= count_per_class:
                        break
                    is_anomalous = str(col) in [str(c) for c in ano_cols]
                    sid = f"sc_{sc_id:05d}"
                    result = gen.generate(sid, "context" if is_anomalous else "normal")

                    # context인 경우만 카운트
                    if is_anomalous:
                        all_ts_rows.extend(result.timeseries_rows)
                        all_sc_rows.append({
                            "scenario_id": sid,
                            "class": "context",
                            "grouping_column": result.grouping_column,
                            "target_member": result.target_id,
                            "members": ",".join(result.member_ids),
                            "defect_start_idx": -1,
                            "defect_params": json.dumps(result.defect_params),
                        })
                        generated += 1
                        pbar.update(1)
                    sc_id += 1
            pbar.close()

        else:
            # normal, mean_shift, std, spike, drift
            for i in tqdm(range(count_per_class), desc=f"{cls}({label})"):
                sid = f"sc_{sc_id:05d}"
                result = gen.generate(sid, cls)

                all_ts_rows.extend(result.timeseries_rows)
                all_sc_rows.append({
                    "scenario_id": sid,
                    "class": cls,
                    "grouping_column": result.grouping_column,
                    "target_member": result.target_id,
                    "members": ",".join(result.member_ids),
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

    test_ratio = split_cfg["test"]
    n_test = max(1, round(samples_per_class * test_ratio))
    n_trainval = samples_per_class - n_test

    # Train+Val
    print(f"\n=== Train+Val (클래스당 {n_trainval}개) ===")
    tv_ts, tv_sc, next_id = generate_batch(config, classes, n_trainval, rng, 0, "train+val")

    # Test
    test_config = scale_config(config, test_scale)
    print(f"\n=== Test (클래스당 {n_test}개, scale={test_scale}) ===")
    te_ts, te_sc, _ = generate_batch(test_config, classes, n_test, rng, next_id, "test")

    # Train/Val split
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

    # Save
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
```

- [ ] **Step 2: Commit**

```bash
git add generate_data.py
git commit -m "refactor: rewrite generate_data for tabular CSV output"
```

---

## Task 4: generate_images.py 전면 재작성

**Files:**
- Rewrite: `generate_images.py`

CSV에서 시계열 로드 → 시나리오별 그룹핑 → overlay 이미지 렌더링.

- [ ] **Step 1: generate_images.py 작성**

```python
"""
이미지 생성 에이전트

data/timeseries.csv + data/scenarios.csv 에서 읽어서 이미지 생성.
모든 클래스가 overlay 포맷 (target 하이라이트 + fleet 회색/연한색).

Usage:
    python generate_images.py
    python generate_images.py --config config.yaml

출력:
    images/{split}/{class}/    학습용 (target=하이라이트 + fleet=회색, 축 없음)
    display/{split}/{class}/   유저용 (전체 멤버 색상 구분, 축 있음)
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.data.image_renderer import ImageRenderer


def render_all(config: dict):
    renderer = ImageRenderer(config)
    out_cfg = config["output"]

    data_dir = Path(out_cfg["data_dir"])
    images_dir = Path(out_cfg["image_dir"])
    display_dir = Path(out_cfg["display_dir"])

    ts_df = pd.read_csv(data_dir / "timeseries.csv")
    sc_df = pd.read_csv(data_dir / "scenarios.csv")

    print(f"이미지 생성: {len(sc_df)}개 시나리오")

    for _, row in tqdm(sc_df.iterrows(), total=len(sc_df), desc="이미지 생성"):
        sid = row["scenario_id"]
        cls = row["class"]
        split = row["split"]
        target_id = row["target_member"]
        members = row["members"].split(",")
        defect_start = int(row.get("defect_start_idx", -1))

        # 이 시나리오의 시계열 추출
        sc_ts = ts_df[ts_df["scenario_id"] == sid]
        if sc_ts.empty:
            continue

        # member별 (values, mask) 딕셔너리 구성
        fleet_data = {}
        for mid in members:
            member_ts = sc_ts[sc_ts["member_id"] == mid].sort_values("time_index")
            if member_ts.empty:
                continue
            max_t = int(member_ts["time_index"].max()) + 1
            values = np.full(max_t, np.nan)
            mask = np.zeros(max_t, dtype=bool)
            for _, pt in member_ts.iterrows():
                t = int(pt["time_index"])
                values[t] = pt["value"]
                mask[t] = True
            fleet_data[mid] = (values, mask, None)

        if not fleet_data:
            continue

        filename = f"{sid}.png"
        train_path = images_dir / split / cls / filename
        disp_path = display_dir / split / cls / filename

        # 학습용: target=하이라이트 + fleet=회색
        renderer.render_overlay(fleet_data, target_id, str(train_path))

        # display용: 전체 멤버 색상 구분
        anomalous_ids = [target_id] if cls != "normal" else []
        renderer.render_overlay_display(
            fleet_data, target_id, str(disp_path),
            anomalous_ids=anomalous_ids,
            defect_start_idx=defect_start if cls not in ("normal", "context") else -1,
        )

    for split in ["train", "val", "test"]:
        n = len(sc_df[sc_df["split"] == split])
        print(f"  {split}: {n}개")
    print(f"\n  학습용: {images_dir}/")
    print(f"  유저용: {display_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    render_all(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add generate_images.py
git commit -m "refactor: rewrite generate_images for tabular CSV input"
```

---

## Task 5: image_renderer.py 수정

**Files:**
- Modify: `src/data/image_renderer.py`

변경사항:
- `render_single`, `render_single_display` 제거 (더 이상 단일 시계열 없음)
- `render_overlay_display`에 `defect_start_idx` 파라미터 추가 (mean_shift/spike/drift 경계선)
- display에서 normal 클래스는 target을 파랑으로, anomaly 클래스는 빨강으로

- [ ] **Step 1: image_renderer.py 수정**

render_overlay_display 시그니처 변경:
```python
def render_overlay_display(self, fleet_data: dict, target_eqp: str,
                           save_path: str, anomalous_ids: list = None,
                           defect_start_idx: int = -1):
```

- anomalous_ids가 비어있으면 (normal): target을 파랑으로, fleet을 연한색으로
- anomalous_ids가 있으면 (anomaly): target을 빨강으로, fleet을 연한색으로
- defect_start_idx > 0이면: 경계선 표시

render_single / render_single_display 삭제.

- [ ] **Step 2: Commit**

```bash
git add src/data/image_renderer.py
git commit -m "refactor: unify image_renderer to overlay-only format"
```

---

## Task 6: 통합 테스트 & 정리

**Files:**
- Modify: `src/data/__init__.py`
- Delete: old data files

- [ ] **Step 1: __init__.py에서 fleet_generator import 제거**

- [ ] **Step 2: 기존 데이터 정리**

```bash
rm -rf data/ images/ display/
```

- [ ] **Step 3: 소수 생성 테스트 (10개/클래스)**

config.yaml에서 `samples_per_class: 10`으로 설정 후:

```bash
python generate_data.py
python generate_images.py
```

확인:
- `data/timeseries.csv` 존재, 컬럼: scenario_id, time_index, member_id, value
- `data/scenarios.csv` 존재, 컬럼: scenario_id, class, grouping_column, target_member, members, ...
- `images/train/normal/` 에 overlay 이미지
- `display/train/mean_shift/` 에 display 이미지
- 모든 이미지가 overlay 포맷 (target + fleet)

- [ ] **Step 4: 이미지 육안 확인**

각 클래스별 1개씩 display 이미지 열어서 확인:
- normal: target(파랑) + fleet(연한색), 전부 정상
- mean_shift: target(빨강) 영역이 이동, 경계선 표시
- spike: target(빨강) 영역에 스파이크
- drift: target(빨강) 영역에 드리프트
- context: target(빨강) fleet 대비 유의차

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: complete tabular data pipeline redesign"
```

---

## Context 다중 컬럼 display 처리 (Task 3 내 포함)

Context 시나리오에서 컬럼이 2~3개인 경우:
- 학습 이미지: 불량 컬럼의 시나리오만 생성 (불량 수만큼)
- display: 각 컬럼별 display 이미지를 별도 생성 (generate_images.py에서 scenarios.csv의 context 행 + 동일 시나리오의 다른 컬럼 정보로)

이 부분은 Task 3의 context 생성 로직에서 다중 컬럼 시나리오의 모든 컬럼 데이터를 저장하고, Task 4에서 display 렌더링 시 활용.

**주의:** 현재 설계에서는 1 시나리오 = 1 컬럼으로 단순화. 다중 컬럼 display는 Phase 2에서 확장 가능.
