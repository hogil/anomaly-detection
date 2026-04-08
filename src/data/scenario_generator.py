"""
시나리오 생성기

구조:
- Chart = device + step + item (고정, 1 chart 정의)
- Context = eqp_id, chamber, recipe (fleet 비교 대상)
- 1 chart당 context 컬럼별 fleet을 생성
- 학습: 불량 멤버 하이라이트 이미지 = 1 학습 샘플
- 추론: context 컬럼별 × 종류별 이미지 전부 생성

모든 클래스가 overlay 포맷:
- normal: target에 불량 없음
- mean_shift/std/spike/drift: target 시계열 오른쪽 끝에 불량 주입
- context: target 전체 시계열이 fleet 대비 유의차
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

from .baseline_generator import BaselineGenerator
from .defect_synthesizer import DefectSynthesizer


@dataclass
class ScenarioResult:
    chart_id: str
    cls: str
    # chart 정의
    device: str
    step: str
    item: str
    # context 정보
    context_column: str             # eqp_id, chamber, recipe
    contexts: List[str]             # fleet 전체 멤버
    target: str                     # 하이라이트 대상
    # 불량 정보
    defect_start_idx: int
    defect_params: dict
    timeseries_rows: list


class ScenarioGenerator:

    def __init__(self, config: dict, rng: np.random.Generator = None):
        self.cfg = config
        self.rng = rng or np.random.default_rng()
        self.baseline_gen = BaselineGenerator(config, rng=self.rng)
        self.defect_synth = DefectSynthesizer(config, rng=self.rng)

        self.chart_cfg = config["chart"]
        self.ctx_cfg = config["context"]

        # context 컬럼별 pool과 count_range
        self._ctx_pool = {}
        self._ctx_count = {}
        for col in self.ctx_cfg["columns"]:
            self._ctx_pool[col] = self.ctx_cfg[col]["ids"]
            self._ctx_count[col] = self.ctx_cfg[col]["count_range"]

    def generate(self, chart_id: str, cls: str,
                 context_column: str = None) -> ScenarioResult:
        """1개 학습 샘플 생성

        Args:
            chart_id: chart ID
            cls: 클래스
            context_column: context 컬럼 강제 지정 (None이면 랜덤)
        """
        # 1) Chart 정의: device + step + item 랜덤 선택
        device = str(self.rng.choice(self.chart_cfg["device"]["ids"]))
        step = str(self.rng.choice(self.chart_cfg["step"]["ids"]))
        item = str(self.rng.choice(self.chart_cfg["item"]["ids"]))

        # 2) Context 컬럼 선택
        if context_column is None:
            context_column = str(self.rng.choice(self.ctx_cfg["columns"]))

        # 3) Context 멤버 선택
        pool = self._ctx_pool[context_column]
        lo, hi = self._ctx_count[context_column]
        count = self.rng.integers(lo, min(hi, len(pool)) + 1)
        contexts = [str(m) for m in self.rng.choice(pool, size=count, replace=False)]

        # 4) Target 선택
        target = str(self.rng.choice(contexts))

        # 5) 공유 에피소드 구조 생성
        ref_values, shared_mask, shared_episodes = self.baseline_gen.generate()

        # 6) 각 context 멤버: 멤버별 mask 변동 + 독립 값
        context_data = {}
        fleet_means = []

        for mid in contexts:
            member_mask = self.baseline_gen.generate_member_mask(shared_mask, shared_episodes)
            values = self.baseline_gen.generate_on_shared_mask(member_mask, shared_episodes)
            context_data[mid] = (values, member_mask)
            valid = np.where(member_mask)[0]
            if len(valid) > 0:
                fleet_means.append(np.nanmean(values[valid]))

        # 7) Fleet 평균 정렬
        fleet_center = np.mean(fleet_means) if fleet_means else 0.0
        fleet_var = self.ctx_cfg["fleet_variation"]

        for mid in contexts:
            values, mask = context_data[mid]
            valid = np.where(mask)[0]
            if len(valid) > 0:
                cur = np.nanmean(values[valid])
                tgt = fleet_center + self.rng.uniform(*fleet_var["mean_range"]) * self.rng.choice([-1, 1])
                values[valid] += (tgt - cur)
                context_data[mid] = (values, mask)

        # 8) 불량 주입
        defect_start_idx = -1
        defect_params = {}

        if cls == "normal":
            self._normalize_normal_target(context_data, contexts, target)
        elif cls == "context":
            defect_params = self._inject_context(context_data, contexts, target)
            # 사후 검증: context는 전체 평균이 fleet 평균에서 floor 이상 이격
            self._enforce_context_floor(context_data, contexts, target)
        else:
            # 불량 영역에 유효 포인트 최소 보장 (config의 enforcement.min_defect_points_range)
            enf = self.cfg.get("defect", {}).get("enforcement", {})
            mdp_range = enf.get("min_defect_points_range", [12, 25])
            MIN_DEFECT_POINTS = int(self.rng.integers(mdp_range[0], mdp_range[1] + 1))

            values, mask = context_data[target]
            total_len = len(mask)

            # target의 현재 평균/std
            valid_now = np.where(mask)[0]
            if len(valid_now) > 5:
                tgt_mean = np.nanmean(values[valid_now])
                tgt_std = max(np.nanstd(values[valid_now]), 0.01)
            else:
                tgt_mean = 0.0
                tgt_std = 0.05

            # === Fleet visible range 계산 (drift max_drift에 사용) ===
            all_fleet_vals = []
            for mid in contexts:
                if mid == target:
                    continue
                fv, fm = context_data[mid]
                fvi = np.where(fm)[0]
                if len(fvi) > 0:
                    all_fleet_vals.extend(fv[fvi].tolist())
            if all_fleet_vals:
                fleet_p5, fleet_p95 = np.percentile(all_fleet_vals, [5, 95])
                fleet_visible_range = max(fleet_p95 - fleet_p5, 0.05)
            else:
                fleet_visible_range = 0.1

            # 불량 주입 (fleet_range 전달)
            new_values, info = self.defect_synth.inject(
                values, mask, cls, fleet_range=fleet_visible_range
            )

            if info.num_affected < MIN_DEFECT_POINTS:
                start = info.start_idx
                end = info.end_idx
                needed = MIN_DEFECT_POINTS - info.num_affected
                empty_idxs = [t for t in range(start, end) if not mask[t]]
                if len(empty_idxs) > 0:
                    n_fill = min(needed, len(empty_idxs))
                    fill_idx = self.rng.choice(empty_idxs, size=n_fill, replace=False)
                    for t in fill_idx:
                        mask[t] = True
                        values[t] = tgt_mean + self.rng.normal(0, tgt_std)
                    new_values, info = self.defect_synth.inject(
                        values, mask, cls, fleet_range=fleet_visible_range
                    )

            context_data[target] = (new_values, mask)

            defect_start_idx = info.start_idx
            defect_params = info.parameters

            # 사후 검증: defect 영역 강도 floor 강제 (fleet_std 기준)
            self._enforce_defect_floor(
                context_data, contexts, target, cls,
                info.start_idx, info.end_idx
            )

        # 9) Tabular rows (6개 메타 컬럼)
        # 시작 time_index 오프셋 (chart마다 다른 시작점)
        offset_range = self.cfg.get("episode", {}).get("time_offset_range", [0, 0])
        time_offset = int(self.rng.integers(offset_range[0], offset_range[1] + 1))

        # defect_start_idx도 오프셋 적용
        if defect_start_idx >= 0:
            defect_start_idx += time_offset

        # 비context 컬럼의 고정값
        other_ctx_fixed = {}
        for other_col in self.ctx_cfg["columns"]:
            if other_col != context_column:
                other_ctx_fixed[other_col] = str(self.rng.choice(self._ctx_pool[other_col]))

        rows = []
        for mid in contexts:
            values, mask = context_data[mid]
            valid_indices = np.where(mask)[0]

            for t in valid_indices:
                row = {
                    "chart_id": chart_id,
                    "time_index": int(t + time_offset),
                    "device": device,
                    "step": step,
                    "item": item,
                    "value": float(values[t]),
                }
                # context 컬럼
                row[context_column] = mid
                for other_col, other_val in other_ctx_fixed.items():
                    row[other_col] = other_val
                rows.append(row)

        return ScenarioResult(
            chart_id=chart_id,
            cls=cls,
            device=device,
            step=step,
            item=item,
            context_column=context_column,
            contexts=contexts,
            target=target,
            defect_start_idx=defect_start_idx,
            defect_params=defect_params,
            timeseries_rows=rows,
        )

    def _normalize_normal_target(self, context_data, contexts, target):
        """Normal 클래스에서 target이 fleet 대비 이상해 보이지 않도록 강제 보정.

        Floor 기준 (config.defect.enforcement):
        - normal_max_right_dev_sigma: |target_right_mean - fleet_right_mean| / fleet_std 상한
        - normal_max_right_shift_sigma: |target_right_mean - target_left_mean| / fleet_std 상한
        - target_std <= fleet_std * 1.2
        """
        enf = self.cfg.get("defect", {}).get("enforcement", {})
        max_right_dev = enf.get("normal_max_right_dev_sigma", 0.5)
        max_right_shift = enf.get("normal_max_right_shift_sigma", 0.5)

        fleet_all_vals = []
        fleet_right_vals = []
        for mid in contexts:
            if mid == target:
                continue
            v, m = context_data[mid]
            vi = np.where(m)[0]
            if len(vi) > 0:
                fleet_all_vals.append(v[vi])
        if not fleet_all_vals:
            return

        fleet_all = np.concatenate(fleet_all_vals)
        fleet_mean = float(np.mean(fleet_all))
        fleet_std = max(float(np.std(fleet_all)), 1e-6)

        tv, tm = context_data[target]
        tvi = np.where(tm)[0]
        if len(tvi) == 0:
            return

        # 1) 전체 산포 제어: target std가 fleet std의 1.2배 초과 시 축소
        target_std = float(np.nanstd(tv[tvi]))
        target_mean = float(np.nanmean(tv[tvi]))
        if target_std > fleet_std * 1.2:
            scale = (fleet_std * float(self.rng.uniform(0.85, 1.1))) / target_std
            tv[tvi] = target_mean + (tv[tvi] - target_mean) * scale
            target_mean = float(np.nanmean(tv[tvi]))

        # 2) 전체 평균 제어: fleet_mean에서 너무 멀면 재정렬
        if abs(target_mean - fleet_mean) > fleet_std * 1.0:
            tv[tvi] += (fleet_mean - target_mean) * float(self.rng.uniform(0.7, 1.0))
            target_mean = float(np.nanmean(tv[tvi]))

        # 3) 좌/우 영역 분리 (time_index 70%)
        total_len = len(tv)
        right_start = int(total_len * 0.7)
        right_tvi = tvi[tvi >= right_start]
        left_tvi = tvi[tvi < right_start]
        if len(right_tvi) < 3 or len(left_tvi) < 3:
            context_data[target] = (tv, tm)
            return

        # fleet 우측 평균
        fleet_right_vals = []
        for mid in contexts:
            if mid == target:
                continue
            v, m = context_data[mid]
            vi = np.where(m)[0]
            ri = vi[vi >= right_start]
            if len(ri) > 0:
                fleet_right_vals.append(v[ri])
        if not fleet_right_vals:
            context_data[target] = (tv, tm)
            return
        fleet_right_mean = float(np.mean(np.concatenate(fleet_right_vals)))

        # 4) 우측 fleet 이격 제어 (강제)
        target_right_mean = float(np.nanmean(tv[right_tvi]))
        cur_dev = (target_right_mean - fleet_right_mean) / fleet_std
        if abs(cur_dev) > max_right_dev:
            target_dev = max_right_dev * float(self.rng.uniform(0.0, 0.7))
            new_right_mean = fleet_right_mean + target_dev * (1 if cur_dev >= 0 else -1) * fleet_std
            tv[right_tvi] += (new_right_mean - target_right_mean)

        # 5) 우측 vs 좌측 shift 제어 (강제)
        target_left_mean = float(np.nanmean(tv[left_tvi]))
        target_right_mean = float(np.nanmean(tv[right_tvi]))
        cur_shift = (target_right_mean - target_left_mean) / fleet_std
        if abs(cur_shift) > max_right_shift:
            target_shift = max_right_shift * float(self.rng.uniform(0.0, 0.7))
            new_right_mean = target_left_mean + target_shift * (1 if cur_shift >= 0 else -1) * fleet_std
            tv[right_tvi] += (new_right_mean - target_right_mean)

        # 6) 우측 산포 제어: target_right_std가 좌측의 1.4배 초과 시 축소
        target_right_std = float(np.nanstd(tv[right_tvi]))
        target_left_std = max(float(np.nanstd(tv[left_tvi])), 1e-6)
        if target_right_std > target_left_std * 1.4:
            target_right_mean = float(np.nanmean(tv[right_tvi]))
            scale = (target_left_std * float(self.rng.uniform(0.95, 1.2))) / target_right_std
            tv[right_tvi] = target_right_mean + (tv[right_tvi] - target_right_mean) * scale

        context_data[target] = (tv, tm)

    def _enforce_defect_floor(self, context_data, contexts, target, cls,
                              defect_start: int, defect_end: int):
        """Defect 주입 후 fleet_std 기준으로 강도 floor 강제.

        클래스별 측정 지표:
        - mean_shift: |target_defect_mean - fleet_defect_mean| / fleet_std >= floor
        - standard_deviation: target_defect_std / max(target_left_std, fleet_std) >= floor
        - spike: max|target_defect - fleet_defect_mean| / fleet_std >= floor
        - drift: |target_defect_end_mean - target_defect_start_mean| / fleet_std >= floor
        """
        enf = self.cfg.get("defect", {}).get("enforcement", {})
        if not enf:
            return

        tv, tm = context_data[target]

        # defect 영역 인덱스
        affected = np.where(tm)[0]
        affected = affected[(affected >= defect_start) & (affected < defect_end)]
        if len(affected) < 3:
            return

        # fleet 통계
        fleet_all_vals = []
        fleet_defect_vals = []
        for mid in contexts:
            if mid == target:
                continue
            v, m = context_data[mid]
            vi = np.where(m)[0]
            if len(vi) > 0:
                fleet_all_vals.append(v[vi])
                di = vi[(vi >= defect_start) & (vi < defect_end)]
                if len(di) > 0:
                    fleet_defect_vals.append(v[di])
        if not fleet_all_vals:
            return
        fleet_all = np.concatenate(fleet_all_vals)
        fleet_std = max(float(np.std(fleet_all)), 1e-6)
        fleet_defect_mean = (
            float(np.mean(np.concatenate(fleet_defect_vals)))
            if fleet_defect_vals else float(np.mean(fleet_all))
        )

        if cls == "mean_shift":
            floor = enf.get("mean_shift_floor_sigma", 1.8)
            cur_mean = float(np.mean(tv[affected]))
            cur_dev = (cur_mean - fleet_defect_mean) / fleet_std
            if abs(cur_dev) < floor:
                direction = 1 if cur_dev >= 0 else -1
                target_dev = floor * float(self.rng.uniform(1.0, 1.25)) * direction
                shift_extra = (target_dev - cur_dev) * fleet_std
                tv[affected] += shift_extra

        elif cls == "standard_deviation":
            floor_ratio = enf.get("std_floor_ratio", 2.0)
            # 좌측 영역 std (defect 영역 밖)
            left = np.where(tm)[0]
            left = left[left < defect_start]
            if len(left) >= 3:
                left_std = max(float(np.std(tv[left])), 1e-6)
            else:
                left_std = fleet_std
            ref_std = max(left_std, fleet_std)
            target_std = ref_std * floor_ratio
            cur_std = max(float(np.std(tv[affected])), 1e-6)
            if cur_std < target_std:
                extra_var = target_std ** 2 - cur_std ** 2
                if extra_var > 0:
                    extra_noise = self.rng.normal(0, np.sqrt(extra_var), len(affected))
                    # 패턴 보존 위해 단순 가산
                    tv[affected] += extra_noise

        elif cls == "spike":
            floor = enf.get("spike_floor_sigma", 5.0)
            max_dev = float(np.max(np.abs(tv[affected] - fleet_defect_mean)))
            if max_dev / fleet_std < floor:
                # 가장 큰 1~3개 점을 강제 boost (최소 보장)
                n_boost = min(3, max(1, len(affected) // 5))
                boost_idx = self.rng.choice(affected, size=n_boost, replace=False)
                for bi in boost_idx:
                    direction = float(self.rng.choice([-1, 1]))
                    mag = floor * float(self.rng.uniform(1.0, 1.4)) * fleet_std
                    tv[bi] = fleet_defect_mean + direction * mag

        elif cls == "drift":
            floor = enf.get("drift_floor_sigma", 1.8)
            n = len(affected)
            q = max(2, n // 4)
            start_mean = float(np.mean(tv[affected[:q]]))
            end_mean = float(np.mean(tv[affected[-q:]]))
            cur_drift = (end_mean - start_mean) / fleet_std
            if abs(cur_drift) < floor:
                direction = 1 if cur_drift >= 0 else -1
                target_drift = floor * float(self.rng.uniform(1.0, 1.3)) * direction
                extra = (target_drift - cur_drift) * fleet_std
                # 시간 비례 추가 drift (defect 영역 [defect_start, defect_end) 전체 기준)
                rel = (affected - defect_start) / max(defect_end - defect_start - 1, 1)
                tv[affected] += rel * extra

        context_data[target] = (tv, tm)

    def _enforce_context_floor(self, context_data, contexts, target):
        """Context: |target_mean - fleet_mean| / fleet_std >= floor"""
        enf = self.cfg.get("defect", {}).get("enforcement", {})
        floor = enf.get("context_floor_sigma", 1.8)

        tv, tm = context_data[target]
        tvi = np.where(tm)[0]
        if len(tvi) < 3:
            return

        fleet_vals = []
        for mid in contexts:
            if mid == target:
                continue
            v, m = context_data[mid]
            vi = np.where(m)[0]
            if len(vi) > 0:
                fleet_vals.append(v[vi])
        if not fleet_vals:
            return
        fleet_all = np.concatenate(fleet_vals)
        fleet_mean = float(np.mean(fleet_all))
        fleet_std = max(float(np.std(fleet_all)), 1e-6)

        target_mean = float(np.mean(tv[tvi]))
        cur_dev = (target_mean - fleet_mean) / fleet_std
        if abs(cur_dev) < floor:
            direction = 1 if cur_dev >= 0 else -1
            target_dev = floor * float(self.rng.uniform(1.0, 1.3)) * direction
            shift_extra = (target_dev - cur_dev) * fleet_std
            tv[tvi] += shift_extra

        context_data[target] = (tv, tm)

    def _inject_context(self, context_data, contexts, target):
        """Context anomaly: target의 전체 평균을 fleet 평균에서 멀리 이동 + 산포 증가.
        단순하고 명확하게:
          - 평균: target_mean → fleet_mean ± factor × fleet_std
          - 산포: 기존 산포에 추가 노이즈 더해 scale 배 확대
        """
        dev_cfg = self.ctx_cfg["target_deviation"]

        values, mask = context_data[target]
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return {}

        # Fleet 통계 (전체 멤버 합쳐서)
        fleet_all = []
        for mid in contexts:
            if mid == target:
                continue
            fv, fm = context_data[mid]
            fvi = np.where(fm)[0]
            if len(fvi) > 0:
                fleet_all.append(fv[fvi])
        if not fleet_all:
            return {}
        fleet_vals = np.concatenate(fleet_all)
        fleet_mean = float(np.mean(fleet_vals))
        fleet_std = max(float(np.std(fleet_vals)), 0.01)

        # target 현재 평균
        target_mean = float(np.mean(values[valid]))

        deviation_type = str(self.rng.choice(["mean", "std", "both"]))
        mean_shift = 0.0
        std_scale = 1.0

        if deviation_type in ("mean", "both"):
            factor = float(self.rng.uniform(*dev_cfg["mean_sigma_range"]))
            direction = float(self.rng.choice([-1, 1]))
            # target 평균을 fleet_mean + direction × factor × fleet_std 로 이동
            target_pos = fleet_mean + direction * factor * fleet_std
            mean_shift = target_pos - target_mean
            values[valid] += mean_shift

        if deviation_type in ("std", "both"):
            std_scale = float(self.rng.uniform(*dev_cfg["std_range"]))
            # 기존 target std 측정 후 target_std = fleet_std × scale 되도록 추가 노이즈
            cur_std = max(float(np.std(values[valid])), 1e-6)
            target_std = fleet_std * std_scale
            extra_var = max(target_std ** 2 - cur_std ** 2, 0)
            if extra_var > 0:
                extra_noise = self.rng.normal(0, float(np.sqrt(extra_var)), len(valid))
                values[valid] += extra_noise

        context_data[target] = (values, mask)
        return {
            "deviation_type": deviation_type,
            "mean_shift": float(mean_shift),
            "std_scale": float(std_scale),
            "fleet_std": float(fleet_std),
        }
