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

    def sample_chart_meta(self) -> tuple[str, str, str]:
        """Sample a single device/step/item triple."""
        device = str(self.rng.choice(self.chart_cfg["device"]["ids"]))
        step = str(self.rng.choice(self.chart_cfg["step"]["ids"]))
        item = str(self.rng.choice(self.chart_cfg["item"]["ids"]))
        return device, step, item

    def generate(self, chart_id: str, cls: str,
                 context_column: str = None,
                 device: str = None,
                 step: str = None,
                 item: str = None) -> ScenarioResult:
        """1개 학습 샘플 생성

        Args:
            chart_id: chart ID
            cls: 클래스
            context_column: context 컬럼 강제 지정 (None이면 랜덤)
        """
        # 1) Chart 정의: device + step + item 랜덤 선택
        if device is None or step is None or item is None:
            device, step, item = self.sample_chart_meta()

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
        # mean_range는 "각 멤버의 center offset"이 아니라 fleet 전체 band 폭으로 해석한다.
        # 이전 구현처럼 각 멤버를 독립적으로 ±range 안에 배치하면 실제 fleet range가
        # 멤버 수가 늘수록 거의 2배까지 벌어져 시각적으로 너무 넓어졌다.
        fleet_center = np.mean(fleet_means) if fleet_means else 0.0
        fleet_var = self.ctx_cfg["fleet_variation"]
        target_offsets = self._sample_centered_offsets(len(contexts), fleet_var["mean_range"])

        for mid, offset in zip(contexts, target_offsets):
            values, mask = context_data[mid]
            valid = np.where(mask)[0]
            if len(valid) > 0:
                cur = np.nanmean(values[valid])
                tgt = fleet_center + offset
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
            mdp_range = self._resolve_min_defect_points_range(cls, enf)
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

            # === Fleet visible range + std 계산 ===
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
                fleet_noise_std = max(float(np.std(all_fleet_vals)), 0.01)
            else:
                fleet_visible_range = 0.1
                fleet_noise_std = 0.05

            # 불량 주입 (fleet_range + fleet_noise_std 전달)
            new_values, info = self.defect_synth.inject(
                values, mask, cls,
                fleet_range=fleet_visible_range,
                fleet_noise_std=fleet_noise_std,
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
                        values, mask, cls,
                        fleet_range=fleet_visible_range,
                        fleet_noise_std=fleet_noise_std,
                        defect_bounds=(start, end),
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

    def _resolve_min_defect_points_range(self, cls: str, enforcement_cfg: dict) -> List[int]:
        """Return class-specific visible-point floor when configured."""
        defect_cfg = self.cfg.get("defect", {}).get(cls, {})
        override = defect_cfg.get("min_defect_points_range")
        if override:
            return [int(override[0]), int(override[1])]
        base = enforcement_cfg.get("min_defect_points_range", [12, 25])
        return [int(base[0]), int(base[1])]

    def _sample_centered_offsets(self, count: int, band_range: List[float]) -> np.ndarray:
        """Zero-mean member offsets whose total range stays inside band_range.

        band_range is interpreted as the full fleet mean spread, matching the
        config comment "fleet 멤버 평균 차이".
        """
        if count <= 1:
            return np.zeros(count, dtype=float)

        band = float(self.rng.uniform(*band_range))

        # Zero-mean raw offsets, then rescale so the final fleet range equals the
        # sampled band instead of each member independently drifting by ±band.
        raw = self.rng.normal(0.0, 1.0, count)
        raw -= float(np.mean(raw))

        raw_range = float(np.max(raw) - np.min(raw))
        if raw_range < 1e-9:
            raw = np.linspace(-0.5, 0.5, count, dtype=float)
            raw_range = float(np.max(raw) - np.min(raw))

        return raw * (band / raw_range)

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
        fleet_member_means = []  # 각 멤버의 평균 (between-member spread 계산용)
        fleet_within_stds = []   # 각 멤버의 내부 std (within-member noise)
        for mid in contexts:
            if mid == target:
                continue
            v, m = context_data[mid]
            vi = np.where(m)[0]
            if len(vi) > 0:
                vals = v[vi]
                fleet_all_vals.append(vals)
                fleet_member_means.append(float(np.mean(vals)))
                fleet_within_stds.append(float(np.std(vals)))
        if not fleet_all_vals:
            return

        fleet_all = np.concatenate(fleet_all_vals)
        fleet_mean = float(np.mean(fleet_all))
        fleet_std = max(float(np.std(fleet_all)), 1e-6)       # 전체 (within + between)
        fleet_within_std = max(float(np.mean(fleet_within_stds)), 1e-6) if fleet_within_stds else fleet_std

        tv, tm = context_data[target]
        tvi = np.where(tm)[0]
        if len(tvi) == 0:
            return

        # 1) 전체 산포 제어: target std가 fleet within_std 의 1.2배 초과 시 축소
        # (between 편차 포함한 fleet_std 대신 within_std 기준 → 더 엄격)
        target_std = float(np.nanstd(tv[tvi]))
        target_mean = float(np.nanmean(tv[tvi]))
        if target_std > fleet_within_std * 1.2:
            scale = (fleet_within_std * float(self.rng.uniform(0.85, 1.1))) / target_std
            tv[tvi] = target_mean + (tv[tvi] - target_mean) * scale
            target_mean = float(np.nanmean(tv[tvi]))

        # 2) 전체 평균 제어: fleet_mean 에서 fleet_within_std × 0.6 초과 이탈 시 재정렬
        # (normal 은 fleet 중심에 가까워야 함 → 엄격)
        mean_gap_thr = fleet_within_std * 0.6
        if abs(target_mean - fleet_mean) > mean_gap_thr:
            tv[tvi] += (fleet_mean - target_mean) * float(self.rng.uniform(0.8, 1.0))
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
        fleet_within_stds = []
        for mid in contexts:
            if mid == target:
                continue
            v, m = context_data[mid]
            vi = np.where(m)[0]
            if len(vi) > 0:
                member_vals = v[vi]
                fleet_all_vals.append(member_vals)
                fleet_within_stds.append(float(np.std(member_vals)))
                di = vi[(vi >= defect_start) & (vi < defect_end)]
                if len(di) > 0:
                    fleet_defect_vals.append(v[di])
        if not fleet_all_vals:
            return
        fleet_all = np.concatenate(fleet_all_vals)
        fleet_std = max(float(np.std(fleet_all)), 1e-6)
        fleet_within_std = max(float(np.mean(fleet_within_stds)), 1e-6) if fleet_within_stds else fleet_std
        fleet_defect_mean = (
            float(np.mean(np.concatenate(fleet_defect_vals)))
            if fleet_defect_vals else float(np.mean(fleet_all))
        )

        left = np.where(tm)[0]
        left = left[left < defect_start]
        if len(left) >= 3:
            left_std = max(float(np.std(tv[left])), 1e-6)
        else:
            left_std = fleet_within_std

        # 실전성 기준: fleet 전체 spread 가 아니라 baseline 내부 noise 기준으로만 바닥값 보정
        ref_noise = max(left_std, fleet_within_std)

        if cls == "mean_shift":
            floor = enf.get("mean_shift_floor_sigma", 1.8)
            self_floor = enf.get("mean_shift_self_floor_sigma", floor)
            cur_mean = float(np.mean(tv[affected]))
            left_mean = float(np.mean(tv[left])) if len(left) >= 3 else fleet_defect_mean

            # Mean shift는 사람 눈 기준으로도 "오른쪽 구간이 baseline에서 이동"해 보여야 한다.
            # 그래서 fleet 대비 gap 뿐 아니라 target 자신의 pre-defect 평균과의 shift도 함께 강제한다.
            cur_dev = (cur_mean - fleet_defect_mean) / ref_noise
            cur_self_shift = (cur_mean - left_mean) / max(left_std, 1e-6)
            if abs(cur_dev) < floor or abs(cur_self_shift) < self_floor:
                direction = 1.0
                if abs(cur_self_shift) > 1e-6:
                    direction = 1.0 if cur_self_shift >= 0 else -1.0
                elif abs(cur_dev) > 1e-6:
                    direction = 1.0 if cur_dev >= 0 else -1.0

                fleet_target_mean = (
                    fleet_defect_mean
                    + direction
                    * floor
                    * float(self.rng.uniform(1.02, 1.08))
                    * ref_noise
                )
                self_target_mean = (
                    left_mean
                    + direction
                    * self_floor
                    * float(self.rng.uniform(1.02, 1.08))
                    * max(left_std, 1e-6)
                )
                target_mean = (
                    max(fleet_target_mean, self_target_mean)
                    if direction > 0
                    else min(fleet_target_mean, self_target_mean)
                )
                tv[affected] += target_mean - cur_mean

        elif cls == "standard_deviation":
            floor_ratio = enf.get("std_floor_ratio", 2.0)
            target_std = ref_noise * floor_ratio * float(self.rng.uniform(1.02, 1.08))
            cur_std = max(float(np.std(tv[affected])), 1e-6)
            if cur_std < target_std:
                cur_mean = float(np.mean(tv[affected]))
                centered = tv[affected] - cur_mean
                if cur_std < 1e-6:
                    centered = self.rng.normal(0, 1.0, len(affected))
                    cur_std = max(float(np.std(centered)), 1e-6)
                # 기대값이 아니라 실제 affected std가 floor를 넘도록 직접 스케일링
                tv[affected] = cur_mean + centered * (target_std / cur_std)

        elif cls == "spike":
            floor = enf.get("spike_floor_sigma", 5.0)
            deviations = np.abs(tv[affected] - fleet_defect_mean)
            max_dev = float(np.max(deviations))
            if max_dev / ref_noise < floor:
                # 이미 생성된 spike 중 가장 튄 점만 최소한으로 강화한다.
                n_boost = min(2, max(1, len(affected) // 8))
                boost_order = np.argsort(deviations)
                boost_idx = affected[boost_order[-n_boost:]]
                target_mag = floor * float(self.rng.uniform(1.02, 1.10)) * ref_noise
                for bi in boost_idx:
                    cur_delta = float(tv[bi] - fleet_defect_mean)
                    direction = 1.0 if cur_delta >= 0 else -1.0
                    if abs(cur_delta) < target_mag:
                        tv[bi] = fleet_defect_mean + direction * target_mag

        elif cls == "drift":
            floor = enf.get("drift_floor_sigma", 1.8)
            drift_cfg = self.cfg.get("defect", {}).get("drift", {})
            visual_floor = max(float(drift_cfg.get("visual_floor_sigma", 0.0)), float(floor))
            n = len(affected)
            q = max(2, n // 4)
            rel = (affected - defect_start) / max(defect_end - defect_start - 1, 1)
            rel_gap = max(float(np.mean(rel[-q:]) - np.mean(rel[:q])), 1e-6)
            start_mean = float(np.mean(tv[affected[:q]]))
            end_mean = float(np.mean(tv[affected[-q:]]))
            drift_ref_noise = max(left_std, 1e-6)
            cur_drift = (end_mean - start_mean) / drift_ref_noise
            if abs(cur_drift) < visual_floor:
                direction = 1 if cur_drift >= 0 else -1
                target_drift = visual_floor * float(self.rng.uniform(1.02, 1.10)) * direction
                target_delta = target_drift * drift_ref_noise
                current_delta = end_mean - start_mean
                # quarter-window 측정치 기준으로 목표 drift를 맞추려면
                # rel 자체가 아니라 (끝 평균 rel - 시작 평균 rel) 만큼 보정량을 나눠야 한다.
                extra = (target_delta - current_delta) / rel_gap
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
        """Context anomaly: target의 전체 평균을 fleet 대비 이동 + 산포 증가.

        reference 모드 (config target_deviation.reference):
          - "within" (기본): fleet_std (군내 noise) 기준 — 기존 방식
          - "between": fleet 멤버별 mean 의 std (군간 산포) 기준 — 더 현실적
        """
        dev_cfg = self.ctx_cfg["target_deviation"]
        mean_ref_mode = dev_cfg.get("mean_reference", dev_cfg.get("reference", "within"))
        std_ref_mode = dev_cfg.get("std_reference", dev_cfg.get("reference", "within"))

        values, mask = context_data[target]
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return {}

        # Fleet 통계 (전체 멤버 합쳐서)
        fleet_all = []
        fleet_member_means = []
        fleet_member_stds = []
        for mid in contexts:
            if mid == target:
                continue
            fv, fm = context_data[mid]
            fvi = np.where(fm)[0]
            if len(fvi) > 0:
                fleet_all.append(fv[fvi])
                fleet_member_means.append(float(np.mean(fv[fvi])))
                fleet_member_stds.append(float(np.std(fv[fvi])))
        if not fleet_all:
            return {}
        fleet_vals = np.concatenate(fleet_all)
        fleet_mean = float(np.mean(fleet_vals))
        fleet_std = max(float(np.std(fleet_vals)), 0.01)  # 군내 (within)

        # 군간 (between) 산포
        fleet_between_std = max(float(np.std(fleet_member_means)), 0.001) if len(fleet_member_means) > 1 else fleet_std * 0.1

        # reference 선택 (mean/std 각각)
        mean_ref_std = fleet_between_std if mean_ref_mode == "between" else fleet_std
        std_ref_std = fleet_std  # std 는 항상 within (between 쓰면 값이 너무 작아 효과 없음)
        if std_ref_mode == "within":
            std_ref_std = fleet_std
        elif std_ref_mode == "between":
            std_ref_std = fleet_between_std
        ref_std = mean_ref_std  # 하위호환

        # target 현재 평균
        target_mean = float(np.mean(values[valid]))

        deviation_type = str(self.rng.choice(["mean", "std", "both"]))
        mean_shift = 0.0
        std_scale = 1.0

        # --- mean 주입: fleet 개별 std (noise) 도 고려한 교차 보정 ---
        mean_target_nsigma = 0.0
        fleet_std_avg = float(np.mean(fleet_member_stds)) if fleet_member_stds else fleet_std
        fleet_std_between = max(float(np.std(fleet_member_stds)), 0.001) if len(fleet_member_stds) > 1 else 0.005

        if deviation_type in ("mean", "both"):
            target_nsigma = float(self.rng.uniform(*dev_cfg["mean_sigma_range"]))
            direction = float(self.rng.choice([-1, 1]))

            # 기본: delta = target_nsigma × fleet_between_std
            delta = target_nsigma * fleet_between_std

            # 교차 보정: fleet 개별 noise (std) 가 크면 delta 를 키워야 점들이 안 겹침
            min_within_ratio = float(dev_cfg.get("mean_min_within_ratio", 0))
            if min_within_ratio > 0:
                min_delta = fleet_std_avg * min_within_ratio
                delta = max(delta, min_delta)

            # target 이동 (fleet 은 건드리지 않음 — 자연 변동만)
            values[valid] += direction * delta - (target_mean - fleet_mean)
            mean_shift = float(np.mean(values[valid])) - target_mean
            mean_target_nsigma = target_nsigma

        # --- std 주입: fleet mean spread 도 고려한 교차 보정 ---
        std_target_nsigma = 0.0
        if deviation_type in ("std", "both"):
            std_key = "std_sigma_range" if "std_sigma_range" in dev_cfg else "std_range"
            target_nsigma = float(self.rng.uniform(*dev_cfg[std_key]))

            # 기본: target_std = fleet_std_avg + target_nsigma × fleet_std_between
            desired_std = fleet_std_avg + target_nsigma * fleet_std_between

            # 교차 보정: fleet mean spread 가 크면 전체 scatter 가 넓어서 std 차이 안 보임
            # → target_std 는 (fleet 전체 시각 산포) × min_scale 이상이어야 함
            # fleet 시각 산포 ≈ fleet_between_std (mean 퍼짐) + fleet_std_avg (개별 noise)
            min_scale = float(dev_cfg.get("std_min_scale", 0))
            if min_scale > 0:
                fleet_visual_spread = fleet_between_std + fleet_std_avg
                min_desired_std = fleet_visual_spread * min_scale
                desired_std = max(desired_std, min_desired_std)

            cur_std = max(float(np.std(values[valid])), 1e-6)
            extra_var = max(desired_std ** 2 - cur_std ** 2, 0)
            if extra_var > 0:
                extra_noise = self.rng.normal(0, float(np.sqrt(extra_var)), len(valid))
                values[valid] += extra_noise
            std_scale = desired_std / max(fleet_std_avg, 1e-6)
            std_target_nsigma = target_nsigma

        # --- 사후 검증 + min_nsigma enforcement ---
        min_mean_nsigma = float(dev_cfg.get("min_mean_nsigma", 0))
        min_std_nsigma = float(dev_cfg.get("min_std_nsigma", 0))
        fleet_std_between_post = max(float(np.std(fleet_member_stds)), 0.001) if len(fleet_member_stds) > 1 else 0.005
        fleet_std_avg_post = float(np.mean(fleet_member_stds))

        # mean enforcement — 실제 fleet stats 기준으로 재측정 + 반복 boost
        for _enforce_iter in range(3):  # 최대 3회 반복
            # 현재 fleet between_std 재계산 (fleet 값은 안 변하므로 동일하지만 명시적)
            _fm_means = []
            for mid in contexts:
                if mid == target: continue
                fv, fm_mask = context_data[mid]
                fvi = np.where(fm_mask)[0]
                if len(fvi) > 0:
                    _fm_means.append(float(np.mean(fv[fvi])))
            _fleet_mean = float(np.mean(_fm_means)) if _fm_means else fleet_mean
            _fleet_bstd = max(float(np.std(_fm_means)), 0.001) if len(_fm_means) > 1 else 0.001

            post_target_mean = float(np.mean(values[valid]))
            post_mean_nsigma = abs(post_target_mean - _fleet_mean) / _fleet_bstd

            if deviation_type not in ("mean", "both") or min_mean_nsigma <= 0:
                break

            # 추가: 가장 가까운 fleet 멤버보다 fleet_within_std 이상 떨어져야 함
            _closest_fleet = min(_fm_means, key=lambda m: abs(m - post_target_mean))
            _gap_from_closest = abs(post_target_mean - _closest_fleet)
            _fleet_within_avg = float(np.mean([ms_val for mid, ms_val in zip(contexts, fleet_member_stds) if mid != target])) if fleet_member_stds else 0.01
            _min_gap = _fleet_within_avg * float(dev_cfg.get("mean_min_within_ratio", 1.5))

            need_boost = False
            if post_mean_nsigma < min_mean_nsigma:
                need_boost = True
            if _gap_from_closest < _min_gap:
                need_boost = True

            if need_boost:
                direction_sign = 1.0 if post_target_mean >= _fleet_mean else -1.0
                # fleet 평균 기준 min_nsigma 와 closest 멤버 기준 min_gap 중 큰 값
                needed_from_avg = min_mean_nsigma * _fleet_bstd
                needed_from_closest = abs(_closest_fleet - _fleet_mean) + _min_gap
                needed_delta = max(needed_from_avg, needed_from_closest)
                current_delta = abs(post_target_mean - _fleet_mean)
                boost = needed_delta - current_delta
                if boost > 0:
                    values[valid] += direction_sign * boost
            else:
                break

        # std enforcement
        post_target_std = float(np.std(values[valid]))
        post_std_nsigma = abs(post_target_std - fleet_std_avg_post) / max(fleet_std_between_post, 1e-6)
        if deviation_type in ("std", "both") and min_std_nsigma > 0 and post_std_nsigma < min_std_nsigma:
            desired_std = fleet_std_avg_post + min_std_nsigma * fleet_std_between_post
            cur_std = max(float(np.std(values[valid])), 1e-6)
            extra_var = max(desired_std ** 2 - cur_std ** 2, 0)
            if extra_var > 0:
                extra_noise = self.rng.normal(0, float(np.sqrt(extra_var)), len(valid))
                values[valid] += extra_noise
            post_target_std = float(np.std(values[valid]))
            post_std_nsigma = abs(post_target_std - fleet_std_avg_post) / max(fleet_std_between_post, 1e-6)

        context_data[target] = (values, mask)
        return {
            "deviation_type": deviation_type,
            "mean_shift": float(np.mean(values[valid]) - target_mean),
            "std_scale": float(post_target_std / max(fleet_std_avg_post, 1e-6)),
            "fleet_std": float(fleet_std),
            "fleet_between_std": float(fleet_between_std),
            "fleet_std_between": float(fleet_std_between_post),
            "mean_target_nsigma": float(mean_target_nsigma),
            "std_target_nsigma": float(std_target_nsigma),
            "post_mean_nsigma": float(post_mean_nsigma),
            "post_std_nsigma": float(post_std_nsigma),
        }
