"""
시나리오 생성기

구조:
- Chart = device + step + item (고정, 1 chart 정의)
- Legend axis = eqp_id, chamber, recipe (fleet 비교 대상)
- 1 chart당 legend axis별 fleet을 생성
- 학습: 불량 멤버 하이라이트 이미지 = 1 학습 샘플
- 추론: legend axis별 × 종류별 이미지 전부 생성

모든 클래스가 overlay 포맷:
- normal: highlighted member에 불량 없음
- mean_shift/std/spike/drift: highlighted member 시계열 오른쪽 끝에 불량 주입
- context: highlighted member 전체 시계열이 fleet 대비 유의차
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
    # legend/member 정보
    legend_axis: str                # eqp_id, chamber, recipe
    members: List[str]              # fleet 전체 멤버
    highlighted_member: str         # 하이라이트 대상
    # 불량 정보
    defect_start_idx: int
    defect_params: dict
    timeseries_rows: list
    # 정상 하위 유형 / chart 수준 변형 ("", "few_spike", "sparse0.20+clean" 등)
    variant: str = ""


class ScenarioGenerator:

    def __init__(self, config: dict, rng: np.random.Generator = None):
        self.cfg = config
        self.rng = rng or np.random.default_rng()
        self.baseline_gen = BaselineGenerator(config, rng=self.rng)
        self.defect_synth = DefectSynthesizer(config, rng=self.rng)

        self.chart_cfg = config["chart"]
        self.ctx_cfg = config["context"]

        # legend axis별 pool과 count_range
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
                 legend_axis: str = None,
                 device: str = None,
                 step: str = None,
                 item: str = None) -> ScenarioResult:
        """1개 학습 샘플 생성

        Args:
            chart_id: chart ID
            cls: 클래스
            legend_axis: legend axis 강제 지정 (None이면 랜덤)
        """
        # 1) Chart 정의: device + step + item 랜덤 선택
        if device is None or step is None or item is None:
            device, step, item = self.sample_chart_meta()

        # 2) Legend axis 선택
        if legend_axis is None:
            legend_axis = str(self.rng.choice(self.ctx_cfg["columns"]))

        # 3) Member 선택
        pool = self._ctx_pool[legend_axis]
        lo, hi = self._ctx_count[legend_axis]
        # 멤버가 2~3대뿐인 chart. 기본 count_range 는 최소 4라 이런 그림이 아예 없었고,
        # "eqp 2대 중 한쪽 모수가 적으면 그쪽이 불량" 이라는 현업 오검의 배경이 된다.
        small_fleet_tag = None
        ep_cfg = self.cfg.get("episode") or {}
        sl_cfg = ep_cfg.get("single_legend") or {}
        sf_cfg = ep_cfg.get("small_fleet") or {}
        # 멤버가 1개뿐이면 비교 대상이 없다 -> 판정 불가 -> 양호. normal 에만 적용한다.
        if (cls == "normal" and float(sl_cfg.get("prob", 0.0)) > 0
                and self.rng.random() < float(sl_cfg["prob"])):
            count = 1
            small_fleet_tag = "single_legend"
        elif float(sf_cfg.get("prob", 0.0)) > 0 and self.rng.random() < float(sf_cfg["prob"]):
            cw = sf_cfg.get("count_weights")
            if cw:
                opts = [int(k) for k, w in cw.items() if float(w) > 0 and int(k) <= len(pool)]
                pw = np.array([float(cw[k]) for k in cw
                               if float(cw[k]) > 0 and int(k) <= len(pool)], dtype=float)
                count = int(self.rng.choice(opts, p=pw / pw.sum()))
            else:
                sf_lo, sf_hi = sf_cfg.get("count_range", [2, 3])
                count = int(self.rng.integers(int(sf_lo), min(int(sf_hi), len(pool)) + 1))
            small_fleet_tag = f"smallfleet{count}"
        else:
            count = self.rng.integers(lo, min(hi, len(pool)) + 1)
        members = [str(m) for m in self.rng.choice(pool, size=count, replace=False)]

        # 4) Highlighted member 선택
        target = str(self.rng.choice(members))

        # 5) 공유 에피소드 구조 생성
        ref_values, shared_mask, shared_episodes = self.baseline_gen.generate()

        # 6) 각 member: 멤버별 mask 변동 + 독립 값
        context_data = {}
        fleet_means = []

        for mid in members:
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
        target_offsets = self._sample_centered_offsets(len(members), fleet_var["mean_range"])

        for mid, offset in zip(members, target_offsets):
            values, mask = context_data[mid]
            valid = np.where(mask)[0]
            if len(valid) > 0:
                cur = np.nanmean(values[valid])
                tgt = fleet_center + offset
                values[valid] += (tgt - cur)
                context_data[mid] = (values, mask)

        # 7b) 계측 모수 축소. 클래스와 무관하게 적용한다 — normal 에만 걸면
        # "점이 적다 = 정상" 이라는 새 지름길이 생긴다.
        variants = []
        if small_fleet_tag:
            variants.append(small_fleet_tag)
        stopped = self._maybe_early_stop(context_data, target, cls)
        if stopped:
            variants.append(stopped)
        late = None if stopped else self._maybe_late_start(context_data, members, target, cls)
        if late:
            variants.append(late)
        thinned = self._maybe_thin(context_data, members, target, cls,
                                   small_fleet=bool(small_fleet_tag))
        sparse_keep = None
        if thinned is not None:
            tag, sparse_keep = thinned
            variants.append(tag)

        # 8) 불량 주입
        defect_start_idx = -1
        defect_params = {}

        if cls == "normal":
            subtype = self._pick_normal_subtype(
                n_points=int(context_data[target][1].sum()))
            variants.append(subtype)
            self._normalize_normal_target(context_data, members, target)
            if subtype == "few_spike":
                defect_params = self._inject_few_spike_normal(context_data, target)
            elif subtype == "mid_shift":
                defect_params = self._inject_mid_shift_normal(context_data, members, target)
            elif subtype == "right_minor":
                defect_params = self._inject_right_minor_normal(context_data, members, target)
            elif subtype == "recovered":
                defect_params = self._inject_recovered_normal(context_data, members, target)
            elif subtype == "common_mode":
                defect_params = self._inject_common_mode_normal(context_data, members)
            elif subtype == "context_like":
                defect_params = self._apply_context_like_normal(context_data, members, target)
            elif subtype == "loose_target":
                defect_params = self._apply_loose_target(context_data, members, target)
        elif cls == "context":
            defect_params = self._inject_context(context_data, members, target)
            # 사후 검증: context는 전체 평균이 fleet 평균에서 floor 이상 이격
            self._enforce_context_floor(context_data, members, target)
        else:
            # 불량 영역에 유효 포인트 최소 보장 (config의 enforcement.min_defect_points_range)
            enf = self.cfg.get("defect", {}).get("enforcement", {})
            mdp_range = self._resolve_min_defect_points_range(cls, enf)
            MIN_DEFECT_POINTS = int(self.rng.integers(mdp_range[0], mdp_range[1] + 1))
            if sparse_keep is not None:
                # 성긴 chart 는 불량 구간도 같이 성기다. 여기서 되채우면
                # "우측만 촘촘하다" 는 인공 단서가 생겨 오히려 지름길이 된다.
                # 하한 10 — 정상 few_spike 가 우측에서 최대 5개라 그 아래로 내려가면
                # spike 불량 개수가 정상 범위와 겹쳐 라벨이 모호해진다.
                MIN_DEFECT_POINTS = max(10, int(MIN_DEFECT_POINTS * sparse_keep))

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
            for mid in members:
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
                context_data, members, target, cls,
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
            if other_col != legend_axis:
                other_ctx_fixed[other_col] = str(self.rng.choice(self._ctx_pool[other_col]))

        rows = []
        for mid in members:
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
                # legend axis 컬럼
                row[legend_axis] = mid
                for other_col, other_val in other_ctx_fixed.items():
                    row[other_col] = other_val
                rows.append(row)

        return ScenarioResult(
            chart_id=chart_id,
            cls=cls,
            device=device,
            step=step,
            item=item,
            legend_axis=legend_axis,
            members=members,
            highlighted_member=target,
            defect_start_idx=defect_start_idx,
            defect_params=defect_params,
            timeseries_rows=rows,
            variant="+".join(variants),
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

    # ==================================================================
    # 현업 FP 대응 — 정상 클래스 하위 유형 + chart 수준 변형
    #
    # 기존 normal 은 "어디에도 아무 움직임이 없는" 한 종류뿐이라, 모델이
    # "움직이면 불량" 이라는 지름길을 배웠다. 현업의 약한 튐 / 전 멤버 공통
    # 변동 / 적은 계측 모수가 전부 그 지름길에 걸려 FP 로 나온다.
    # config 에 해당 키가 없으면 RNG 를 건드리지 않으므로 기존 데이터셋은
    # 그대로 재현된다.
    # ==================================================================

    def _normal_variant_cfg(self) -> dict:
        return self.cfg.get("normal_variants") or {}

    def _pick_normal_subtype(self, n_points: int | None = None) -> str:
        """정상 하위 유형 추첨.

        모양 기반 유형(mid_shift, recovered, right_minor …)은 그 모양이 보일 만큼
        점이 있어야 의미가 있다. 점 15개짜리 차트에 mid_shift 를 넣으면 "변화 후
        안정 구간이 길다" 는 신호가 6점으로 표현돼 학습에 쓸 수 없는 그림이 된다.
        min_points 미달인 유형은 후보에서 빼고 나머지로 다시 정규화한다.
        """
        cfg = self._normal_variant_cfg()
        weights = cfg.get("weights") or {}
        names = [n for n, w in weights.items() if float(w) > 0]
        if not names:
            return "clean"
        if n_points is not None:
            allowed = [n for n in names
                       if n_points >= int((cfg.get(n) or {}).get("min_points_for_shape", 0))]
            names = allowed or ["clean"]
            if names == ["clean"]:
                return "clean"
        p = np.array([float(weights[n]) for n in names], dtype=float)
        return str(self.rng.choice(names, p=p / p.sum()))

    def _fleet_spread(self, context_data, members, target) -> tuple[float, float]:
        """(within, between) — 멤버 내부 noise 평균, 멤버 평균들의 산포."""
        stds, means = [], []
        for mid in members:
            if mid == target:
                continue
            v, m = context_data[mid]
            vi = np.where(m)[0]
            if len(vi) > 2:
                stds.append(float(np.nanstd(v[vi])))
                means.append(float(np.nanmean(v[vi])))
        if not stds:  # 멤버가 1개뿐이면 비교 대상이 없다 -> target 자기 산포를 쓴다
            tv, tm = context_data[target]
            tvi = np.where(tm)[0]
            within = max(float(np.nanstd(tv[tvi])), 1e-6) if len(tvi) > 2 else 0.05
            return within, within * 0.1
        within = max(float(np.mean(stds)), 1e-6)
        between = max(float(np.std(means)), 1e-6) if len(means) > 1 else within * 0.1
        return within, between

    @staticmethod
    def _defect_region_points(mask, region: float) -> int:
        """불량 구간(우측 region 비율)에 남아 있는 유효 점 수."""
        vi = np.where(mask)[0]
        return int((vi >= int(len(mask) * (1 - region))).sum())

    def _maybe_thin(self, context_data, members, target, cls,
                    small_fleet: bool = False) -> tuple[str, float] | None:
        """계측 모수 축소. mode=chart(전 멤버) / member(일부 멤버만).

        현업에는 특정 설비 하나만 계측이 성긴 경우가 흔한데 기존에는 그런 그림이
        없었다. 불량 클래스에서는 불량 구간 점 수를 먼저 확인해 지나치게 깎이지
        않게 한다.
        """
        cfg = (self.cfg.get("episode") or {}).get("sparse_chart") or {}
        # 멤버가 2~3대인 chart 는 현업에서 '소수인데 불량' 오검이 특히 많다.
        # 그 조합을 확률·강도 양쪽으로 더 만든다.
        prob = float(cfg.get("small_fleet_prob", cfg.get("prob", 0.0)) if small_fleet
                     else cfg.get("prob", 0.0))
        if prob <= 0 or self.rng.random() >= prob:
            return None
        keep_range = cfg.get("small_fleet_keep_ratio_range") if small_fleet else None
        keep = float(self.rng.uniform(*(keep_range or cfg.get("keep_ratio_range", [0.08, 0.30]))))

        modes = cfg.get("mode_weights") or {"chart": 0.5, "member": 0.5}
        names = [m for m, w in modes.items() if float(w) > 0] or ["chart"]
        p = np.array([float(modes[m]) for m in names], dtype=float)
        mode = str(self.rng.choice(names, p=p / p.sum()))

        if mode == "member":
            frac = float(self.rng.uniform(*cfg.get("member_frac_range", [0.2, 0.5])))
            n = max(1, int(round(len(members) * frac)))
            others = [m for m in members if m != target]
            self.rng.shuffle(others)
            if self.rng.random() < float(cfg.get("target_prob", 0.6)):
                chosen = [target] + others[:n - 1]
            else:
                chosen = others[:n]
        else:
            chosen = list(members)

        guard = cls not in ("normal", "context") and target in chosen
        original = context_data[target][1].copy() if guard else None

        min_points = int(cfg.get("min_points", 8))
        for mid in chosen:
            values, mask = context_data[mid]
            vi = np.where(mask)[0]
            n_keep = max(min_points, int(round(len(vi) * keep)))
            if n_keep >= len(vi):
                continue
            drop = self.rng.choice(vi, size=len(vi) - n_keep, replace=False)
            mask[drop] = False
            context_data[mid] = (values, mask)

        # 불량 구간에 점이 너무 적게 남으면 target 만 원래대로 되돌린다.
        # 점 3개짜리 불량은 판정 근거가 못 되므로 그런 표본은 만들지 않는다.
        if guard:
            region = float(cfg.get("defect_region_ratio", 0.12))
            if self._defect_region_points(context_data[target][1], region) < int(
                    cfg.get("min_defect_points", 8)):
                context_data[target] = (context_data[target][0], original)
                if mode == "member" and len(chosen) == 1:
                    return None
                return f"sparse_{mode}{keep:.2f}_fleetonly", keep
        return f"sparse_{mode}{keep:.2f}", keep

    def _inject_mid_shift_normal(self, context_data, members, target) -> dict:
        """target 한 멤버만 시계열 중간에서 레벨이 바뀌고 그대로 유지된다.

        불량 mean_shift 는 우측 끝 구간(14~30%)에만 들어간다. 여기서는 훨씬 앞에서
        바뀌므로 "바뀐 뒤 안정 구간이 얼마나 긴가" 가 구분 신호가 된다 — 절대 위치가
        아니라 폭 특징이라 global average pooling 구조에서도 상대적으로 배우기 쉽다.
        """
        cfg = self._normal_variant_cfg().get("mid_shift") or {}
        values, mask = context_data[target]
        vi = np.where(mask)[0]
        if len(vi) < 12:
            return {}
        start_r = float(self.rng.uniform(*cfg.get("start_ratio_range", [0.25, 0.55])))
        start = int(len(mask) * start_r)
        after, before = vi[vi >= start], vi[vi < start]
        min_side = int(cfg.get("min_side_points", 5))
        if len(after) < min_side or len(before) < min_side:
            return {}
        # 단위는 **눈에 보이는 fleet 밴드 폭**(within + between). within 만 쓰면
        # fleet 이 촘촘할 때 1σ 이동이 밴드를 통째로 벗어나 불량처럼 보인다.
        within, between = self._fleet_spread(context_data, members, target)
        unit = within + between
        ratio = float(self.rng.uniform(*cfg.get("shift_visual_ratio_range", [0.15, 0.5])))
        sigma = ratio * unit / max(within, 1e-9)
        # **앞쪽**을 어긋나게 둔다 — 예전에 벗어나 있다가 중간에 정상으로 돌아온 그림.
        # 뒤쪽을 올리면 우측 끝이 fleet 대비 떠 버려서 그건 불량이 맞다.
        max_mean = float(cfg.get("max_mean_offset_sigma", 1.5))
        frac_before = len(before) / len(vi)
        if frac_before * sigma > max_mean:
            sigma = max_mean / max(frac_before, 1e-6)
        sign = 1 if self.rng.random() < 0.5 else -1
        values[before] += within * sigma * sign
        context_data[target] = (values, mask)
        return {"normal_variant": "mid_shift", "start_ratio": round(start_r, 3),
                "shift_sigma": round(sigma, 3), "visual_ratio": round(ratio, 3),
                "mean_offset_sigma": round(frac_before * sigma, 3),
                "points_after": int(len(after)), "points_before": int(len(before))}

    def _inject_few_spike_normal(self, context_data, target) -> dict:
        """target 에 1~4개짜리 약한 튐.

        불량 spike 는 min_spikes(10)+ / min_magnitude_sigma 이상이므로
        "몇 개 이상, 얼마나 크게" 라는 경계가 생긴다. 기존에는 정상=0개라
        1~9개 구간이 완전히 비어 있어 2~3개도 불량으로 나왔다.
        """
        cfg = self._normal_variant_cfg().get("few_spike") or {}
        values, mask = context_data[target]
        vi = np.where(mask)[0]
        if len(vi) < 5:
            return {}
        # 위치를 먼저 정하고, 위치에 따라 개수 범위를 다르게 준다.
        #   우측 : 불량(10개+)과 같은 구간이므로 개수를 확실히 적게 (1~3)
        #   중/좌: 개수가 많아도 정상 — 불량 구간이 아니기 때문 (1~9)
        start = int(len(mask) * float(cfg.get("right_start_ratio", 0.7)))
        cand_right = vi[vi >= start]
        cand_left = vi[vi < start]
        right = (self.rng.random() < float(cfg.get("right_prob", 0.0))
                 and len(cand_right) >= 1)
        if right:
            lo, hi = cfg.get("right_count_range", [1, 3])
            pool = cand_right
        else:
            lo, hi = cfg.get("count_range", [1, 9])
            pool = cand_left if len(cand_left) >= int(lo) else vi
        n = min(int(self.rng.integers(int(lo), int(hi) + 1)), len(pool))
        idx = self.rng.choice(pool, size=n, replace=False)
        std = max(float(np.nanstd(values[vi])), 1e-6)
        sigmas = self.rng.uniform(*cfg.get("magnitude_sigma_range", [2.0, 3.0]), size=n)
        values[idx] += std * sigmas * self.rng.choice([-1, 1], size=n)
        context_data[target] = (values, mask)
        return {"normal_variant": "few_spike", "num_spikes": int(n),
                "position": "right" if right else "any",
                "avg_magnitude_sigma": round(float(np.mean(sigmas)), 3)}

    def _inject_common_mode_normal(self, context_data, members) -> dict:
        """fleet 전원이 같은 위치에서 같이 움직인다 — target 의 상대 편차는 0.

        기존 데이터에는 회색 fleet 이 이상해지는 그림이 한 장도 없어서
        (불량은 항상 target 1명에게만 주입), 설비 전체가 같이 튄 상황에서
        모든 멤버가 불량으로 잡혔다.
        """
        cfg = self._normal_variant_cfg().get("common_mode") or {}
        kinds = cfg.get("kind_weights") or {"spike": 0.6, "shift": 0.4}
        names = [k for k, w in kinds.items() if float(w) > 0] or ["spike"]
        p = np.array([float(kinds[k]) for k in names], dtype=float)
        kind = str(self.rng.choice(names, p=p / p.sum()))

        stds = []
        for mid in members:
            v, m = context_data[mid]
            vi = np.where(m)[0]
            if len(vi) > 2:
                stds.append(float(np.nanstd(v[vi])))
        base_std = max(float(np.mean(stds)), 1e-6) if stds else 0.05
        total_len = len(context_data[members[0]][1])
        sign = 1 if self.rng.random() < 0.5 else -1

        if kind == "shift":
            ratio = float(self.rng.uniform(*cfg.get("shift_start_ratio_range", [0.45, 0.8])))
            start = int(total_len * ratio)
            sigma = float(self.rng.uniform(*cfg.get("shift_sigma_range", [1.0, 2.0])))
            for mid in members:
                v, m = context_data[mid]
                vi = np.where(m)[0]
                v[vi[vi >= start]] += base_std * sigma * sign
                context_data[mid] = (v, m)
            return {"normal_variant": "common_mode", "kind": "shift",
                    "start_idx": int(start), "shift_sigma": round(sigma, 3)}

        # spike: 같은 time_index 에서 전원이 함께 튄다
        lo, hi = cfg.get("spike_count_range", [3, 12])
        n = min(int(self.rng.integers(int(lo), int(hi) + 1)), total_len)
        shared_idx = self.rng.choice(total_len, size=n, replace=False)
        sigma_lo, sigma_hi = cfg.get("spike_magnitude_sigma_range", [3.0, 6.0])
        sigmas = self.rng.uniform(sigma_lo, sigma_hi, size=n)
        for mid in members:
            v, m = context_data[mid]
            hit = shared_idx[m[shared_idx]]
            if len(hit) == 0:
                continue
            jitter = self.rng.uniform(0.85, 1.15, size=len(hit))
            v[hit] += base_std * sigmas[m[shared_idx]] * jitter * sign
            context_data[mid] = (v, m)
        return {"normal_variant": "common_mode", "kind": "spike",
                "num_spikes": int(n),
                "avg_magnitude_sigma": round(float(np.mean(sigmas)), 3)}

    def _maybe_early_stop(self, context_data, target, cls) -> str | None:
        """target 이 **왼쪽에만 있고 최근(우측)에는 없는** chart. normal 전용.

        최근에 안 돌린 설비. 불량은 전부 우측 끝 구간에 들어가므로 그 구간에 점이
        아예 없으면 판정 대상 자체가 아니다 — context(avg/std)처럼 전 구간을 보는
        것도 마찬가지로 "최근 데이터가 없으면 판정하지 않는다" 로 둔다.
        fleet 은 전 구간 유지해서 "혼자만 최근에 안 돌았다" 가 드러나게 한다.
        """
        if cls != "normal":
            return None
        cfg = (self.cfg.get("episode") or {}).get("early_stop") or {}
        prob = float(cfg.get("prob", 0.0))
        if prob <= 0 or self.rng.random() >= prob:
            return None
        values, mask = context_data[target]
        original = mask.copy()
        head = float(self.rng.uniform(*cfg.get("head_ratio_range", [0.35, 0.80])))
        mask[int(len(mask) * head):] = False
        if int(mask.sum()) < int(cfg.get("min_points", 5)):
            context_data[target] = (values, original)
            return None
        context_data[target] = (values, mask)

        # 최근에 안 돌렸는데 그 전 구간에 열화(drift/shift/std)가 있는 경우도 정상이다.
        # 판정할 최근 데이터가 없으면 옛날 열화만으로 불량이라 할 수 없다.
        tag = f"early_stop{head:.2f}"
        if self.rng.random() < float(cfg.get("degrade_prob", 0.0)):
            vi = np.where(mask)[0]
            if len(vi) >= 12:
                kinds = cfg.get("degrade_kind_weights") or {
                    "drift": 0.45, "mean_shift": 0.35, "standard_deviation": 0.2}
                names = [k for k, w in kinds.items() if float(w) > 0]
                p = np.array([float(kinds[k]) for k in names], dtype=float)
                kind = str(self.rng.choice(names, p=p / p.sum()))
                lo_i, hi_i = int(vi[0]), int(vi[-1]) + 1
                span = hi_i - lo_i
                start = lo_i + int(span * float(self.rng.uniform(0.5, 0.7)))
                new_values, _info = self.defect_synth.inject(
                    values, mask, kind, fleet_range=0.1, fleet_noise_std=0.05,
                    defect_bounds=(start, hi_i))
                context_data[target] = (new_values, mask)
                tag += f"_degraded_{kind}"
        return tag

    def _maybe_late_start(self, context_data, members, target, cls) -> str | None:
        """**우측 끝 구간에서 뒤늦게 시작**한 chart.

        2달치 데이터인데 최근 며칠만 진행된 경우. 남은 점이 전부 불량 구간에 몰려
        있어 오검이 많이 났다. 두 가지를 다 만든다:
          scope=target : 그 설비만 늦게 시작, 다른 eqp 는 계속 진행 (fleet 전 구간)
          scope=all    : 다들 안 돌다가 같이 최근에 시작 (chart 전체가 우측에만)

        불량 클래스에서는 꼬리에 점이 min_defect_points 미만이면 되돌린다 —
        점 몇 개로는 판정하지 않는다.
        """
        cfg = (self.cfg.get("episode") or {}).get("late_start") or {}
        prob = float(cfg.get("prob", 0.0))
        if prob <= 0 or self.rng.random() >= prob:
            return None
        scopes = cfg.get("scope_weights") or {"target": 0.5, "all": 0.5}
        names = [k for k, w in scopes.items() if float(w) > 0] or ["target"]
        p = np.array([float(scopes[k]) for k in names], dtype=float)
        scope = str(self.rng.choice(names, p=p / p.sum()))

        tail = float(self.rng.uniform(*cfg.get("tail_ratio_range", [0.03, 0.25])))
        chosen = list(members) if scope == "all" else [target]
        original = {mid: context_data[mid][1].copy() for mid in chosen}
        start = None
        for mid in chosen:
            values, mask = context_data[mid]
            start = int(len(mask) * (1.0 - tail))
            mask[:start] = False
            context_data[mid] = (values, mask)

        mask_now = context_data[target][1]
        kept = int(mask_now.sum())
        too_few = kept < int(cfg.get("min_points", 3)) or (
            cls != "normal" and kept < int(cfg.get("min_defect_points", 10)))
        # mean_shift / standard_deviation / drift 는 '우측 구간 vs 자기 좌측' 으로
        # 정의된다. 늦게 시작해서 좌측 baseline 이 없으면 불량 자체가 성립하지 않는다
        # (그냥 다른 레벨에서 시작한 것과 구분 불가) -> 되돌린다.
        # 꼬리 전체 점 수가 충분해도 그게 넓게 퍼져 있으면 실제 불량 구간(우측 12%)에는
        # 몇 개 안 남는다. spike 개수가 정상 범위까지 내려가 라벨이 겹치므로 여기도 본다.
        if not too_few and cls != "normal":
            region = float(cfg.get("defect_region_ratio", 0.12))
            if self._defect_region_points(mask_now, region) < int(
                    cfg.get("min_defect_points", 10)):
                too_few = True
        if not too_few and cls in ("mean_shift", "standard_deviation", "drift"):
            vi = np.where(mask_now)[0]
            baseline = int((vi < int(len(mask_now) * 0.7)).sum())
            if baseline < int(cfg.get("min_baseline_points", 15)):
                too_few = True
        if too_few:
            for mid, m in original.items():
                context_data[mid] = (context_data[mid][0], m)
            return None
        return f"late_start_{scope}{tail:.2f}"

    def _inject_right_minor_normal(self, context_data, members, target) -> dict:
        """우측 끝에서 **소량** 변동 — 불량 판정 하한 아래로 유지한다.

        기존 normal 은 우측이 거의 평평하도록 강제(normal_max_right_*_sigma 0.5)돼
        "우측이 조금이라도 움직이면 불량" 이 돼 버렸다. 불량 floor 와 그 0.5 사이의
        빈 구간을 정상으로 채워 경계를 만든다.
        """
        cfg = self._normal_variant_cfg().get("right_minor") or {}
        values, mask = context_data[target]
        vi = np.where(mask)[0]
        if len(vi) < 12:
            return {}
        ratio = float(self.rng.uniform(*cfg.get("start_ratio_range", [0.72, 0.90])))
        start = int(len(mask) * ratio)
        right = vi[vi >= start]
        left = vi[vi < start]
        if len(right) < int(cfg.get("min_points", 5)) or len(left) < 5:
            return {}

        kinds = cfg.get("kind_weights") or {"shift": 0.6, "spread": 0.4}
        names = [k for k, w in kinds.items() if float(w) > 0] or ["shift"]
        p = np.array([float(kinds[k]) for k in names], dtype=float)
        kind = str(self.rng.choice(names, p=p / p.sum()))
        within, between = self._fleet_spread(context_data, members, target)

        if kind == "shift":
            vr = float(self.rng.uniform(*cfg.get("shift_visual_ratio_range", [0.12, 0.4])))
            delta = vr * (within + between)
            sign = 1 if self.rng.random() < 0.5 else -1
            values[right] += delta * sign
            detail = {"kind": "shift", "visual_ratio": round(vr, 3),
                      "shift_sigma": round(delta / max(within, 1e-9), 3)}
        else:
            scale = float(self.rng.uniform(*cfg.get("spread_scale_range", [1.2, 1.7])))
            mean = float(np.nanmean(values[right]))
            values[right] = mean + (values[right] - mean) * scale
            detail = {"kind": "spread", "spread_scale": round(scale, 3)}

        context_data[target] = (values, mask)
        return {"normal_variant": "right_minor", "start_ratio": round(ratio, 3),
                "points_right": int(len(right)), **detail}

    def _inject_recovered_normal(self, context_data, members, target) -> dict:
        """중간에서 이상이 났다가 **되돌아온** 정상.

        mean_shift / standard_deviation / drift / spike 를 시계열 중간 구간에만
        넣고 그 뒤는 baseline 그대로 둔다 — 우측 끝이 깨끗하면 양호라는 규칙을
        데이터에 새긴다. 진폭은 **진짜 불량과 같은 설정**을 쓴다. 약하게 넣으면
        모델이 "작으면 정상" 을 배울 뿐 복귀 여부를 안 본다.
        """
        cfg = self._normal_variant_cfg().get("recovered") or {}
        values, mask = context_data[target]
        vi = np.where(mask)[0]
        if len(vi) < 20:
            return {}

        kinds = cfg.get("kind_weights") or {
            "mean_shift": 0.3, "standard_deviation": 0.25, "drift": 0.25, "spike": 0.2}
        names = [k for k, w in kinds.items() if float(w) > 0] or ["mean_shift"]
        p = np.array([float(kinds[k]) for k in names], dtype=float)
        kind = str(self.rng.choice(names, p=p / p.sum()))

        total = len(mask)
        start_r = float(self.rng.uniform(*cfg.get("start_ratio_range", [0.15, 0.42])))
        span_r = float(self.rng.uniform(*cfg.get("span_ratio_range", [0.12, 0.26])))
        end_r = min(start_r + span_r, float(cfg.get("max_end_ratio", 0.65)))
        start, end = int(total * start_r), int(total * end_r)
        inside = vi[(vi >= start) & (vi < end)]
        after = vi[vi >= end]
        if len(inside) < int(cfg.get("min_points", 8)) or len(after) < int(
                cfg.get("min_recovery_points", 12)):
            return {}

        fleet_vals = []
        for mid in members:
            if mid == target:
                continue
            fv, fm = context_data[mid]
            fvi = np.where(fm)[0]
            if len(fvi) > 0:
                fleet_vals.extend(fv[fvi].tolist())
        if fleet_vals:
            p5, p95 = np.percentile(fleet_vals, [5, 95])
            fleet_range = max(float(p95 - p5), 0.05)
            fleet_noise = max(float(np.std(fleet_vals)), 0.01)
        else:
            fleet_range, fleet_noise = 0.1, 0.05

        if kind == "drift_return":
            # 좌~중 서서히 열화 -> 정점 -> 우측으로 가며 서서히 원복 (봉우리 모양).
            # 실제 fab 에서 가장 흔한 회복 패턴이고, 계단식 복귀보다 자연스럽다.
            std = max(float(np.nanstd(values[vi])), 1e-6)
            peak = float(self.rng.uniform(*cfg.get("drift_return_sigma_range", [1.5, 3.0])))
            sign = 1 if self.rng.random() < 0.5 else -1
            mid = (start + end) / 2.0
            for t in inside:
                frac = 1.0 - abs(t - mid) / max((end - start) / 2.0, 1e-6)
                values[t] += std * peak * sign * max(frac, 0.0)
            context_data[target] = (values, mask)
            return {"normal_variant": "recovered", "kind": "drift_return",
                    "peak_sigma": round(peak, 3),
                    "start_ratio": round(start_r, 3), "end_ratio": round(end_r, 3),
                    "points_inside": int(len(inside)), "points_after": int(len(after))}

        new_values, info = self.defect_synth.inject(
            values, mask, kind, fleet_range=fleet_range,
            fleet_noise_std=fleet_noise, defect_bounds=(start, end))
        context_data[target] = (new_values, mask)
        return {"normal_variant": "recovered", "kind": kind,
                "start_ratio": round(start_r, 3), "end_ratio": round(end_r, 3),
                "points_inside": int(len(inside)), "points_after": int(len(after))}

    def _apply_context_like_normal(self, context_data, members, target) -> dict:
        """fleet 에서 **조금** 떨어져 있는 정상 — context 느낌은 나지만 불량은 아니다.

        기존 normal 은 target 전체 평균이 fleet 평균의 0.6σ 이내로 강제되고
        (_normalize_normal_target), context 불량은 mean_min_within_ratio 이상 떨어져야
        한다. 그 사이 구간이 비어 있어 "조금만 치우쳐도 context 불량" 이 됐다.
        여기서 그 구간을 채운다. 상한은 context 하한 아래로 clamp 한다.
        """
        cfg = self._normal_variant_cfg().get("context_like") or {}
        values, mask = context_data[target]
        vi = np.where(mask)[0]
        if len(vi) < 5:
            return {}
        within, _between = self._fleet_spread(context_data, members, target)

        lo, hi = cfg.get("mean_offset_sigma_range", [0.8, 2.0])
        # context 불량 하한(mean_min_within_ratio) 아래로 반드시 유지
        ceiling = float(self.ctx_cfg.get("target_deviation", {}).get(
            "mean_min_within_ratio", 1.8)) * float(cfg.get("max_ratio_of_floor", 0.75))
        hi = min(float(hi), ceiling)
        if hi <= float(lo):
            return {"normal_variant": "context_like", "offset_sigma": 0.0, "capped": True}

        offset = float(self.rng.uniform(float(lo), hi))
        sign = 1 if self.rng.random() < 0.5 else -1
        values[vi] += within * offset * sign
        context_data[target] = (values, mask)
        return {"normal_variant": "context_like",
                "offset_sigma": round(offset, 3), "ceiling_sigma": round(ceiling, 3)}

    def _apply_loose_target(self, context_data, members, target) -> dict:
        """target 산포를 전 구간 균일하게 **조금만** 키운다 (평균 유지).

        이웃보다 월등히 넓은 것은 context 불량이 맡는다. context 는
        target_std >= fleet_visual_spread * std_min_scale(1.25) 를 하한으로 쓰므로,
        여기서는 fleet_visual_spread * max_visual_ratio(<1) 로 상한을 걸어 겹침을
        원천 차단한다. std 불량과는 또 다르게, 이쪽은 좌우가 균일해서
        '우측이 자기 좌측보다 넓다' 판별에도 걸리지 않는다.
        """
        cfg = self._normal_variant_cfg().get("loose_target") or {}
        values, mask = context_data[target]
        vi = np.where(mask)[0]
        if len(vi) < 5:
            return {}
        scale = float(self.rng.uniform(*cfg.get("std_scale_range", [1.15, 1.6])))
        within, between = self._fleet_spread(context_data, members, target)
        cap = (within + between) * float(cfg.get("max_visual_ratio", 1.0))

        cur = max(float(np.nanstd(values[vi])), 1e-6)
        desired = min(cur * scale, cap)
        if desired <= cur * 1.02:
            return {"normal_variant": "loose_target", "std_scale": 1.0, "capped": True}
        mean = float(np.nanmean(values[vi]))
        values[vi] = mean + (values[vi] - mean) * (desired / cur)
        context_data[target] = (values, mask)
        return {"normal_variant": "loose_target",
                "std_scale": round(desired / cur, 3),
                "vs_fleet_within": round(desired / within, 3),
                "capped": bool(desired < cur * scale * 0.999)}

    def _normalize_normal_target(self, context_data, members, target):
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
        for mid in members:
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
        for mid in members:
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

    def _enforce_defect_floor(self, context_data, members, target, cls,
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
        for mid in members:
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

    def _enforce_context_floor(self, context_data, members, target):
        """Context: |target_mean - fleet_mean| / fleet_std >= floor"""
        enf = self.cfg.get("defect", {}).get("enforcement", {})
        floor = enf.get("context_floor_sigma", 1.8)

        tv, tm = context_data[target]
        tvi = np.where(tm)[0]
        if len(tvi) < 3:
            return

        fleet_vals = []
        for mid in members:
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

    def _inject_context(self, context_data, members, target):
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
        for mid in members:
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

        dt_w = dev_cfg.get("deviation_type_weights")
        if dt_w:
            _names = [k for k, w in dt_w.items() if float(w) > 0] or ["mean"]
            _p = np.array([float(dt_w[k]) for k in _names], dtype=float)
            deviation_type = str(self.rng.choice(_names, p=_p / _p.sum()))
        else:
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
            for mid in members:
                if mid == target:
                    continue
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
            # fleet_member_stds 는 위에서 이미 target 을 뺀 목록이다. members 와 zip 하면
            # 길이가 어긋나 엉뚱한 멤버가 걸러졌고, 멤버가 2개일 때는 빈 목록이 됐다.
            _fleet_within_avg = float(np.mean(fleet_member_stds)) if fleet_member_stds else 0.01
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
