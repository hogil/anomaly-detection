"""
불량 합성기 (Defect Synthesizer)

정상 베이스라인 시계열의 오른쪽 끝(마지막 영역)에 불량값을 더해서 생성.
불량 강도는 베이스라인 노이즈(std)에 비례 → 노이즈 강하면 불량도 강하게.

4가지 불량 유형:
1. Mean Shift: 특정 구간 평균이 이동
2. Standard Deviation: 특정 구간 산포가 증가
3. Spike: 개별 포인트가 튐
4. Drift: 점진적으로 한 방향 이동
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DefectInfo:
    """주입된 불량 정보"""
    defect_type: str
    start_idx: int
    end_idx: int
    num_affected: int
    parameters: dict


class DefectSynthesizer:
    """시계열 오른쪽 끝에 불량 주입"""

    def __init__(self, config: dict, rng: np.random.Generator = None):
        self.cfg = config["defect"]
        self.rng = rng or np.random.default_rng()

    def inject(self, values: np.ndarray, mask: np.ndarray,
               defect_type: str, fleet_range: float = 0.0) -> Tuple[np.ndarray, DefectInfo]:
        self._fleet_range = fleet_range  # drift에서 사용
        anomaly_values = values.copy()
        total_length = len(values)
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            return anomaly_values, DefectInfo(defect_type, 0, 0, 0, {})

        # 불량 영역: 오른쪽 끝에서부터 (drift, spike 는 전용 범위 사용)
        if defect_type == "drift" and "region_ratio_range" in self.cfg.get("drift", {}):
            region_ratio = self.rng.uniform(*self.cfg["drift"]["region_ratio_range"])
        elif defect_type == "spike" and "region_ratio_range" in self.cfg.get("spike", {}):
            region_ratio = self.rng.uniform(*self.cfg["spike"]["region_ratio_range"])
        else:
            region_ratio = self.rng.uniform(*self.cfg["region_ratio_range"])
        defect_length = max(1, int(total_length * region_ratio))
        start_idx = total_length - defect_length
        end_idx = total_length

        affected = valid_indices[(valid_indices >= start_idx) & (valid_indices < end_idx)]

        if len(affected) == 0:
            return anomaly_values, DefectInfo(defect_type, start_idx, end_idx, 0, {})

        # 정상 영역의 std 측정 (불량 강도의 기준)
        normal_indices = valid_indices[valid_indices < start_idx]
        if len(normal_indices) > 2:
            baseline_std = np.nanstd(values[normal_indices])
        else:
            baseline_std = np.nanstd(values[valid_indices])
        baseline_std = max(baseline_std, 0.01)  # 최소값 보장

        inject_fn = {
            "mean_shift": self._inject_mean_shift,
            "standard_deviation": self._inject_standard_deviation,
            "spike": self._inject_spike,
            "drift": self._inject_drift,
        }[defect_type]

        params = inject_fn(anomaly_values, affected, start_idx, end_idx, baseline_std)

        return anomaly_values, DefectInfo(
            defect_type=defect_type,
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            num_affected=len(affected),
            parameters=params,
        )

    def _inject_mean_shift(self, values: np.ndarray, affected: np.ndarray,
                           start: int, end: int, baseline_std: float) -> dict:
        """baseline 산포는 그대로 유지, shift만 추가 (두 줄 직선 방지)."""
        cfg = self.cfg["mean_shift"]
        sigma_factor = self.rng.uniform(*cfg["shift_sigma_range"])
        shift = baseline_std * sigma_factor
        direction = self.rng.choice([-1, 1])
        shift *= direction

        values[affected] += shift

        return {"shift": float(shift), "sigma_factor": float(sigma_factor),
                "baseline_std": float(baseline_std)}

    def _inject_standard_deviation(self, values: np.ndarray, affected: np.ndarray,
                                   start: int, end: int, baseline_std: float) -> dict:
        cfg = self.cfg["standard_deviation"]
        scale = self.rng.uniform(*cfg["scale_range"])

        # 목표 산포 = baseline_std × scale
        # baseline에 추가 노이즈만 더해서 패턴 보존하면서 산포 확대
        target_std = baseline_std * scale
        extra_var = max(target_std ** 2 - baseline_std ** 2, 0)
        extra_std = np.sqrt(extra_var)

        n = len(affected)
        extra_noise = self.rng.normal(0, extra_std, n)
        values[affected] += extra_noise  # 기존 값 + 추가 노이즈 (패턴 보존)

        return {"scale": float(scale), "baseline_std": float(baseline_std),
                "target_std": float(target_std), "extra_std": float(extra_std)}

    def _inject_spike(self, values: np.ndarray, affected: np.ndarray,
                      start: int, end: int, baseline_std: float) -> dict:
        cfg = self.cfg["spike"]
        spike_ratio = self.rng.uniform(*cfg["spike_ratio_range"])
        min_spikes = cfg.get("min_spikes", 3)
        num_spikes = max(min_spikes, int(len(affected) * spike_ratio))
        num_spikes = min(num_spikes, len(affected))
        spike_indices = self.rng.choice(affected, size=num_spikes, replace=False)

        sigma_factors = self.rng.uniform(*cfg["magnitude_sigma_range"], size=num_spikes)
        magnitudes = baseline_std * sigma_factors

        # 최소 magnitude 보장
        min_mag_sigma = cfg.get("min_magnitude_sigma", 5.0)
        min_mag = baseline_std * min_mag_sigma
        magnitudes = np.maximum(magnitudes, min_mag)

        directions = self.rng.choice([-1, 1], size=num_spikes)
        values[spike_indices] += magnitudes * directions

        return {"num_spikes": int(num_spikes),
                "avg_magnitude": float(np.mean(magnitudes)),
                "baseline_std": float(baseline_std)}

    def _inject_drift(self, values: np.ndarray, affected: np.ndarray,
                      start: int, end: int, baseline_std: float) -> dict:
        """기존 baseline 값(산포/노이즈)은 그대로 유지하고 linear drift만 위에 더함.
        → fleet과 동일한 산포 수준 유지 + 점진적 이동.
        """
        cfg = self.cfg["drift"]
        sigma_factor = self.rng.uniform(*cfg["slope_sigma_range"])
        direction = self.rng.choice([-1, 1])

        n = len(affected)
        if n < 2:
            return {"slope": 0.0, "sigma_factor": float(sigma_factor),
                    "baseline_std": float(baseline_std), "max_drift": 0.0}

        # max_drift 결정
        sigma_based = baseline_std * sigma_factor
        fleet_range = getattr(self, '_fleet_range', 0.0)
        range_based = fleet_range * 1.2
        min_sigma = cfg.get("min_max_drift_sigma", 2.5)
        floor = baseline_std * min_sigma
        max_drift_abs = max(sigma_based, range_based, floor)
        max_drift = max_drift_abs * direction

        # defect 영역 [start, end) 기준 비례 위치 0~1 (dense cluster여도 영역 전체 기준)
        rel = (affected - start) / max(end - start - 1, 1)
        drift_trend = rel * max_drift

        # 기존 값에 linear trend만 더하기 (원래 노이즈/산포 유지)
        values[affected] += drift_trend

        return {"slope": float(max_drift / max(n, 1)),
                "sigma_factor": float(sigma_factor),
                "baseline_std": float(baseline_std),
                "max_drift": float(abs(max_drift)),
                "fleet_range": float(fleet_range)}
