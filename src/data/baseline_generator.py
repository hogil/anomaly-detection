"""
에피소드 기반 정상 베이스라인 시계열 생성기

반도체 계측 데이터 특성:
- 주기성/계절성 없음
- 밀집/희소/결핍 영역이 랜덤으로 교차
- 에피소드 = 영역(1종) × 노이즈(부분집합, 중첩 가능)
- 이미지 1장 = 에피소드 K개 이어붙임 (K도 랜덤)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Set


@dataclass
class Episode:
    """에피소드 1개의 정보"""
    region_type: str                # "dense", "sparse", "missing"
    noise_types: Set[str]           # {"gaussian", "laplacian", "correlated"} 부분집합
    start_idx: int                  # 전체 시계열 내 시작 인덱스
    end_idx: int                    # 전체 시계열 내 끝 인덱스
    length: int                     # 포인트 수
    density: float                  # 실제 데이터 비율
    noise_params: dict              # 각 노이즈의 실제 파라미터


class BaselineGenerator:
    """에피소드 기반 정상 시계열 생성"""

    def __init__(self, config: dict, rng: np.random.Generator = None):
        self.cfg = config
        self.episode_cfg = config["episode"]
        self.baseline_cfg = config["baseline"]
        self.rng = rng or np.random.default_rng()

    def generate(self) -> Tuple[np.ndarray, np.ndarray, List[Episode]]:
        """
        정상 시계열 1개 생성

        Returns:
            values: 전체 시계열 값 (NaN 포함)
            mask: 유효 데이터 마스크
            episodes: 에피소드 정보 리스트
        """
        # 0) Chart 전체 단일 노이즈 type (영역별 산포 차이 절대 금지)
        # gaussian 80%, laplacian 15%, correlated 5% — 단일 type만 사용
        ntype = self.rng.choice(
            ["gaussian", "laplacian", "correlated"],
            p=[0.80, 0.15, 0.05],
        )
        self._chart_noise_types = {str(ntype)}
        self._chart_base_noise = self._sample_noise_params(self._chart_noise_types)

        # 1) 에피소드 수 결정
        num_episodes = self.rng.integers(
            self.episode_cfg["count_range"][0],
            self.episode_cfg["count_range"][1] + 1
        )

        # 결핍 영역 개수 제한
        missing_cfg = self.episode_cfg.get("missing", {})
        max_missing = missing_cfg.get("max_count", num_episodes)
        missing_count = 0

        # 2) 각 에피소드 생성
        episodes = []
        current_idx = 0

        for ep_i in range(num_episodes):
            forbid_missing = (missing_count >= max_missing)
            # 마지막 1개 에피소드만 missing 금지 (defect 영역 보호)
            forbid_missing_last = forbid_missing or (ep_i == num_episodes - 1)
            ep = self._create_episode(current_idx, forbid_missing_last, force_dense=False)
            if ep.region_type == "missing":
                missing_count += 1
            episodes.append(ep)
            current_idx = ep.end_idx

        total_length = current_idx

        # 3) 베이스 신호 생성 (랜덤 워크, 주기성 없음)
        base_signal = self._generate_random_walk(total_length)

        # 4) 에피소드별 영역 밀도 + 노이즈 적용
        values = np.full(total_length, np.nan)
        mask = np.zeros(total_length, dtype=bool)

        for ep in episodes:
            if ep.region_type == "missing":
                continue  # NaN 유지

            # 영역 내 유효 인덱스 (밀도에 따라)
            ep_indices = np.arange(ep.start_idx, ep.end_idx)
            num_valid = max(1, int(ep.length * ep.density))
            valid_indices = self.rng.choice(ep_indices, size=num_valid, replace=False)
            valid_indices.sort()

            mask[valid_indices] = True
            values[valid_indices] = base_signal[valid_indices]

            # 노이즈 중첩 적용
            noise = self._generate_combined_noise(len(valid_indices), ep.noise_types, ep.noise_params)
            values[valid_indices] += noise

        return values, mask, episodes

    def generate_member_mask(self, shared_mask: np.ndarray, episodes: List[Episode]) -> np.ndarray:
        """공유 마스크 기반 멤버별 밀도 변동 적용

        전체 결핍(missing) 영역은 유지 (모든 멤버 공유).
        비결핍 에피소드 내에서 멤버별 dropout/thin/densify 패치 적용.
        """
        member_cfg = self.episode_cfg.get("member_variation", {})
        if not member_cfg:
            return shared_mask.copy()

        member_mask = shared_mask.copy()

        count_range = member_cfg["event_count_range"]
        event_count = self.rng.integers(count_range[0], count_range[1] + 1)
        if event_count == 0:
            return member_mask

        non_missing = [ep for ep in episodes if ep.region_type != "missing"]
        if not non_missing:
            return member_mask

        length_range = member_cfg["event_length_range"]
        event_weights = member_cfg.get("event_weights",
                                       {"dropout": 0.30, "thin": 0.40, "densify": 0.30})
        ev_types = list(event_weights.keys())
        ev_w = np.array(list(event_weights.values()))
        ev_w /= ev_w.sum()

        for _ in range(event_count):
            ep = non_missing[self.rng.integers(0, len(non_missing))]

            event_len = self.rng.integers(length_range[0], length_range[1] + 1)
            event_len = min(event_len, ep.length)

            max_start = ep.end_idx - event_len
            if max_start <= ep.start_idx:
                start = ep.start_idx
            else:
                start = self.rng.integers(ep.start_idx, max_start + 1)
            end = min(start + event_len, ep.end_idx)

            event_type = self.rng.choice(ev_types, p=ev_w)

            if event_type == "dropout":
                member_mask[start:end] = False

            elif event_type == "thin":
                valid_in = np.where(member_mask[start:end])[0] + start
                if len(valid_in) > 1:
                    drop_ratio = self.rng.uniform(0.3, 0.7)
                    n_drop = max(1, int(len(valid_in) * drop_ratio))
                    drop_idx = self.rng.choice(valid_in, size=n_drop, replace=False)
                    member_mask[drop_idx] = False

            else:  # densify
                invalid_in = np.where(~member_mask[start:end])[0] + start
                if len(invalid_in) > 0:
                    add_ratio = self.rng.uniform(0.3, 0.7)
                    n_add = max(1, int(len(invalid_in) * add_ratio))
                    n_add = min(n_add, len(invalid_in))
                    add_idx = self.rng.choice(invalid_in, size=n_add, replace=False)
                    member_mask[add_idx] = True

        return member_mask

    def generate_on_shared_mask(self, mask: np.ndarray, episodes: List[Episode]) -> np.ndarray:
        """공유 mask 위에 새 값 생성 (시간 구조는 동일, 값만 독립).
        chart 전체 단일 노이즈 파라미터 사용 (영역별 차이 절대 금지)."""
        total_length = len(mask)
        base_signal = self._generate_random_walk(total_length)
        values = np.full(total_length, np.nan)

        # chart 단일 노이즈 (멤버별로 동일 type, 동일 파라미터)
        chart_types = getattr(self, '_chart_noise_types', None) or set()
        chart_params = getattr(self, '_chart_base_noise', None) or {}

        for ep in episodes:
            if ep.region_type == "missing":
                continue
            ep_valid = np.where(mask[ep.start_idx:ep.end_idx])[0] + ep.start_idx
            if len(ep_valid) == 0:
                continue

            values[ep_valid] = base_signal[ep_valid]

            if chart_types:
                noise = self._generate_combined_noise(len(ep_valid), chart_types, chart_params)
                values[ep_valid] += noise

        return values

    def _create_episode(self, start_idx: int, forbid_missing: bool = False, force_dense: bool = False) -> Episode:
        """에피소드 1개 사양 결정"""
        if force_dense:
            # 마지막 에피소드는 dense 강제
            region_weights = {"dense": 1.0}
        else:
            region_weights = self.episode_cfg.get("region_weights", {"dense": 0.55, "sparse": 0.30, "missing": 0.15})
        types = list(region_weights.keys())
        weights = np.array(list(region_weights.values()))

        if forbid_missing and "missing" in types:
            idx = types.index("missing")
            types.pop(idx)
            weights = np.delete(weights, idx)

        weights /= weights.sum()
        region_type = self.rng.choice(types, p=weights)

        # 길이: 결핍은 별도 range (너비 다양성)
        if region_type == "missing":
            missing_cfg = self.episode_cfg.get("missing", {})
            length_range = missing_cfg.get("length_range", self.episode_cfg["length_range"])
        else:
            length_range = self.episode_cfg["length_range"]

        length = self.rng.integers(length_range[0], length_range[1] + 1)

        # 밀도 (config에 정의된 모든 region type 지원)
        region_cfg = self.episode_cfg["region"]
        if region_type == "missing":
            density = 0.0
        elif region_type in region_cfg and "density_range" in region_cfg[region_type]:
            density = self.rng.uniform(*region_cfg[region_type]["density_range"])
        else:
            density = self.rng.uniform(0.30, 0.55)  # fallback

        # 노이즈: chart 전체 단일 set 사용 (영역별 차이 절대 금지)
        noise_types = set()
        noise_params = {}
        if region_type != "missing":
            chart_types = getattr(self, '_chart_noise_types', None)
            if chart_types:
                noise_types = set(chart_types)
                noise_params = self._sample_noise_params(noise_types)
            else:
                available = ["gaussian", "laplacian", "correlated"]
                num_noises = self.rng.integers(1, len(available) + 1)
                chosen = self.rng.choice(available, size=num_noises, replace=False)
                noise_types = set(chosen)
                noise_params = self._sample_noise_params(noise_types)

        return Episode(
            region_type=region_type,
            noise_types=noise_types,
            start_idx=start_idx,
            end_idx=start_idx + length,
            length=length,
            density=density,
            noise_params=noise_params,
        )

    def _sample_noise_params(self, noise_types: Set[str]) -> dict:
        """노이즈 파라미터를 range 내에서 랜덤 샘플링.
        chart base noise가 있으면 그대로 반환 (영역별 차이 절대 금지)."""
        noise_cfg = self.episode_cfg["noise"]
        params = {}
        base = getattr(self, '_chart_base_noise', None)

        if "gaussian" in noise_types:
            if base and "gaussian_sigma" in base:
                params["gaussian_sigma"] = base["gaussian_sigma"]
            else:
                params["gaussian_sigma"] = self.rng.uniform(*noise_cfg["gaussian"]["sigma_range"])

        if "laplacian" in noise_types:
            if base and "laplacian_b" in base:
                params["laplacian_b"] = base["laplacian_b"]
            else:
                params["laplacian_b"] = self.rng.uniform(*noise_cfg["laplacian"]["b_range"])

        if "correlated" in noise_types:
            if base and "correlated_rho" in base:
                params["correlated_rho"] = base["correlated_rho"]
            else:
                params["correlated_rho"] = self.rng.uniform(*noise_cfg["correlated"]["rho_range"])
            if base and "correlated_sigma" in base:
                params["correlated_sigma"] = base["correlated_sigma"]
            else:
                params["correlated_sigma"] = self.rng.uniform(*noise_cfg["correlated"]["sigma_range"])

        return params

    def _generate_random_walk(self, length: int) -> np.ndarray:
        """거의 평탄한 baseline. 출렁임 금지.
        매우 작은 step + 강한 평균회귀로 OU 정상상태 std가 매우 작음.
        실제 변동은 노이즈(gaussian/laplacian/correlated)에서만 나옴.
        """
        vmin, vmax = self.baseline_cfg["value_range"]
        step_range = self.baseline_cfg["random_walk_step"]
        mr_range = self.baseline_cfg.get("mean_revert_range", [0.20, 0.40])

        step_size = self.rng.uniform(*step_range)
        mean_revert = self.rng.uniform(*mr_range)

        # 시작점 (chart 전체의 평균 위치)
        offset = self.rng.uniform(vmin * 0.3, vmax * 0.3)
        signal = np.full(length, offset, dtype=float)

        for t in range(1, length):
            # OU process: 강한 평균회귀로 offset에서 거의 벗어나지 않음
            revert = -mean_revert * (signal[t - 1] - offset)
            noise = self.rng.normal(0, step_size)
            signal[t] = signal[t - 1] + revert + noise

        return signal

    def _generate_combined_noise(self, length: int, noise_types: Set[str],
                                 params: dict) -> np.ndarray:
        """선택된 노이즈들을 중첩 생성"""
        combined = np.zeros(length)

        if "gaussian" in noise_types:
            sigma = params["gaussian_sigma"]
            combined += self.rng.normal(0, sigma, length)

        if "laplacian" in noise_types:
            b = params["laplacian_b"]
            combined += self.rng.laplace(0, b, length)

        if "correlated" in noise_types:
            rho = params["correlated_rho"]
            sigma = params["correlated_sigma"]
            combined += self._ar1_noise(length, rho, sigma)

        return combined

    def _ar1_noise(self, length: int, rho: float, sigma: float) -> np.ndarray:
        """AR(1) 상관 노이즈: x_t = ρ·x_{t-1} + (1-ρ)·ε"""
        eps = self.rng.normal(0, sigma, length)
        noise = np.zeros(length)
        noise[0] = eps[0]
        for t in range(1, length):
            noise[t] = rho * noise[t - 1] + (1 - rho) * eps[t]
        return noise
