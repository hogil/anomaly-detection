"""
이미지 렌더러 (Overlay 통일)

모든 클래스가 overlay 포맷:
- 학습용: target=하이라이트색 + fleet=회색, 축 없음 (224x224)
- display: target=빨강(불량)/파랑(정상) + fleet=연한 고유색, 축/legend

fleet_data 포맷: {member_id: (x_vals, y_vals)}
  - x_vals: numpy array of int/float/datetime (연속 x 좌표)
  - y_vals: numpy array of measurement values
  - matplotlib이 x dtype 자동 처리 (int/float/datetime axis 자동)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

COLOR_NORMAL = "#4878CF"
COLOR_ANOMALY = "#D62728"

DISPLAY_FIG_SIZE = (7.0, 4.5)   # legend 우측 배치 후 비율 보정
DISPLAY_DPI = 150
DISPLAY_MARKER_FLEET = 18
DISPLAY_MARKER_TARGET = 25

LIGHT_COLORS = [
    "#7EB8DA", "#A8D8A8", "#D4A8D4", "#F0C987",
    "#A8D4D4", "#D4C8A8", "#C8A8B8", "#B8C8D4",
]


def _is_valid_defect_start(ds) -> bool:
    """defect_start가 유효한 x 좌표인지 확인 (None/NaN/음수면 False)"""
    if ds is None:
        return False
    if isinstance(ds, (int, float)):
        try:
            if np.isnan(ds):
                return False
        except (TypeError, ValueError):
            pass
        return ds > 0
    # datetime, np.datetime64, pd.Timestamp 등은 그 자체로 유효
    return True


def _is_datetime_x(x_val) -> bool:
    """x 값이 datetime 계열인지 확인"""
    if x_val is None:
        return False
    # numpy datetime64
    if hasattr(x_val, 'dtype'):
        return np.issubdtype(x_val.dtype, np.datetime64)
    # pd.Timestamp, datetime.datetime
    if hasattr(x_val, 'year') and hasattr(x_val, 'month'):
        return True
    return False


def _filter_outliers(fleet_data, sigma=5):
    """전체 fleet의 y값 mean ± sigma*std 밖의 점 제거 (시각화 정리용)"""
    if sigma <= 0 or not fleet_data:
        return fleet_data
    all_y = []
    for _, (x, y) in fleet_data.items():
        if len(y) > 0:
            all_y.append(np.asarray(y, dtype=float))
    if not all_y:
        return fleet_data
    concat = np.concatenate(all_y)
    finite = concat[np.isfinite(concat)]
    if len(finite) == 0:
        return fleet_data
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    if std <= 0:
        return fleet_data
    lower = mean - sigma * std
    upper = mean + sigma * std
    filtered = {}
    for mid, (x, y) in fleet_data.items():
        if len(y) == 0:
            filtered[mid] = (x, y)
            continue
        y_arr = np.asarray(y, dtype=float)
        mask = (y_arr >= lower) & (y_arr <= upper)
        filtered[mid] = (np.asarray(x)[mask], y_arr[mask])
    return filtered


def _x_span(x_min, x_max):
    """x_min~x_max의 spread 길이 (numeric or timedelta) 또는 None"""
    try:
        span = x_max - x_min
        return span
    except (TypeError, ValueError):
        return None


def _add_x_margin(ax, x_min, x_max, margin_frac=0.02):
    """x축 margin 추가 (numeric/datetime 호환). 단일 점이면 그대로."""
    span = _x_span(x_min, x_max)
    if span is None:
        ax.set_xlim(x_min, x_max)
        return
    try:
        # numeric: span > 0, datetime: timedelta > 0
        if hasattr(span, "total_seconds"):
            zero = span * 0
            if span <= zero:
                ax.set_xlim(x_min, x_max)
                return
        elif span <= 0:
            ax.set_xlim(x_min, x_max)
            return
        margin = span * margin_frac
        ax.set_xlim(x_min - margin, x_max + margin)
    except (TypeError, ValueError):
        ax.set_xlim(x_min, x_max)


class ImageRenderer:

    def __init__(self, config: dict):
        img_cfg = config["image"]
        self.width = img_cfg["width"]
        self.height = img_cfg["height"]
        self.dpi = img_cfg["dpi"]
        self.background = img_cfg["background"]

        overlay = img_cfg["overlay"]
        self.fleet_color = overlay["fleet_color"]
        self.fleet_alpha = overlay["fleet_alpha"]
        self.fleet_marker = overlay["fleet_marker_size"]
        self.target_color = overlay["target_color"]
        self.target_alpha = overlay["target_alpha"]
        self.target_marker = overlay["target_marker_size"]

        # outlier filter sigma (0 = disabled, default 5)
        self.outlier_sigma = float(img_cfg.get("outlier_sigma", 5))

    # ================================================================
    # 학습용 (target=하이라이트 + fleet=회색, 축 없음)
    # ================================================================

    def render_overlay(self, fleet_data: dict, target_id: str, save_path: str):
        # outlier filtering (mean ± N*std 밖의 점 제거)
        fleet_data = _filter_outliers(fleet_data, sigma=self.outlier_sigma)
        fig, ax = self._create_figure()
        vmin, vmax = self._fleet_minmax(fleet_data)
        val_range = vmax - vmin if vmax - vmin > 1e-10 else 1.0

        # Fleet (회색)
        for mid, (x_vals, y_vals) in fleet_data.items():
            if mid == target_id:
                continue
            if len(x_vals) > 0:
                y_norm = (y_vals - vmin) / val_range
                ax.scatter(x_vals, y_norm,
                           s=self.fleet_marker, c=self.fleet_color,
                           alpha=self.fleet_alpha, edgecolors="none", zorder=1)

        # Target (하이라이트)
        if target_id in fleet_data:
            x_vals, y_vals = fleet_data[target_id]
            if len(x_vals) > 0:
                y_norm = (y_vals - vmin) / val_range
                ax.scatter(x_vals, y_norm,
                           s=self.target_marker, c=self.target_color,
                           alpha=self.target_alpha, edgecolors="none", zorder=2)

        ax.set_ylim(-0.1, 1.1)
        x_min, x_max = self._x_range(fleet_data)
        if x_min is not None and x_max is not None:
            _add_x_margin(ax, x_min, x_max, margin_frac=0.03)
        self._finalize_train(fig, ax, save_path)

    # ================================================================
    # Display (전체 멤버 색상 구분)
    # ================================================================

    def render_overlay_display(self, fleet_data: dict, target_id: str,
                               save_path: str, anomalous_ids: list = None,
                               defect_start_idx=None, title: str = None,
                               x_label: str = "Sample Index",
                               y_label: str = "Measurement Value (nm)"):
        if anomalous_ids is None:
            anomalous_ids = []

        # outlier filtering (mean ± N*std 밖의 점 제거)
        fleet_data = _filter_outliers(fleet_data, sigma=self.outlier_sigma)

        fig, ax = self._create_figure_display()

        # 정상 멤버 (연한 고유색)
        color_idx = 0
        for mid, (x_vals, y_vals) in fleet_data.items():
            if mid in anomalous_ids:
                continue
            if len(x_vals) > 0:
                color = LIGHT_COLORS[color_idx % len(LIGHT_COLORS)]
                ax.scatter(x_vals, y_vals,
                           s=DISPLAY_MARKER_FLEET, c=color,
                           alpha=0.40, edgecolors="none",
                           zorder=1, label=mid)
                color_idx += 1

        # 불량 멤버 (진한 빨강) 또는 정상 target (파랑)
        if target_id in fleet_data:
            x_vals, y_vals = fleet_data[target_id]
            if len(x_vals) > 0:
                if target_id in anomalous_ids:
                    # 불량: 빨강
                    if _is_valid_defect_start(defect_start_idx):
                        # 개별 불량: 경계선 전후 색 분리
                        try:
                            normal_m = x_vals < defect_start_idx
                            anomaly_m = x_vals >= defect_start_idx
                        except TypeError:
                            # x_vals와 defect_start_idx 타입 불일치 시 fallback
                            normal_m = np.zeros(len(x_vals), dtype=bool)
                            anomaly_m = np.ones(len(x_vals), dtype=bool)
                        if normal_m.any():
                            ax.scatter(x_vals[normal_m], y_vals[normal_m],
                                       s=DISPLAY_MARKER_TARGET, c=COLOR_NORMAL,
                                       alpha=0.65, edgecolors="white", linewidths=0.3,
                                       zorder=2, label=f"{target_id} (normal)")
                        if anomaly_m.any():
                            ax.scatter(x_vals[anomaly_m], y_vals[anomaly_m],
                                       s=DISPLAY_MARKER_TARGET, c=COLOR_ANOMALY,
                                       alpha=0.75, edgecolors="white", linewidths=0.3,
                                       zorder=3, label=f"* {target_id} (anomaly)")
                        try:
                            ax.axvline(x=defect_start_idx, color="#CCCCCC",
                                       linestyle="--", linewidth=0.8, zorder=1)
                        except (TypeError, ValueError):
                            pass
                    else:
                        # Context 불량 또는 defect 시점 없음: 전체 빨강
                        ax.scatter(x_vals, y_vals,
                                   s=DISPLAY_MARKER_TARGET, c=COLOR_ANOMALY,
                                   alpha=0.75, edgecolors="white", linewidths=0.3,
                                   zorder=2, label=f"* {target_id}")
                else:
                    # 정상: 파랑
                    ax.scatter(x_vals, y_vals,
                               s=DISPLAY_MARKER_TARGET, c=COLOR_NORMAL,
                               alpha=0.65, edgecolors="white", linewidths=0.3,
                               zorder=2, label=target_id)

        # 축 범위
        x_min, x_max = self._x_range(fleet_data)
        if x_min is not None and x_max is not None:
            _add_x_margin(ax, x_min, x_max, margin_frac=0.04)

        ymin, ymax = self._fleet_minmax(fleet_data)
        if ymax > ymin:
            margin = (ymax - ymin) * 0.15
            ax.set_ylim(ymin - margin, ymax + margin)

        # x축이 datetime이면 yy/mm/dd 포맷 적용
        try:
            if x_min is not None and _is_datetime_x(x_min):
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
                fig.autofmt_xdate(rotation=30, ha="right")
        except Exception:
            pass

        self._style_display(ax, title=title, x_label=x_label, y_label=y_label)
        self._finalize_display(fig, ax, save_path)

    # ================================================================
    # 내부 helpers
    # ================================================================

    @staticmethod
    def _fleet_minmax(fleet_data):
        """모든 멤버의 y값 (min, max)"""
        all_vals = []
        for _, (x_vals, y_vals) in fleet_data.items():
            if len(y_vals) > 0:
                all_vals.append(y_vals)
        if not all_vals:
            return (0.0, 1.0)
        concat = np.concatenate(all_vals)
        return (float(np.nanmin(concat)), float(np.nanmax(concat)))

    @staticmethod
    def _x_range(fleet_data):
        """모든 멤버의 x값 (min, max). x가 datetime/object이면 그대로 반환."""
        x_min = None
        x_max = None
        for _, (x_vals, _) in fleet_data.items():
            if len(x_vals) == 0:
                continue
            try:
                cur_min = x_vals.min()
                cur_max = x_vals.max()
            except (TypeError, ValueError):
                continue
            if x_min is None or cur_min < x_min:
                x_min = cur_min
            if x_max is None or cur_max > x_max:
                x_max = cur_max
        return x_min, x_max

    def _create_figure(self):
        fig_w = self.width / self.dpi
        fig_h = self.height / self.dpi
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=self.dpi)
        fig.patch.set_facecolor(self.background)
        ax.set_facecolor(self.background)
        return fig, ax

    def _create_figure_display(self):
        fig, ax = plt.subplots(1, 1, figsize=DISPLAY_FIG_SIZE, dpi=DISPLAY_DPI)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")
        return fig, ax

    @staticmethod
    def _style_display(ax, title: str = None,
                       x_label: str = "Sample Index",
                       y_label: str = "Measurement Value (nm)"):
        if title:
            ax.set_title(title, fontsize=10, color="#333333", fontweight="bold", pad=8)
        ax.set_xlabel(x_label, fontsize=9, color="#444444")
        ax.set_ylabel(y_label, fontsize=9, color="#444444")
        ax.tick_params(labelsize=8, colors="#666666")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.grid(True, alpha=0.15, linewidth=0.5)
        # legend를 plot 바깥 (오른쪽)에 배치 → 데이터 가림 방지
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.85, edgecolor="#DDDDDD", fancybox=True,
                  borderaxespad=0)

    def _finalize_train(self, fig, ax, save_path):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, pad_inches=0, facecolor=self.background)
        plt.close(fig)

    @staticmethod
    def _finalize_display(fig, ax, save_path):
        plt.tight_layout(pad=0.8)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DISPLAY_DPI, bbox_inches="tight",
                    facecolor="#FAFAFA", edgecolor="none")
        plt.close(fig)
