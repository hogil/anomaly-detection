# 🎨 Visualization Guide

## 📊 개요
이 문서는 Anomaly Detection 프로젝트의 시각화 시스템과 개선사항들을 정리합니다.

## 🎯 시각화 개선사항

### Before vs After
| 항목 | Before | After | 개선사항 |
|------|--------|-------|----------|
| **Predicted Area 표시** | 5개 이상 연속만 | 2개 이상 연속 | 더 세밀한 탐지 |
| **단독 포인트** | 표시 안함 | 세로선 표시 | 놓치는 이상 없음 |
| **색상 투명도** | 고정 0.3 | 조정 가능 0.5 | 가독성 향상 |
| **파일명 규칙** | 일반적 | Type 우선 표시 | 찾기 쉬움 |

## 🎨 Plot 구조

### 전체 레이아웃
```
📊 Single Series Plot (16x10 inch)
├── 🟦 상단: 시계열 데이터
│   ├── 파란색 선: Normal Points  
│   ├── 빨간색 점: Anomaly Points
│   └── 연한 초록: Predicted Anomaly Area
└── 🟠 하단: 이상 점수 (오른쪽 Y축)
    ├── 주황색 선: Anomaly Score
    └── 빨간 점선: Threshold
```

### 색상 체계
```python
COLORS = {
    'normal_points': 'b-',           # 파란색 선
    'anomaly_points': 'ro-',         # 빨간색 마커
    'predicted_area': 'lightgreen',  # 연한 초록 (alpha=0.5)
    'predicted_line': 'lightgreen',  # 연한 초록 (alpha=0.8) 
    'anomaly_score': 'orange',       # 주황색 선
    'threshold': 'crimson',          # 진홍색 점선
}
```

## 📁 파일 구조

### 결과 폴더 구조
```
results/
├── plots/                    # 시각화 결과
│   ├── CARLA/
│   │   ├── TP/              # True Positives
│   │   ├── FP/              # False Positives  
│   │   ├── FN/              # False Negatives
│   │   └── TN/              # True Negatives
│   ├── TraceGPT/
│   ├── PatchAD/
│   ├── PatchTRAD/
│   └── ProDiffAD/
├── metrics/                  # 성능 메트릭
│   ├── all_models_metrics_heatmap.png
│   └── all_models_report.json
├── confusion_matrix/         # Confusion Matrix
└── samples/                  # 데이터 샘플 예시
```

### 파일명 규칙
```
{anomaly_type}_true_{true_class}_pred_{pred_class}_series_{idx}.png

예시:
- normal_true_0_pred_0_series_123.png      # 정상을 정상으로 예측
- spike_true_4_pred_4_series_456.png       # 스파이크를 올바르게 탐지
- avg_change_true_1_pred_0_series_789.png  # 평균 변화를 놓침
```

## 🔧 주요 함수들

### 1. plot_single_series_result()
**용도**: 개별 시계열의 상세 분석 결과 시각화

```python
def plot_single_series_result(
    data: np.ndarray,           # 시계열 데이터
    score: np.ndarray,          # 이상 점수
    threshold: float,           # 임계값
    true_label: np.ndarray,     # 실제 라벨
    pred_label: np.ndarray,     # 예측 라벨
    model_name: str,            # 모델명
    series_idx: int,            # 시리즈 인덱스
    category: str,              # TP/FP/FN/TN
    true_class: int,            # 실제 클래스
    pred_class: int,            # 예측 클래스
    true_series_label: int,     # 시리즈 라벨
    save_path: str              # 저장 경로
):
    """단일 시계열 결과 시각화 (개선된 버전)"""
```

**주요 개선사항**:
- 2개 이상 연속 포인트도 영역으로 표시
- 단독 포인트는 세로선으로 표시
- 적절한 투명도로 가독성 향상

### 2. plot_metrics_heatmap()
**용도**: 모든 모델의 성능 메트릭을 히트맵으로 시각화

```python
def plot_metrics_heatmap(
    all_model_metrics: Dict[str, Dict[str, float]],
    save_path: str
):
    """모델별 성능 메트릭 히트맵 생성"""
```

### 3. categorize_predictions()
**용도**: 예측 결과를 TP/FP/FN/TN으로 분류

```python
def categorize_predictions(
    true_series: np.ndarray,
    pred_series: np.ndarray, 
    sample_labels: torch.Tensor
) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """예측 결과를 4가지 카테고리로 분류"""
```

## 🎯 이상 타입별 특성

### 5가지 이상 타입
1. **Normal (0)**: 정상 패턴
2. **Avg Change (1)**: 평균값 변화
3. **Std Change (2)**: 표준편차 변화  
4. **Drift (3)**: 점진적 변화
5. **Spike (4)**: 급격한 스파이크
6. **Complex (5)**: 복합적 패턴

### 시각화에서 구분법
```python
label_names = {
    0: 'Normal',
    1: 'Avg Change', 
    2: 'Std Change',
    3: 'Drift',
    4: 'Spike', 
    5: 'Complex'
}
```

## 📊 성능 메트릭 시각화

### 히트맵 구성
- **X축**: Metrics (Precision, Recall, F1-Score, Accuracy)
- **Y축**: Models (CARLA, TraceGPT, PatchAD, PatchTRAD, ProDiffAD)
- **색상**: 성능 점수 (0~1, 빨강→노랑→초록)

### JSON 리포트 구조
```json
{
  "CARLA": {
    "metrics": {
      "series_accuracy": 0.85,
      "series_precision": 0.78,
      "series_recall": 0.82,
      "series_f1": 0.80,
      "point_accuracy": 0.88,
      "point_f1": 0.65,
      "series_auc": 0.83
    },
    "best_threshold": 0.49,
    "best_f1": 0.80
  }
}
```

## 🔍 Predicted Anomaly Area 표시 로직

### 개선된 알고리즘
```python
# 연속된 anomaly 구간 찾기
for start, end in anomaly_regions:
    if end - start >= 1:  # 2개 이상 연속
        # 배경 영역으로 표시
        ax1.axvspan(left_bound, right_bound, 
                   color='lightgreen', alpha=0.5)
    else:  # 단독 포인트
        # 세로선으로 표시  
        ax1.axvline(x=start, color='lightgreen', 
                   alpha=0.8, linewidth=2)
```

### Before (문제점)
- 5개 이상 연속된 포인트만 영역 표시
- 단독 포인트는 표시하지 않음
- 많은 이상 포인트를 놓침

### After (개선점)
- 2개 이상 연속 포인트도 영역 표시
- 단독 포인트는 세로선으로 명확히 표시
- 모든 threshold 초과 포인트 시각화

## 🎨 커스터마이징 가이드

### 색상 변경
```python
# utils/plot_generator.py에서 수정
CUSTOM_COLORS = {
    'predicted_area_alpha': 0.5,    # 투명도 조정
    'predicted_line_alpha': 0.8,    # 세로선 투명도
    'background_color': '#f8f9fa',  # 배경색
}
```

### 폰트 크기 조정
```python
FONT_SETTINGS = {
    'title_size': 16,
    'label_size': 14, 
    'tick_size': 12,
    'legend_size': 12,
}
```

### 이미지 크기 변경
```python
plt.figure(figsize=(16, 10))  # 기본 크기
plt.savefig(save_path, dpi=150, bbox_inches='tight')
```

## 🚀 사용 예제

### 기본 사용법
```python
from utils.plot_generator import plot_single_series_result

# 단일 시계열 시각화
plot_single_series_result(
    data=time_series_data,
    score=anomaly_scores,
    threshold=0.5,
    true_label=ground_truth,
    pred_label=predictions,
    model_name="CARLA",
    series_idx=0,
    category="TP",
    true_class=1,
    pred_class=1, 
    true_series_label=2,
    save_path="results/plots/CARLA/TP/example.png"
)
```

### 배치 처리
```python
# 여러 시계열 일괄 처리
for idx, (data, score, true_label) in enumerate(dataset):
    plot_single_series_result(
        data=data,
        score=score,
        threshold=best_threshold,
        true_label=true_label,
        pred_label=(score > best_threshold),
        model_name=model_name,
        series_idx=idx,
        category=determine_category(true_label, pred_label),
        save_path=f"results/plots/{model_name}/series_{idx}.png"
    )
```

## 💡 활용 팁

### 1. 결과 분석 순서
1. **전체 히트맵** 확인 → 모델별 성능 비교
2. **TP 사례** 검토 → 잘 탐지된 패턴 분석
3. **FN 사례** 검토 → 놓친 이상 패턴 분석  
4. **FP 사례** 검토 → 오탐 원인 파악

### 2. 디버깅 가이드
- **예측 영역이 없다면**: threshold가 너무 높은지 확인
- **너무 많은 FP**: threshold가 너무 낮은지 확인
- **시각화가 깨진다면**: 데이터 shape 확인

### 3. 성능 개선 힌트  
- FN이 많으면 → threshold 조정 또는 모델 개선
- FP가 많으면 → 데이터 전처리 또는 feature engineering
- 특정 타입에서 성능이 낮으면 → 해당 타입 데이터 증강

이 가이드를 활용하면 anomaly detection 결과를 효과적으로 분석하고 개선할 수 있습니다! 🎉 