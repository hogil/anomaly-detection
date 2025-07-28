#!/usr/bin/env python3
"""
plot_generator.py - 개선된 시각화 함수들
"""
import os
import logging
from typing import Dict, List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

def plot_metrics_heatmap(all_model_metrics: Dict[str, Dict[str, float]], save_path: str) -> None:
    """모델별 메트릭을 heatmap으로 시각화"""
    try:
        models = list(all_model_metrics.keys())
        
        # Series metrics
        series_metrics = ['accuracy', 'precision', 'recall', 'f1']
        series_data = []
        
        for model in models:
            row = [all_model_metrics[model][f'series_{metric}'] for metric in series_metrics]
            series_data.append(row)
        
        # Point metrics
        point_data = []
        for model in models:
            row = [all_model_metrics[model][f'point_{metric}'] for metric in series_metrics]
            point_data.append(row)
        
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Series metrics heatmap
        sns.heatmap(series_data, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=series_metrics, yticklabels=models, ax=axes[0])
        axes[0].set_title('Series-level Metrics', fontsize=14)
        axes[0].set_xlabel('Metrics')
        axes[0].set_ylabel('Models')
        
        # Point metrics heatmap
        sns.heatmap(point_data, annot=True, fmt='.3f', cmap='Reds',
                    xticklabels=series_metrics, yticklabels=models, ax=axes[1])
        axes[1].set_title('Point-level Metrics', fontsize=14)
        axes[1].set_xlabel('Metrics')
        axes[1].set_ylabel('Models')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"메트릭 히트맵이 {save_path}에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"메트릭 히트맵 생성 실패: {e}")

def plot_confusion_matrices(model_name: str, true_series: np.ndarray, pred_series: np.ndarray,
                          true_point: np.ndarray, pred_point: np.ndarray, save_path: str) -> None:
    """Series-level과 Point-level confusion matrix를 함께 시각화"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Series-level confusion matrix
        cm_series = confusion_matrix(true_series, pred_series)
        sns.heatmap(cm_series, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        axes[0].set_title(f'{model_name} - Series-level Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Point-level confusion matrix
        cm_point = confusion_matrix(true_point, pred_point)
        sns.heatmap(cm_point, annot=True, fmt='d', cmap='Reds', ax=axes[1],
                    xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        axes[1].set_title(f'{model_name} - Point-level Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix가 {save_path}에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"Confusion matrix 생성 실패: {e}")

def plot_single_series_result(data: np.ndarray, score: np.ndarray, threshold: float,
                            true_label: np.ndarray, pred_label: np.ndarray, 
                            model_name: str, series_idx: int, category: str,
                            true_class: int, pred_class: int, true_series_label: int,
                            save_path: str) -> None:
    """개선된 단일 시계열 결과 시각화"""
    try:
        fig, ax1 = plt.subplots(figsize=(16, 10))
        
        # 시간 축
        t = range(len(data))
        
        # 1. 실제 이상 구간을 정확히 계산
        true_anomaly_points = (true_label > 0)
        
        # 2. 예측 이상 구간을 정확히 계산  
        # pred_label이 point-wise 예측이므로 이를 사용
        if len(pred_label.shape) > 1:
            pred_anomaly_points = (pred_label > 0).any(axis=1) if pred_label.shape[0] == 1 else (pred_label > 0)
        else:
            pred_anomaly_points = (pred_label > 0)
        
        # 3. 예측 이상 구간 배경 (연속 구간만 색칠)
        pred_segments = []
        start = None
        for i, is_anomaly in enumerate(pred_anomaly_points):
            if is_anomaly and start is None:
                start = i
            elif not is_anomaly and start is not None:
                pred_segments.append((start, i-1))
                start = None
        if start is not None:
            pred_segments.append((start, len(pred_anomaly_points)-1))
        
        # 예측 이상 구간을 녹색 배경으로 표시 (더 연한 색상)
        for start, end in pred_segments:
            ax1.axvspan(start, end, alpha=0.2, color='lightgreen', 
                       label='Predicted Anomaly Area' if start == pred_segments[0][0] else "")
        
        # 4. 기본 시계열 데이터 (파란색)
        ax1.plot(t, data, 'b-', linewidth=2, alpha=0.8, label='Normal Points', zorder=10)
        
        # 5. 실제 이상 포인트를 빨간 점으로 강조
        true_anomaly_indices = np.where(true_anomaly_points)[0]
        if len(true_anomaly_indices) > 0:
            ax1.scatter(true_anomaly_indices, data[true_anomaly_indices], 
                       color='red', s=80, alpha=0.9, label='Anomaly Points', zorder=20)
        
        # 6. 이상 점수 (보조 축)
        ax2 = ax1.twinx()
        ax2.plot(t, score, color='orange', linewidth=2.5, alpha=0.8, 
                 label='Anomaly Score', zorder=30)
        
        # 7. 임계값 선 (더 진한 색상)
        ax2.axhline(threshold, color='crimson', linestyle='--', linewidth=3, 
                   alpha=0.9, label=f'Threshold ({threshold:.3f})', zorder=40)
        
        # 8. 축 설정
        ax1.set_xlim(-1, len(data))
        data_range = data.max() - data.min()
        ax1.set_ylim(data.min() - data_range * 0.05, data.max() + data_range * 0.05)
        
        score_range = score.max() - score.min()
        ax2.set_ylim(score.min() - score_range * 0.1, score.max() + score_range * 0.1)
        
        # 9. 라벨링
        ax1.set_xlabel('Time Step', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time Series Value', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Anomaly Score', fontsize=14, fontweight='bold', color='orange')
        
        # 10. 제목 (더 명확한 정보)
        series_label_names = {0: 'Normal', 1: 'Avg Change', 2: 'Std Change', 
                             3: 'Drift', 4: 'Spike', 5: 'Complex'}
        series_type = series_label_names.get(true_series_label, f'Label_{true_series_label}')
        
        # 정확한 포인트 개수 계산
        true_anomaly_count = np.sum(true_anomaly_points)
        pred_anomaly_count = np.sum(pred_anomaly_points)
        
        title = f'{model_name} - Series {series_idx} [{category}]\\n'
        title += f'Series Type: {series_type} (Label {true_series_label})\\n'
        title += f'True Class: {true_class}, Pred Class: {pred_class} | '
        title += f'True Points: {true_anomaly_count}/{len(data)}, Pred Points: {pred_anomaly_count}/{len(data)}'
        
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 11. 그리드
        ax1.grid(True, alpha=0.3, linestyle='-', color='gray')
        ax2.grid(True, alpha=0.2, linestyle=':', color='orange')
        
        # 12. 범례 (중복 제거 및 위치 최적화)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
        
        # 중복 제거
        unique_lines = []
        unique_labels = []
        for line, label in zip(all_lines, all_labels):
            if label not in unique_labels:
                unique_lines.append(line)
                unique_labels.append(label)
        
        # 범례를 오른쪽 상단에 배치
        legend = ax1.legend(unique_lines, unique_labels, 
                           loc='upper right', frameon=True, framealpha=0.95,
                           facecolor='white', edgecolor='gray',
                           fontsize=12, bbox_to_anchor=(0.98, 0.98))
        legend.set_zorder(100)
        
        # 13. 축 색상 설정
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.spines['right'].set_color('orange')
        
        # 14. 디버그 정보 (로그에 출력)
        logger.debug(f"Series {series_idx}: True anomalies: {true_anomaly_count}, "
                    f"Pred anomalies: {pred_anomaly_count}, Threshold: {threshold:.3f}")
        
        plt.tight_layout()
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        logger.error(f"Single series plot 생성 실패 (series {series_idx}): {e}")
        plt.close()

def categorize_predictions(true_series: np.ndarray, pred_series: np.ndarray,
                         sample_labels: torch.Tensor) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """예측 결과를 TP, TN, FP, FN으로 분류"""
    categories = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
    
    for i in range(len(true_series)):
        true_class = int(true_series[i])  # 이진 분류 결과 (0 or 1)
        pred_class = int(pred_series[i])  # 예측 결과 (0 or 1)
        series_label = int(sample_labels[i].item())  # 원래 라벨 (0~5)
        
        if true_class == 1 and pred_class == 1:
            categories['TP'].append((i, true_class, pred_class, series_label))
        elif true_class == 0 and pred_class == 0:
            categories['TN'].append((i, true_class, pred_class, series_label))
        elif true_class == 0 and pred_class == 1:
            categories['FP'].append((i, true_class, pred_class, series_label))
        elif true_class == 1 and pred_class == 0:
            categories['FN'].append((i, true_class, pred_class, series_label))
    
    return categories
