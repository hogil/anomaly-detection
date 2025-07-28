#!/usr/bin/env python3
"""
plot_utils.py - 시각화 관련 함수들
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
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"메트릭 heatmap 저장: {save_path}")
        
    except Exception as e:
        logger.error(f"Metric heatmap 생성 실패: {e}")

def plot_confusion_matrices(model_name: str, true_series: np.ndarray, pred_series: np.ndarray,
                          true_point: np.ndarray, pred_point: np.ndarray, save_path: str) -> None:
    """모델별 confusion matrix 시각화"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Series-level confusion matrix
        cm_series = confusion_matrix(true_series, pred_series)
        sns.heatmap(cm_series, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
                    ax=axes[0])
        axes[0].set_title(f'{model_name} - Series-level Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        # Point-level confusion matrix
        cm_point = confusion_matrix(true_point, pred_point)
        sns.heatmap(cm_point, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
                    ax=axes[1])
        axes[1].set_title(f'{model_name} - Point-level Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix 저장: {save_path}")
        
    except Exception as e:
        logger.error(f"Confusion matrix 생성 실패: {e}")

def plot_single_series_result(data: np.ndarray, score: np.ndarray, threshold: float,
                            true_label: np.ndarray, pred_label: np.ndarray,
                            model_name: str, series_idx: int, category: str,
                            true_class: int, pred_class: int, true_series_label: int,
                            save_path: str) -> None:
    """단일 시계열 결과를 1x1 플롯으로 시각화 (깔끔한 버전)"""
    
    try:
        plt.figure(figsize=(16, 10))
        t = np.arange(len(data))
        
        # 메인 축 설정
        ax1 = plt.gca()
        ax1.set_facecolor('#f8f9fa')
        
        # 실제 threshold 기반 predicted labels 재계산 (더 정확하게)
        actual_pred_label = (score > threshold).astype(int)
        
        # 1. Predicted anomaly 영역 (개선된 버전)
        pred_anomaly_shown = False
        
        # threshold를 넘는 구간들을 찾아서 연속된 영역으로 표시
        anomaly_regions = []
        in_anomaly = False
        start_idx = 0
        
        for i in range(len(actual_pred_label)):
            if actual_pred_label[i] == 1 and not in_anomaly:
                # 새로운 anomaly 구간 시작
                start_idx = i
                in_anomaly = True
            elif actual_pred_label[i] == 0 and in_anomaly:
                # anomaly 구간 종료
                anomaly_regions.append((start_idx, i-1))
                in_anomaly = False
        
        # 마지막이 anomaly로 끝나는 경우
        if in_anomaly:
            anomaly_regions.append((start_idx, len(actual_pred_label)-1))
        
        # 각 anomaly 영역을 표시 (조건 완화, predicted area만 연하게)
        for start, end in anomaly_regions:
            # 2개 이상의 연속된 포인트는 영역으로 표시 (조건 완화)
            if end - start >= 1:  # 2개 이상
                left_bound = max(0, start - 0.3)
                right_bound = min(len(data) - 1, end + 0.3)
                
                if not pred_anomaly_shown:
                    ax1.axvspan(left_bound, right_bound, color='lightgreen', alpha=0.5,  # 배경영역 0.5로
                               label='Predicted Anomaly Area', zorder=1)
                    pred_anomaly_shown = True
                else:
                    ax1.axvspan(left_bound, right_bound, color='lightgreen', alpha=0.5, zorder=1)
            else:  # 단독 포인트: 세로선으로 표시
                if not pred_anomaly_shown:
                    ax1.axvline(x=start, color='lightgreen', linestyle='-', alpha=0.8, linewidth=2,
                               label='Predicted Anomaly Area')
                    pred_anomaly_shown = True
                else:
                    ax1.axvline(x=start, color='lightgreen', linestyle='-', alpha=0.8, linewidth=2)
        
        # 예측된 anomaly가 없으면 범례에 표시하지 않음
        if not pred_anomaly_shown and len(anomaly_regions) == 0:
            pass  # 아무것도 표시하지 않음
        
        # 2. Raw data를 포인트별 라벨에 따라 다른 색상으로 표시
        normal_mask = (true_label == 0)
        anomaly_mask = (true_label > 0)
        
        # Normal 포인트들 (파란색 선으로 연결)
        if np.any(normal_mask):
            normal_indices = np.where(normal_mask)[0]
            ax1.plot(normal_indices, data[normal_indices], 'b-', linewidth=2.5, 
                    alpha=0.8, label='Normal Points', zorder=3)
        
        # Anomaly 포인트들 (빨간색 마커와 선으로 연결)
        if np.any(anomaly_mask):
            anomaly_indices = np.where(anomaly_mask)[0]
            ax1.plot(anomaly_indices, data[anomaly_indices], 'ro-', linewidth=2, 
                    markersize=6, alpha=0.9, label='Anomaly Points', zorder=4)
            
            # 연속된 anomaly 구간들은 선으로 연결
            segments = []
            start = None
            for i in range(len(true_label)):
                if true_label[i] > 0 and start is None:
                    start = i
                elif true_label[i] == 0 and start is not None:
                    if i - start > 1:  # 2개 이상 연속된 경우만 선으로 연결
                        segments.append((start, i-1))
                    start = None
            if start is not None:
                if len(true_label) - start > 1:
                    segments.append((start, len(true_label)-1))
            
            # 연속 구간 선으로 연결
            for seg_start, seg_end in segments:
                ax1.plot(range(seg_start, seg_end+1), data[seg_start:seg_end+1], 
                        'r-', linewidth=3, alpha=0.7, zorder=3)
        
        # 3. Anomaly score (보조 축) - 범례보다 뒤에 표시
        ax2 = ax1.twinx()
        ax2.plot(t, score, color='orange', linewidth=2.5, alpha=0.8, 
                 label='Anomaly Score', zorder=50)  # 범례(100)보다 낮게 설정
        
        # 4. Threshold 선 - 범례보다 뒤에 표시
        ax2.axhline(threshold, color='crimson', linestyle='--', linewidth=3, 
                   alpha=0.9, label=f'Threshold ({threshold:.3f})', zorder=51)  # 범례(100)보다 낮게
        
        # 축 설정 (여백 최적화)
        ax1.set_xlim(-1, len(data))
        data_range = data.max() - data.min()
        ax1.set_ylim(data.min() - data_range * 0.08, data.max() + data_range * 0.08)  # 여백 줄임
        ax2.set_ylim(score.min() - (score.max() - score.min()) * 0.1, 
                     score.max() + (score.max() - score.min()) * 0.1)
        
        # 스타일링
        ax1.set_xlabel('Time Step', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time Series Value', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Anomaly Score', fontsize=14, fontweight='bold', color='orange')
        
        # 제목 (시리즈 라벨 정보 추가)
        series_label_names = {0: 'Normal', 1: 'Avg Change', 2: 'Std Change', 
                             3: 'Drift', 4: 'Spike', 5: 'Complex'}
        series_type = series_label_names.get(true_series_label, f'Label_{true_series_label}')
        
        pred_accuracy = f"Pred Points: {actual_pred_label.sum()}/{len(actual_pred_label)}"
        true_accuracy = f"True Points: {(true_label > 0).sum()}/{len(true_label)}"
        
        title = f'{model_name} - Series {series_idx} [{category}]\n'
        title += f'Series Type: {series_type} (Label {true_series_label})\n'
        title += f'True Class: {true_class}, Pred Class: {pred_class} | {true_accuracy}, {pred_accuracy}'
        
        ax1.set_title(title, fontsize=18, fontweight='bold', pad=40)
        
        # 그리드
        ax1.grid(True, alpha=0.3, linestyle='-', color='gray')
        ax2.grid(True, alpha=0.2, linestyle=':', color='orange')
        
        # 범례 (중복 제거 및 위치 조정 - 왼쪽 상단, 최상위)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        # 중복된 라벨 제거
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
        unique_labels = []
        unique_lines = []
        for line, label in zip(all_lines, all_labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_lines.append(line)
        
        # 범례를 왼쪽 상단으로, 최상위 zorder로 설정
        legend = ax1.legend(unique_lines, unique_labels, 
                           loc='upper left', frameon=True, framealpha=1.0,  # 완전 불투명
                           facecolor='white',  # 배경 흰색
                           fancybox=True, shadow=True, fontsize=13, 
                           bbox_to_anchor=(0.02, 0.98))  # 왼쪽 상단 모서리
        
        # 범례를 최상위로 설정 (아무것도 가리지 않게)
        legend.set_zorder(100)
        
        # 축 색상 설정
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.spines['right'].set_color('orange')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        logger.error(f"Single series plot 생성 실패: {e}")

def categorize_predictions(true_series: np.ndarray, pred_series: np.ndarray,
                         sample_labels: torch.Tensor) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """예측 결과를 TP, TN, FP, FN으로 분류 (시리즈 라벨 정보 포함)"""
    categories = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
    
    for i in range(len(true_series)):
        true_class = int(sample_labels[i].item())  # 원래 라벨 (0~5)
        pred_class = int(pred_series[i])  # 예측 라벨 (0 or 1)
        series_label = true_class  # 시리즈 타입 라벨 (0~5)
        
        if true_series[i] == 1 and pred_series[i] == 1:
            categories['TP'].append((i, true_class, pred_class, series_label))
        elif true_series[i] == 0 and pred_series[i] == 0:
            categories['TN'].append((i, true_class, pred_class, series_label))
        elif true_series[i] == 0 and pred_series[i] == 1:
            categories['FP'].append((i, true_class, pred_class, series_label))
        elif true_series[i] == 1 and pred_series[i] == 0:
            categories['FN'].append((i, true_class, pred_class, series_label))
    
    return categories