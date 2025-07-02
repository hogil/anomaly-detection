#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Complete Anomaly Detection System
- Real models from models folder
- Enhanced difficulty data generation
- Model-specific folders with detailed plots
- Point/Series level metrics and confusion matrices
- Performance optimizations
- Auto GPU/DDP detection
"""

import os
import sys
import argparse
import time
import math
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler

# Visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    print("✅ matplotlib available")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("❌ matplotlib not available")

# Metrics
try:
    from sklearn.metrics import (confusion_matrix, roc_auc_score, accuracy_score, 
                                precision_score, recall_score, f1_score, roc_curve)
    SKLEARN_AVAILABLE = True
    print("✅ sklearn available")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("❌ sklearn not available")

warnings.filterwarnings('ignore')

# ============================================================================
# 🔧 GLOBAL CONFIGURATION
# ============================================================================

# GPU Environment Detection
def detect_gpu_environment():
    """GPU 환경 자동 감지 및 설정"""
    if not torch.cuda.is_available():
        return {'device': 'cpu', 'world_size': 1, 'use_ddp': False}
    
    gpu_count = torch.cuda.device_count()
    print(f"🔧 GPU 감지: {gpu_count}개 GPU 발견")
    
    if gpu_count == 1:
        return {
            'device': 'cuda:0',
            'world_size': 1,
            'use_ddp': False,
            'gpu_count': gpu_count
        }
    else:
        return {
            'device': 'cuda',
            'world_size': gpu_count,
            'use_ddp': True,
            'gpu_count': gpu_count
        }

GPU_CONFIG = detect_gpu_environment()
print(f"🚀 GPU Configuration: {GPU_CONFIG}")

# Configuration
CONFIG = {
    'DATA_SIZE': 1000,
    'SEQ_LEN': 64,
    'BATCH_SIZE': 16 if not GPU_CONFIG['use_ddp'] else 8,  # Adjust for DDP
    'EPOCHS': 5,
    'LEARNING_RATE': 1e-4,
    'THRESHOLD': 0.5,
    'SEED': 42,
    'WORKERS': 4
}

# Create directories
DIRS = ['samples', 'plots', 'metrics', 'confusion_matrices', 'pre_trained']
for base_dir in DIRS:
    os.makedirs(base_dir, exist_ok=True)
    
# Model-specific plot directories
MODEL_NAMES = ['carla', 'tracegpt', 'patchtrad', 'prodiffad', 
               'patch_trace_ensemble', 'transfer_learning_ensemble', 'multi_model_ensemble']

for model_name in MODEL_NAMES:
    for subdir in ['true_positive', 'true_negative', 'false_positive', 'false_negative']:
        os.makedirs(f'plots/{model_name}/{subdir}', exist_ok=True)

print(f"📁 Created {len(MODEL_NAMES)} model directories with 4 subdirectories each")

# ============================================================================
# 🎯 ENHANCED DIFFICULT DATASET
# ============================================================================

class DifficultAnomalyDataset(Dataset):
    """향상된 난이도의 이상 탐지 데이터셋 - Normal에 가까운 이상 패턴"""
    
    def __init__(self, mode='train', size=None, difficulty='hard'):
        np.random.seed(CONFIG['SEED'] if mode == 'train' else CONFIG['SEED'] + 1)
        
        self.mode = mode
        self.difficulty = difficulty
        data_size = size if size else (CONFIG['DATA_SIZE'] if mode == 'train' else CONFIG['DATA_SIZE'] // 4)
        
        print(f"📊 Generating {mode} dataset: {data_size} samples (difficulty: {difficulty})")
        
        self.data = []
        self.point_labels = []
        self.series_labels = []
        self.anomaly_types = []
        self.anomaly_info = []  # 상세 정보
        
        # 난이도별 설정
        if difficulty == 'easy':
            self.noise_std = 0.05
            self.anomaly_strength = (2.0, 4.0)
        elif difficulty == 'medium':
            self.noise_std = 0.08
            self.anomaly_strength = (1.2, 2.5)
        else:  # hard
            self.noise_std = 0.12
            self.anomaly_strength = (0.8, 1.5)  # Normal에 매우 가까움
        
        for idx in range(data_size):
            series, point_label, series_label, atype, info = self._generate_difficult_series(idx)
            self.data.append(series)
            self.point_labels.append(point_label)
            self.series_labels.append(series_label)
            self.anomaly_types.append(atype)
            self.anomaly_info.append(info)
        
        # 통계 출력
        normal_count = sum(1 for x in self.series_labels if x == 0)
        anomaly_count = len(self.series_labels) - normal_count
        
        type_counts = {}
        for atype in self.anomaly_types:
            type_counts[atype] = type_counts.get(atype, 0) + 1
        
        print(f"✅ {mode} dataset complete: Normal {normal_count}, Anomaly {anomaly_count}")
        print(f"   📊 Type distribution: {type_counts}")
    
    def _generate_difficult_series(self, idx):
        """Normal에 가까운 어려운 이상 패턴 생성"""
        series = np.random.normal(0, self.noise_std, CONFIG['SEQ_LEN']).astype(np.float32)
        point_label = np.zeros(CONFIG['SEQ_LEN'], dtype=np.float32)
        
        anomaly_type = idx % 6  # 6가지 타입
        
        if anomaly_type == 0:
            # Normal
            series_label = 0.0
            type_name = 'Normal'
            info = {'type': 'normal', 'start': 0, 'end': 0, 'magnitude': 0.0, 'count': 0}
            
        elif anomaly_type == 1:
            # Subtle Spike - 매우 약한 스파이크
            n_spikes = np.random.randint(1, 3)
            spike_positions = []
            for _ in range(n_spikes):
                pos = np.random.randint(10, CONFIG['SEQ_LEN']-10)
                # 약한 스파이크 (기존 대비 1/3 강도)
                magnitude = np.random.uniform(*self.anomaly_strength) * np.random.choice([-1, 1])
                series[pos] += magnitude
                point_label[pos] = 1.0
                
                # 주변 영향도 줄임
                for offset in [-1, 1]:
                    if 0 <= pos + offset < CONFIG['SEQ_LEN']:
                        series[pos + offset] += magnitude * 0.15  # 기존 0.3에서 0.15로
                        point_label[pos + offset] = 1.0
                
                spike_positions.append(pos)
            
            series_label = 1.0
            type_name = 'Subtle_Spike'
            info = {'type': 'spike', 'start': spike_positions[0] if spike_positions else 0, 'end': spike_positions[-1] if spike_positions else 0, 'magnitude': magnitude, 'count': n_spikes}
            
        elif anomaly_type == 2:
            # Gradual Mean Shift - 점진적 변화
            start = np.random.randint(15, CONFIG['SEQ_LEN']-25)
            end = start + np.random.randint(15, 25)
            
            # 점진적 변화 (급격하지 않음)
            shift_magnitude = np.random.uniform(*self.anomaly_strength) * np.random.choice([-1, 1])
            transition_length = min(8, (end - start) // 3)
            
            # 시작 부분 점진적 증가
            for i in range(transition_length):
                alpha = i / transition_length
                series[start + i] += shift_magnitude * alpha
                point_label[start + i] = alpha  # 그라데이션 라벨
            
            # 중간 부분 일정
            mid_start = start + transition_length
            mid_end = end - transition_length
            if mid_end > mid_start:
                series[mid_start:mid_end] += shift_magnitude
                point_label[mid_start:mid_end] = 1.0
            
            # 끝 부분 점진적 감소
            for i in range(transition_length):
                alpha = 1 - (i / transition_length)
                series[end - transition_length + i] += shift_magnitude * alpha
                point_label[end - transition_length + i] = alpha
            
            series_label = 1.0
            type_name = 'Gradual_Shift'
            info = {'type': 'shift', 'start': start, 'end': end, 'magnitude': shift_magnitude, 'count': 1}
            
        elif anomaly_type == 3:
            # Subtle Variance Change - 미묘한 분산 변화
            start = np.random.randint(10, CONFIG['SEQ_LEN']-20)
            end = start + np.random.randint(15, 25)
            
            # 기존 노이즈 대비 1.5-2.5배 (기존 3-6배에서 줄임)
            new_std = self.noise_std * np.random.uniform(1.5, 2.5)
            length = end - start
            if length > 0:
                series[start:end] = np.random.normal(0, new_std, length)
                point_label[start:end] = 1.0
            
            series_label = 1.0
            type_name = 'Subtle_Variance'
            info = {'type': 'variance', 'start': start, 'end': end, 'magnitude': new_std, 'count': 1}
            
        elif anomaly_type == 4:
            # Slow Trend - 천천히 변하는 트렌드
            start = np.random.randint(5, CONFIG['SEQ_LEN']//3)
            end = start + np.random.randint(CONFIG['SEQ_LEN']//3, CONFIG['SEQ_LEN']//2)
            end = min(end, CONFIG['SEQ_LEN'])
            
            # 약한 트렌드
            trend_magnitude = np.random.uniform(*self.anomaly_strength) * np.random.choice([-1, 1])
            length = end - start
            if length > 0:
                trend = np.linspace(0, trend_magnitude, length)
                series[start:end] += trend
                
                # 트렌드 강도에 따른 라벨 (선형 증가)
                trend_labels = np.linspace(0.3, 1.0, length)
                point_label[start:end] = trend_labels
            
            series_label = 1.0
            type_name = 'Slow_Trend'
            info = {'type': 'trend', 'start': start, 'end': end, 'magnitude': trend_magnitude, 'count': 1}
            
        else:  # anomaly_type == 5
            # Complex Pattern - 복합 패턴 (매우 어려움)
            # 작은 스파이크 + 미묘한 트렌드
            
            # 작은 스파이크
            spike_pos = np.random.randint(10, CONFIG['SEQ_LEN']//2)
            spike_mag = np.random.uniform(*self.anomaly_strength) * 0.7  # 더 약하게
            series[spike_pos] += spike_mag
            point_label[spike_pos] = 0.8
            
            # 미묘한 트렌드
            trend_start = CONFIG['SEQ_LEN']//2
            trend_end = CONFIG['SEQ_LEN'] - 5
            trend_mag = np.random.uniform(*self.anomaly_strength) * 0.5 * np.random.choice([-1, 1])
            trend_length = trend_end - trend_start
            if trend_length > 0:
                trend = np.linspace(0, trend_mag, trend_length)
                series[trend_start:trend_end] += trend
                point_label[trend_start:trend_end] = np.linspace(0.3, 0.7, trend_length)
            
            series_label = 1.0
            type_name = 'Complex_Pattern'
            info = {'type': 'complex', 'start': spike_pos, 'end': trend_end, 'magnitude': spike_mag, 'count': 2}
        
        return series, point_label, series_label, type_name, info
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx]).unsqueeze(-1),  # [seq_len, 1]
            torch.from_numpy(self.point_labels[idx]),        # [seq_len]
            torch.tensor(self.series_labels[idx], dtype=torch.float32),  # scalar
            self.anomaly_types[idx],                         # string
            self.anomaly_info[idx]                          # dict
        ) 

# ============================================================================
# 🤖 MODEL IMPORTS AND HYPERPARAMETERS
# ============================================================================

def safe_model_import():
    """안전한 모델 import 및 fallback 처리"""
    models = {}
    
    # 1. CARLA 모델
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'models/carla'))
        from models.carla.model import create_carla_model
        models['carla'] = create_carla_model
        print("✅ CARLA model imported successfully")
    except Exception as e:
        print(f"❌ CARLA import failed: {e}")
        models['carla'] = None
    
    # 2. TraceGPT 모델
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'models/tracegpt'))
        from models.tracegpt.model import create_tracegpt_model
        models['tracegpt'] = create_tracegpt_model
        print("✅ TraceGPT model imported successfully")
    except Exception as e:
        print(f"❌ TraceGPT import failed: {e}")
        models['tracegpt'] = None
    
    # 3. PatchTrAD 모델
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'models/patchtrad'))
        from models.patchtrad.model import create_patchtrad_model
        models['patchtrad'] = create_patchtrad_model
        print("✅ PatchTrAD model imported successfully")
    except Exception as e:
        print(f"❌ PatchTrAD import failed: {e}")
        models['patchtrad'] = None
    
    # 4. ProDiffAD 모델
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'models/prodiffad'))
        from models.prodiffad.model import create_prodiffad_model
        models['prodiffad'] = create_prodiffad_model
        print("✅ ProDiffAD model imported successfully")
    except Exception as e:
        print(f"❌ ProDiffAD import failed: {e}")
        models['prodiffad'] = None
    
    return models

# Fallback 단순 모델
class FallbackAnomalyModel(nn.Module):
    def __init__(self, seq_len=64, hidden_dim=128):
        super().__init__()
        self.encoder = nn.LSTM(1, hidden_dim//2, 2, batch_first=True, bidirectional=True)
        self.decoder = nn.Linear(hidden_dim, 1)
        self.reconstructor = nn.Linear(hidden_dim, seq_len)
        
    def forward(self, x):
        lstm_out, _ = self.encoder(x)
        anomaly_scores = torch.sigmoid(self.decoder(lstm_out[:, -1, :]))
        reconstruction = self.reconstructor(lstm_out[:, -1, :]).unsqueeze(-1)
        return anomaly_scores.squeeze(), reconstruction.expand_as(x)
    
    def detect_anomalies(self, x):
        with torch.no_grad():
            anomaly_scores, reconstruction = self.forward(x)
            recon_error = torch.mean((x - reconstruction) ** 2, dim=[1, 2])
            point_scores = torch.mean((x - reconstruction) ** 2, dim=2)
        return recon_error, point_scores

# 모델별 최적 하이퍼파라미터
MODEL_HP = {
    'carla': {
        'seq_len': CONFIG['SEQ_LEN'],
        'hidden_dim': 256,
        'encoder_layers': 3,
        'temperature': 0.07,  # 더 낮은 온도로 더 집중적 학습
        'margin': 1.2,        # 더 큰 마진으로 구분력 향상
        'dropout': 0.1,
        'lr': 1e-4,
        'epochs': 8
    },
    'tracegpt': {
        'seq_len': CONFIG['SEQ_LEN'],
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 8,        # 더 깊은 네트워크
        'd_ff': 1024,
        'dropout': 0.1,
        'lr': 5e-5,           # 더 낮은 학습률
        'epochs': 10
    },
    'patchtrad': {
        'seq_len': CONFIG['SEQ_LEN'],
        'patch_size': 8,      # 더 큰 패치로 글로벌 패턴 포착
        'stride': 4,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 8,
        'd_ff': 1024,
        'dropout': 0.1,
        'lr': 1e-4,
        'epochs': 8
    },
    'prodiffad': {
        'seq_len': CONFIG['SEQ_LEN'],
        'hidden_dim': 256,
        'num_layers': 6,
        'dropout': 0.1,
        'lr': 1e-4,
        'epochs': 12         # Diffusion은 더 많은 epochs 필요
    },
    'fallback': {
        'seq_len': CONFIG['SEQ_LEN'],
        'hidden_dim': 128,
        'lr': 1e-3,
        'epochs': 5
    }
}

# 앙상블 모델들
class PatchTraceEnsemble(nn.Module):
    """PatchTrAD + TraceGPT 앙상블"""
    def __init__(self, models_dict):
        super().__init__()
        self.patch_model = models_dict.get('patchtrad')
        self.trace_model = models_dict.get('tracegpt')
        self.weight = nn.Parameter(torch.tensor([0.6, 0.4]))  # 가중치 학습
        
    def detect_anomalies(self, x):
        if self.patch_model and self.trace_model:
            with torch.no_grad():
                patch_series, patch_point = self.patch_model.detect_anomalies(x)
                trace_series, trace_point = self.trace_model.detect_anomalies(x)
                
                weights = torch.softmax(self.weight, dim=0)
                combined_series = weights[0] * patch_series + weights[1] * trace_series
                combined_point = weights[0] * patch_point + weights[1] * trace_point
                
                return combined_series, combined_point
        return torch.zeros(x.shape[0]), torch.zeros(x.shape[0], x.shape[1])

class TransferLearningEnsemble(nn.Module):
    """전이학습 기반 앙상블"""
    def __init__(self, models_dict):
        super().__init__()
        self.carla_model = models_dict.get('carla')
        self.trace_model = models_dict.get('tracegpt')
        
    def detect_anomalies(self, x):
        if self.carla_model and self.trace_model:
            with torch.no_grad():
                carla_series, carla_point = self.carla_model.detect_anomalies(x)
                trace_series, trace_point = self.trace_model.detect_anomalies(x)
                
                # CARLA의 representation learning 활용
                combined_series = 0.7 * carla_series + 0.3 * trace_series
                combined_point = 0.7 * carla_point + 0.3 * trace_point
                
                return combined_series, combined_point
        return torch.zeros(x.shape[0]), torch.zeros(x.shape[1])

class MultiModelEnsemble(nn.Module):
    """전체 모델 앙상블"""
    def __init__(self, models_dict):
        super().__init__()
        self.models = {k: v for k, v in models_dict.items() if v is not None}
        self.weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))
        
    def detect_anomalies(self, x):
        if not self.models:
            return torch.zeros(x.shape[0]), torch.zeros(x.shape[0], x.shape[1])
            
        with torch.no_grad():
            series_scores = []
            point_scores = []
            
            for model in self.models.values():
                series, point = model.detect_anomalies(x)
                series_scores.append(series)
                point_scores.append(point)
            
            # 가중평균
            weights = torch.softmax(self.weights, dim=0)
            combined_series = sum(w * s for w, s in zip(weights, series_scores))
            combined_point = sum(w * p for w, p in zip(weights, point_scores))
            
            return combined_series, combined_point

# 모델 생성 함수
def create_all_models():
    """모든 모델 생성 및 설정"""
    print("🚀 Creating all models...")
    
    # Import models
    imported_models = safe_model_import()
    models = {}
    
    # 개별 모델들
    for name, create_func in imported_models.items():
        # 모델 생성용 하이퍼파라미터 (lr, epochs 제외)
        model_params = {k: v for k, v in MODEL_HP[name].items() if k not in ['lr', 'epochs']}
        fallback_params = {k: v for k, v in MODEL_HP['fallback'].items() if k not in ['lr', 'epochs']}
        
        if create_func is not None:
            try:
                model = create_func(**model_params)
                if GPU_CONFIG['device'] != 'cpu':
                    model = model.to(GPU_CONFIG['device'])
                models[name] = model
                print(f"✅ {name.upper()} model created successfully")
            except Exception as e:
                print(f"❌ {name.upper()} model creation failed: {e}")
                # Fallback 모델 사용
                model = FallbackAnomalyModel(**fallback_params)
                if GPU_CONFIG['device'] != 'cpu':
                    model = model.to(GPU_CONFIG['device'])
                models[name] = model
                print(f"🔄 Using fallback model for {name.upper()}")
        else:
            # Fallback 모델
            model = FallbackAnomalyModel(**fallback_params)
            if GPU_CONFIG['device'] != 'cpu':
                model = model.to(GPU_CONFIG['device'])
            models[name] = model
            print(f"🔄 Using fallback model for {name.upper()}")
    
    # 앙상블 모델들
    models['patch_trace_ensemble'] = PatchTraceEnsemble(models)
    models['transfer_learning_ensemble'] = TransferLearningEnsemble(models)
    models['multi_model_ensemble'] = MultiModelEnsemble(models)
    
    if GPU_CONFIG['device'] != 'cpu':
        for name in ['patch_trace_ensemble', 'transfer_learning_ensemble', 'multi_model_ensemble']:
            models[name] = models[name].to(GPU_CONFIG['device'])
    
    print(f"🎯 Total {len(models)} models created")
    return models 

# ============================================================================
# 📊 DETAILED VISUALIZATION AND EVALUATION
# ============================================================================

def create_detailed_plot(data, point_true, point_pred, series_true, series_pred, 
                        threshold, model_name, sample_idx, anomaly_type, save_dir):
    """상세한 개별 시계열 플롯 생성"""
    if not MATPLOTLIB_AVAILABLE:
        return
        
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Time axis
    time_axis = np.arange(len(data))
    
    # 1. Original Signal with Anomalies
    axes[0].plot(time_axis, data, 'b-', linewidth=1.5, label='Signal', alpha=0.8)
    
    # True anomalies
    true_anomaly_mask = point_true > 0.5
    if np.any(true_anomaly_mask):
        axes[0].scatter(time_axis[true_anomaly_mask], data[true_anomaly_mask], 
                       color='red', s=30, label='True Anomalies', alpha=0.7, zorder=5)
    
    # Predicted anomalies
    pred_anomaly_mask = point_pred > threshold
    if np.any(pred_anomaly_mask):
        axes[0].scatter(time_axis[pred_anomaly_mask], data[pred_anomaly_mask], 
                       color='orange', s=20, marker='x', label='Pred Anomalies', alpha=0.7, zorder=4)
    
    axes[0].set_title(f'{model_name} - Sample {sample_idx} ({anomaly_type})\n'
                     f'Series: True={series_true:.1f}, Pred={series_pred:.3f}', fontsize=12)
    axes[0].set_ylabel('Signal Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Anomaly Scores
    axes[1].plot(time_axis, point_pred, 'g-', linewidth=2, label='Anomaly Score')
    axes[1].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.3f})')
    
    # Fill anomaly zones
    axes[1].fill_between(time_axis, 0, point_pred, 
                        where=(point_pred > threshold), 
                        color='red', alpha=0.2, label='Anomaly Zone')
    
    axes[1].set_ylabel('Anomaly Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # 3. True vs Predicted Comparison
    axes[2].plot(time_axis, point_true, 'r-', linewidth=2, label='True Labels', alpha=0.8)
    axes[2].plot(time_axis, point_pred, 'g-', linewidth=2, label='Pred Scores', alpha=0.8)
    axes[2].axhline(y=threshold, color='black', linestyle='--', linewidth=1, label=f'Threshold')
    
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Score/Label')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    filename = f'{save_dir}/sample_{sample_idx}_{anomaly_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def create_confusion_matrix_plot(true_labels, pred_labels, model_name, level, save_dir):
    """혼동 행렬 생성"""
    if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        return None
        
    # 이진 분류로 변환
    true_binary = (true_labels > 0.5).astype(int)
    pred_binary = (pred_labels > 0.5).astype(int)
    
    cm = confusion_matrix(true_binary, pred_binary)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # 클래스 이름
    classes = ['Normal', 'Anomaly']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f'{model_name} - {level} Level Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # 텍스트 주석
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filename = f'{save_dir}/{model_name}_{level}_confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def calculate_detailed_metrics(true_labels, pred_scores, threshold=0.5):
    """상세 메트릭 계산"""
    if not SKLEARN_AVAILABLE:
        return {}
        
    # 이진 예측값
    pred_binary = (pred_scores > threshold).astype(int)
    true_binary = (true_labels > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(true_binary, pred_binary),
        'precision': precision_score(true_binary, pred_binary, zero_division='warn'),
        'recall': recall_score(true_binary, pred_binary, zero_division='warn'),
        'f1': f1_score(true_binary, pred_binary, zero_division='warn'),
        'threshold': threshold
    }
    
    # AUC 계산 (가능한 경우)
    try:
        metrics['auc'] = roc_auc_score(true_binary, pred_scores)
    except:
        metrics['auc'] = 0.0
        
    return metrics

def save_sample_data(dataset, model_name, num_samples=10):
    """샘플 데이터 저장"""
    if not MATPLOTLIB_AVAILABLE:
        return
        
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(dataset))):
        data, point_labels, series_label, anomaly_type, info = dataset[i]
        
        ax = axes[i]
        time_axis = np.arange(len(data))
        
        # 신호 플롯
        ax.plot(time_axis, data.squeeze().numpy(), 'b-', linewidth=1.5)
        
        # 이상치 포인트 하이라이트
        anomaly_mask = point_labels.numpy() > 0.5
        if np.any(anomaly_mask):
            ax.scatter(time_axis[anomaly_mask], data.squeeze().numpy()[anomaly_mask], 
                      color='red', s=20, alpha=0.7)
        
        ax.set_title(f'{anomaly_type}\n(Series: {series_label:.1f})', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if i >= 5:
            ax.set_xlabel('Time Step')
        if i % 5 == 0:
            ax.set_ylabel('Value')
    
    plt.suptitle(f'{model_name} - Sample Data', fontsize=16)
    plt.tight_layout()
    
    filename = f'samples/{model_name}_samples.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename 

# ============================================================================
# 🏋️ TRAINING AND EVALUATION ENGINE
# ============================================================================

def train_single_model(model, model_name, train_loader, optimizer, device, epochs):
    """단일 모델 훈련"""
    print(f"🔥 Training {model_name.upper()} for {epochs} epochs...")
    
    model.train()
    total_loss = 0
    num_batches = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, point_labels, series_labels, anomaly_types, info) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # 모델별 loss 계산
            if hasattr(model, 'compute_loss'):  # CARLA, TraceGPT, PatchTrAD
                loss = model.compute_loss(data)
            else:  # Fallback 모델
                pred_scores, reconstruction = model(data)
                # 재구성 손실 + 이상 탐지 손실
                recon_loss = F.mse_loss(reconstruction, data)
                
                # 시리즈별 이상 스코어 생성
                series_labels_tensor = series_labels.to(device)
                anomaly_loss = F.binary_cross_entropy(pred_scores, series_labels_tensor)
                
                loss = recon_loss + anomaly_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"   Epoch {epoch+1} completed. Avg Loss: {avg_epoch_loss:.6f}")
    
    avg_total_loss = total_loss / num_batches
    print(f"✅ {model_name.upper()} training completed. Avg Loss: {avg_total_loss:.6f}")
    
    return avg_total_loss

def evaluate_model_comprehensive(model, model_name, test_loader, device, threshold=0.5):
    """포괄적 모델 평가"""
    print(f"📊 Evaluating {model_name.upper()}...")
    
    model.eval()
    
    # 결과 저장용 리스트들
    all_data = []
    all_point_true = []
    all_series_true = []
    all_point_pred = []
    all_series_pred = []
    all_anomaly_types = []
    all_anomaly_info = []
    
    with torch.no_grad():
        for data, point_labels, series_labels, anomaly_types, info in test_loader:
            data = data.to(device)
            
            # 모델 예측
            series_scores, point_scores = model.detect_anomalies(data)
            
            # CPU로 이동 및 numpy 변환
            data_np = data.cpu().numpy()
            point_true_np = point_labels.numpy()
            series_true_np = series_labels.numpy()
            point_pred_np = point_scores.cpu().numpy()
            series_pred_np = series_scores.cpu().numpy()
            
            # 배치 데이터 저장
            all_data.extend(data_np)
            all_point_true.extend(point_true_np)
            all_series_true.extend(series_true_np)
            all_point_pred.extend(point_pred_np)
            all_series_pred.extend(series_pred_np)
            all_anomaly_types.extend(anomaly_types)
            all_anomaly_info.extend(info)
    
    # Numpy 배열로 변환
    all_data = np.array(all_data)
    all_point_true = np.array(all_point_true)
    all_series_true = np.array(all_series_true)
    all_point_pred = np.array(all_point_pred)
    all_series_pred = np.array(all_series_pred)
    
    # 메트릭 계산
    series_metrics = calculate_detailed_metrics(all_series_true, all_series_pred, threshold)
    
    # Point-level 메트릭 (평평하게 만들어서 계산)
    point_true_flat = all_point_true.flatten()
    point_pred_flat = all_point_pred.flatten()
    point_metrics = calculate_detailed_metrics(point_true_flat, point_pred_flat, threshold)
    
    # 결과 출력
    print(f"🎯 {model_name.upper()} Results:")
    print(f"   Series Level - Acc: {series_metrics['accuracy']:.3f}, F1: {series_metrics['f1']:.3f}, AUC: {series_metrics['auc']:.3f}")
    print(f"   Point Level  - Acc: {point_metrics['accuracy']:.3f}, F1: {point_metrics['f1']:.3f}, AUC: {point_metrics['auc']:.3f}")
    
    # 상세 플롯 생성 (샘플별)
    create_detailed_plots_by_classification(
        all_data, all_point_true, all_point_pred, all_series_true, all_series_pred,
        all_anomaly_types, threshold, model_name
    )
    
    # Confusion Matrix 생성
    create_confusion_matrix_plot(all_series_true, all_series_pred, model_name, 'Series', 'confusion_matrices')
    create_confusion_matrix_plot(point_true_flat, point_pred_flat, model_name, 'Point', 'confusion_matrices')
    
    return {
        'series_metrics': series_metrics,
        'point_metrics': point_metrics,
        'predictions': {
            'data': all_data,
            'point_true': all_point_true,
            'series_true': all_series_true,
            'point_pred': all_point_pred,
            'series_pred': all_series_pred,
            'anomaly_types': all_anomaly_types
        }
    }

def create_detailed_plots_by_classification(data, point_true, point_pred, series_true, series_pred, 
                                          anomaly_types, threshold, model_name, max_per_category=5):
    """분류별 상세 플롯 생성"""
    if not MATPLOTLIB_AVAILABLE:
        return
        
    # 예측 결과에 따른 분류
    series_pred_binary = (series_pred > threshold).astype(int)
    series_true_binary = (series_true > 0.5).astype(int)
    
    # 4가지 카테고리
    tp_indices = np.where((series_true_binary == 1) & (series_pred_binary == 1))[0]  # True Positive
    tn_indices = np.where((series_true_binary == 0) & (series_pred_binary == 0))[0]  # True Negative
    fp_indices = np.where((series_true_binary == 0) & (series_pred_binary == 1))[0]  # False Positive
    fn_indices = np.where((series_true_binary == 1) & (series_pred_binary == 0))[0]  # False Negative
    
    categories = {
        'true_positive': tp_indices,
        'true_negative': tn_indices,
        'false_positive': fp_indices,
        'false_negative': fn_indices
    }
    
    print(f"📈 Creating detailed plots for {model_name}:")
    for category, indices in categories.items():
        print(f"   {category}: {len(indices)} samples")
        
        # 각 카테고리에서 최대 max_per_category개 샘플 저장
        selected_indices = indices[:max_per_category] if len(indices) > max_per_category else indices
        
        for i, idx in enumerate(selected_indices):
            save_dir = f'plots/{model_name}/{category}'
            
            create_detailed_plot(
                data=data[idx].squeeze(),
                point_true=point_true[idx],
                point_pred=point_pred[idx],
                series_true=series_true[idx],
                series_pred=series_pred[idx],
                threshold=threshold,
                model_name=model_name,
                sample_idx=idx,
                anomaly_type=anomaly_types[idx],
                save_dir=save_dir
            )

def create_summary_metrics_plot(all_results):
    """전체 모델 성능 요약 플롯"""
    if not MATPLOTLIB_AVAILABLE:
        return
        
    models = list(all_results.keys())
    
    # Series-level 메트릭
    series_acc = [all_results[m]['series_metrics']['accuracy'] for m in models]
    series_f1 = [all_results[m]['series_metrics']['f1'] for m in models]
    series_auc = [all_results[m]['series_metrics']['auc'] for m in models]
    
    # Point-level 메트릭
    point_acc = [all_results[m]['point_metrics']['accuracy'] for m in models]
    point_f1 = [all_results[m]['point_metrics']['f1'] for m in models]
    point_auc = [all_results[m]['point_metrics']['auc'] for m in models]
    
    # 플롯 생성
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics_data = [
        (series_acc, 'Series Accuracy', axes[0, 0]),
        (series_f1, 'Series F1-Score', axes[0, 1]),
        (series_auc, 'Series AUC', axes[0, 2]),
        (point_acc, 'Point Accuracy', axes[1, 0]),
        (point_f1, 'Point F1-Score', axes[1, 1]),
        (point_auc, 'Point AUC', axes[1, 2])
    ]
    
    for data, title, ax in metrics_data:
        bars = ax.bar(models, data, alpha=0.7)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, value in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    
    filename = 'metrics/final_performance_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Summary metrics saved to {filename}")
    return filename

# ============================================================================
# 🚀 MAIN EXECUTION
# ============================================================================

def main():
    """메인 실행 함수"""
    print("🚀 Starting Final Complete Anomaly Detection System")
    print(f"📱 Device: {GPU_CONFIG['device']}")
    print(f"🔧 Configuration: {CONFIG}")
    
    # Set random seeds
    torch.manual_seed(CONFIG['SEED'])
    np.random.seed(CONFIG['SEED'])
    
    # Create datasets
    print("\n📊 Creating datasets...")
    train_dataset = DifficultAnomalyDataset('train', CONFIG['DATA_SIZE'], 'hard')
    test_dataset = DifficultAnomalyDataset('test', CONFIG['DATA_SIZE']//4, 'hard')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=0)
    
    print(f"✅ Datasets created: Train {len(train_dataset)}, Test {len(test_dataset)}")
    
    # Save sample data
    save_sample_data(test_dataset, 'final_system', 10)
    
    # Create all models
    print("\n🤖 Creating models...")
    models = create_all_models()
    
    # Training and evaluation
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"🔥 Processing {model_name.upper()}")
        print(f"{'='*60}")
        
        # Skip ensemble models for training (they don't need training)
        if 'ensemble' not in model_name:
            # Create optimizer
            hp = MODEL_HP.get(model_name, MODEL_HP['fallback'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=1e-5)
            
            # Train model
            train_loss = train_single_model(
                model, model_name, train_loader, optimizer, 
                GPU_CONFIG['device'], hp['epochs']
            )
            
            # Save model
            torch.save(model.state_dict(), f'pre_trained/{model_name}_final.pth')
            print(f"💾 Model saved to pre_trained/{model_name}_final.pth")
        
        # Evaluate model
        results = evaluate_model_comprehensive(
            model, model_name, test_loader, GPU_CONFIG['device'], CONFIG['THRESHOLD']
        )
        
        all_results[model_name] = results
        
        print(f"✅ {model_name.upper()} processing completed")
    
    # Create summary plots
    print("\n📊 Creating summary visualizations...")
    create_summary_metrics_plot(all_results)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("🎯 FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for model_name, results in all_results.items():
        series_acc = results['series_metrics']['accuracy']
        point_acc = results['point_metrics']['accuracy']
        series_f1 = results['series_metrics']['f1']
        point_f1 = results['point_metrics']['f1']
        
        print(f"{model_name.upper():25} | Series: Acc={series_acc:.3f} F1={series_f1:.3f} | Point: Acc={point_acc:.3f} F1={point_f1:.3f}")
    
    print(f"\n🎉 Complete system execution finished!")
    print(f"📁 Check outputs in: plots/, metrics/, confusion_matrices/, samples/, pre_trained/")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Final Complete Anomaly Detection System')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['all'] + MODEL_NAMES,
                       help='Model to run (default: all)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override epochs for all models')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--data-size', type=int, default=None,
                       help='Override dataset size')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Anomaly detection threshold')
    parser.add_argument('--difficulty', type=str, default='hard',
                       choices=['easy', 'medium', 'hard'],
                       help='Dataset difficulty level')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.epochs:
        for model_hp in MODEL_HP.values():
            if 'epochs' in model_hp:
                model_hp['epochs'] = args.epochs
    
    if args.batch_size:
        CONFIG['BATCH_SIZE'] = args.batch_size
    
    if args.data_size:
        CONFIG['DATA_SIZE'] = args.data_size
    
    CONFIG['THRESHOLD'] = args.threshold
    
    # Run main function
    try:
        results = main()
        print("\n✅ All operations completed successfully!")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc() 