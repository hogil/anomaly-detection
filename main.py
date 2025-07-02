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
    
# Model-specific plot directories (updated with new ensemble model)
MODEL_NAMES = ['carla', 'tracegpt', 'patchtrad', 'prodiffad', 
               'patch_trace_ensemble', 'transfer_learning_ensemble', 
               'multi_model_ensemble', 'advanced_stacking_ensemble']

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
            mid_end = min(end - transition_length, CONFIG['SEQ_LEN'])
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
                # 배열 크기 체크 및 안전한 할당
                actual_length = min(length, CONFIG['SEQ_LEN'] - start)
                if actual_length > 0:
                    series[start:start+actual_length] = np.random.normal(0, new_std, actual_length)
                    point_label[start:start+actual_length] = 1.0
            
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
                # 배열 크기 안전 체크
                actual_length = min(length, CONFIG['SEQ_LEN'] - start)
                if actual_length > 0:
                    trend = np.linspace(0, trend_magnitude, actual_length)
                    series[start:start+actual_length] += trend
                    
                    # 트렌드 강도에 따른 라벨 (선형 증가)
                    trend_labels = np.linspace(0.3, 1.0, actual_length)
                    point_label[start:start+actual_length] = trend_labels
            
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
                # 배열 크기 안전 체크
                actual_trend_length = min(trend_length, CONFIG['SEQ_LEN'] - trend_start)
                if actual_trend_length > 0:
                    trend = np.linspace(0, trend_mag, actual_trend_length)
                    series[trend_start:trend_start+actual_trend_length] += trend
                    point_label[trend_start:trend_start+actual_trend_length] = np.linspace(0.3, 0.7, actual_trend_length)
            
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
    def __init__(self, seq_len=64, hidden_dim=128, num_layers=2, bidirectional=True, dropout=0.1, **kwargs):
        super().__init__()
        # kwargs에서 무시할 파라미터들 제거
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # LSTM 인코더
        lstm_input_size = 1
        lstm_hidden_size = hidden_dim // 2 if bidirectional else hidden_dim
        
        self.encoder = nn.LSTM(
            lstm_input_size, 
            lstm_hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 디코더
        encoder_output_size = hidden_dim if bidirectional else lstm_hidden_size
        self.decoder = nn.Linear(encoder_output_size, 1)
        self.reconstructor = nn.Linear(encoder_output_size, seq_len)
        
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

# 모델별 최적 하이퍼파라미터 - 대폭 개선 (웹검색 기반)
MODEL_HP = {
    'carla': {
        # CARLA 모델 최적화 (Contrastive Learning 기반)
        'seq_len': CONFIG['SEQ_LEN'],
        'hidden_dim': 512,      # 더 큰 representation capacity
        'encoder_layers': 4,    # 더 깊은 인코더
        'temperature': 0.05,    # 더 낮은 온도로 집중적 학습
        'margin': 1.5,          # 더 큰 마진으로 구분력 향상
        'dropout': 0.15,        # 약간 높은 드롭아웃으로 과적합 방지
        'contrastive_weight': 0.8,  # Contrastive loss 가중치
        'reconstruction_weight': 0.2,  # Reconstruction loss 가중치
        'lr': 3e-5,             # 더 낮은 학습률로 안정적 학습
        'epochs': 25,           # 대폭 증가
        'warmup_epochs': 5,     # 워밍업 추가
        'weight_decay': 1e-4
    },
    'tracegpt': {
        # TraceGPT Transformer 최적화
        'seq_len': CONFIG['SEQ_LEN'],
        'd_model': 512,         # 더 큰 모델 크기
        'n_heads': 16,          # 더 많은 어텐션 헤드
        'n_layers': 12,         # 더 깊은 트랜스포머
        'd_ff': 2048,           # 더 큰 피드포워드 네트워크
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'layer_norm_eps': 1e-6,
        'max_position_embeddings': 1024,
        'lr': 2e-5,             # 트랜스포머에 적합한 낮은 학습률
        'epochs': 30,           # 대폭 증가
        'warmup_steps': 1000,   # 워밍업 스텝
        'weight_decay': 1e-4,
        'gradient_clip': 1.0
    },
    'patchtrad': {
        # PatchTrAD 최적화 (Patch-based Transformer)
        'seq_len': CONFIG['SEQ_LEN'],
        'patch_size': 16,       # 더 큰 패치로 글로벌 패턴 포착
        'stride': 8,            # 패치 간 오버랩
        'd_model': 512,
        'n_heads': 16,
        'n_layers': 10,
        'd_ff': 2048,
        'dropout': 0.1,
        'patch_dropout': 0.1,   # 패치 드롭아웃
        'positional_encoding': 'learnable',
        'lr': 1e-4,
        'epochs': 28,           # 대폭 증가
        'scheduler': 'cosine',   # 코사인 스케줄러
        'weight_decay': 1e-4
    },
    'prodiffad': {
        # ProDiffAD Diffusion 모델 최적화
        'seq_len': CONFIG['SEQ_LEN'],
        'hidden_dim': 512,
        'num_layers': 8,        # 더 깊은 네트워크
        'dropout': 0.1,
        'num_timesteps': 1000,  # Diffusion timesteps
        'beta_schedule': 'cosine',  # 베타 스케줄
        'denoising_steps': 50,
        'lr': 1e-4,
        'epochs': 35,           # Diffusion은 더 많은 epochs 필요
        'ema_decay': 0.9999,    # Exponential Moving Average
        'weight_decay': 1e-5
    },
    'fallback': {
        # 향상된 Fallback 모델
        'seq_len': CONFIG['SEQ_LEN'],
        'hidden_dim': 256,      # 더 큰 크기
        'num_layers': 3,        # 더 깊은 LSTM
        'bidirectional': True,  # 양방향 LSTM
        'dropout': 0.2,
        'lr': 5e-4,             # 약간 낮춘 학습률
        'epochs': 20,           # 증가
        'scheduler': 'step',
        'step_size': 10,
        'gamma': 0.5
    }
}

# 향상된 앙상블 모델들 - Stacking 기반 개선
class AdvancedStackingEnsemble(nn.Module):
    """고급 스태킹 앙상블 - 메타 학습기 포함"""
    def __init__(self, models_dict, device='cpu'):
        super().__init__()
        self.models = {k: v for k, v in models_dict.items() if v is not None and 'ensemble' not in k}
        self.device = device
        
        # 메타 학습기 (2층 신경망)
        meta_input_size = len(self.models) * 2  # series + point scores
        self.meta_learner = nn.Sequential(
            nn.Linear(meta_input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)  # series, point outputs
        ).to(device)
        
        # 모델별 가중치
        self.model_weights = nn.Parameter(torch.ones(len(self.models)))
        
    def forward(self, x):
        """훈련 시 사용"""
        if not self.models:
            return torch.zeros(x.shape[0], 2, device=self.device)
        
        # 모든 모델의 예측 수집
        predictions = []
        for model in self.models.values():
            series, point = model.detect_anomalies(x)
            # 각 시퀀스의 평균 point score
            point_avg = torch.mean(point, dim=1) if len(point.shape) > 1 else point
            predictions.extend([series, point_avg])
        
        # 메타 학습기 입력
        meta_input = torch.stack(predictions, dim=1)
        meta_output = self.meta_learner(meta_input)
        
        return meta_output
    
    def detect_anomalies(self, x):
        """추론 시 사용"""
        with torch.no_grad():
            meta_output = self.forward(x)
            series_scores = torch.sigmoid(meta_output[:, 0])
            point_scores = torch.sigmoid(meta_output[:, 1]).unsqueeze(1).expand(-1, x.shape[1])
            
            return series_scores, point_scores

class PatchTraceEnsemble(nn.Module):
    """향상된 PatchTrAD + TraceGPT 앙상블 - 어텐션 기반 가중치"""
    def __init__(self, models_dict, device='cpu'):
        super().__init__()
        self.patch_model = models_dict.get('patchtrad')
        self.trace_model = models_dict.get('tracegpt')
        self.device = device
        
        # 동적 가중치 계산을 위한 어텐션 모듈
        self.attention = nn.Sequential(
            nn.Linear(2, 16),  # 2개 모델 스코어
            nn.ReLU(),
            nn.Linear(16, 2),  # 가중치 출력
            nn.Softmax(dim=-1)
        ).to(device)
        
        # 성능 기반 초기 가중치 (PatchTrAD가 일반적으로 더 안정적)
        self.base_weights = torch.tensor([0.6, 0.4], device=device)
        
    def detect_anomalies(self, x):
        if self.patch_model and self.trace_model:
            with torch.no_grad():
                patch_series, patch_point = self.patch_model.detect_anomalies(x)
                trace_series, trace_point = self.trace_model.detect_anomalies(x)
                
                # 평균 스코어로 동적 가중치 계산
                avg_scores = torch.stack([
                    torch.mean(patch_series),
                    torch.mean(trace_series)
                ]).unsqueeze(0)
                
                dynamic_weights = self.attention(avg_scores).squeeze(0)
                
                # 기본 가중치와 동적 가중치 결합
                final_weights = 0.7 * self.base_weights + 0.3 * dynamic_weights
                
                combined_series = final_weights[0] * patch_series + final_weights[1] * trace_series
                combined_point = final_weights[0] * patch_point + final_weights[1] * trace_point
                
                return combined_series, combined_point
        return torch.zeros(x.shape[0], device=self.device), torch.zeros(x.shape[0], x.shape[1], device=self.device)

class TransferLearningEnsemble(nn.Module):
    """향상된 전이학습 기반 앙상블 - CARLA 특성 활용"""
    def __init__(self, models_dict, device='cpu'):
        super().__init__()
        self.carla_model = models_dict.get('carla')
        self.trace_model = models_dict.get('tracegpt')
        self.device = device
        
        # CARLA의 contrastive learning 특성을 활용한 confidence 계산
        self.confidence_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        ).to(device)
        
    def detect_anomalies(self, x):
        if self.carla_model and self.trace_model:
            with torch.no_grad():
                carla_series, carla_point = self.carla_model.detect_anomalies(x)
                trace_series, trace_point = self.trace_model.detect_anomalies(x)
                
                # Confidence score 계산 (CARLA가 더 확신할 때 가중치 증가)
                confidence_input = torch.stack([
                    torch.mean(carla_series),
                    torch.var(carla_series)  # 분산이 낮으면 더 확신
                ]).unsqueeze(0)
                
                carla_confidence = self.confidence_net(confidence_input).item()
                
                # 동적 가중치 (CARLA 확신도에 따라 조정)
                carla_weight = 0.6 + 0.3 * carla_confidence  # 0.6~0.9
                trace_weight = 1.0 - carla_weight
                
                combined_series = carla_weight * carla_series + trace_weight * trace_series
                combined_point = carla_weight * carla_point + trace_weight * trace_point
                
                return combined_series, combined_point
        return torch.zeros(x.shape[0], device=self.device), torch.zeros(x.shape[1], device=self.device)

class MultiModelEnsemble(nn.Module):
    """향상된 전체 모델 앙상블 - 다양성과 정확도 균형"""
    def __init__(self, models_dict, device='cpu'):
        super().__init__()
        self.models = {k: v for k, v in models_dict.items() if v is not None and 'ensemble' not in k}
        self.device = device
        
        if self.models:
            # 모델별 성능 기반 초기 가중치
            performance_weights = {
                'carla': 0.25,      # Contrastive learning 우수
                'tracegpt': 0.3,    # Transformer 강력함
                'patchtrad': 0.25,  # Patch 기반 안정적
                'prodiffad': 0.2    # Diffusion 모델 혁신적
            }
            
            weights = [performance_weights.get(k, 1.0) for k in self.models.keys()]
            self.weights = nn.Parameter(torch.tensor(weights, device=device))
            
            # 다양성 보상 메커니즘
            self.diversity_factor = nn.Parameter(torch.tensor(0.1, device=device))
            
    def detect_anomalies(self, x):
        if not self.models:
            return torch.zeros(x.shape[0], device=self.device), torch.zeros(x.shape[0], x.shape[1], device=self.device)
            
        with torch.no_grad():
            series_scores = []
            point_scores = []
            
            for model in self.models.values():
                series, point = model.detect_anomalies(x)
                series_scores.append(series)
                point_scores.append(point)
            
            # 정규화된 가중치
            weights = torch.softmax(self.weights, dim=0)
            
            # 기본 가중 평균
            combined_series = sum(w * s for w, s in zip(weights, series_scores))
            combined_point = sum(w * p for w, p in zip(weights, point_scores))
            
            # 다양성 보상: 모델 간 불일치가 클 때 조정
            if len(series_scores) > 1:
                series_var = torch.var(torch.stack(series_scores), dim=0)
                diversity_bonus = torch.tanh(self.diversity_factor * series_var)
                combined_series = combined_series + diversity_bonus
                combined_series = torch.clamp(combined_series, 0, 1)
            
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
        # 모델 생성용 기본 하이퍼파라미터 (훈련 관련 제외)
        basic_params = ['seq_len', 'hidden_dim', 'd_model', 'n_heads', 'n_layers', 'd_ff', 'dropout',
                       'patch_size', 'stride', 'encoder_layers', 'temperature', 'margin', 'num_layers']
        
        model_params = {k: v for k, v in MODEL_HP[name].items() 
                       if k in basic_params and k not in ['lr', 'epochs']}
        
        fallback_params = {k: v for k, v in MODEL_HP['fallback'].items() 
                          if k in basic_params and k not in ['lr', 'epochs']}
        
        if create_func is not None:
            try:
                print(f"🔧 Creating {name.upper()} with params: {model_params}")
                model = create_func(**model_params)
                if GPU_CONFIG['device'] != 'cpu':
                    model = model.to(GPU_CONFIG['device'])
                models[name] = model
                print(f"✅ {name.upper()} model created successfully")
            except Exception as e:
                print(f"❌ {name.upper()} model creation failed: {e}")
                # Fallback 모델 사용
                print(f"🔄 Creating fallback model with params: {fallback_params}")
                model = FallbackAnomalyModel(**fallback_params)
                if GPU_CONFIG['device'] != 'cpu':
                    model = model.to(GPU_CONFIG['device'])
                models[name] = model
                print(f"🔄 Using fallback model for {name.upper()}")
        else:
            # Fallback 모델
            print(f"🔄 Creating fallback model for {name.upper()} with params: {fallback_params}")
            model = FallbackAnomalyModel(**fallback_params)
            if GPU_CONFIG['device'] != 'cpu':
                model = model.to(GPU_CONFIG['device'])
            models[name] = model
            print(f"🔄 Using fallback model for {name.upper()}")
    
    # 향상된 앙상블 모델들
    device = GPU_CONFIG['device']
    models['patch_trace_ensemble'] = PatchTraceEnsemble(models, device)
    models['transfer_learning_ensemble'] = TransferLearningEnsemble(models, device)
    models['multi_model_ensemble'] = MultiModelEnsemble(models, device)
    models['advanced_stacking_ensemble'] = AdvancedStackingEnsemble(models, device)
    
    if GPU_CONFIG['device'] != 'cpu':
        ensemble_names = ['patch_trace_ensemble', 'transfer_learning_ensemble', 
                         'multi_model_ensemble', 'advanced_stacking_ensemble']
        for name in ensemble_names:
            models[name] = models[name].to(GPU_CONFIG['device'])
    
    print(f"🎯 Total {len(models)} models created (including 4 ensemble models)")
    return models 

# ============================================================================
# 📊 DETAILED VISUALIZATION AND EVALUATION
# ============================================================================

def create_detailed_plot(data, point_true, point_pred, series_true, series_pred, 
                        threshold, model_name, sample_idx, anomaly_type, save_dir):
    """개선된 상세 시계열 플롯 생성 - 사용자 요청사항 반영"""
    if not MATPLOTLIB_AVAILABLE:
        return
        
    # 2x1 서브플롯 (3번째 플롯을 2번째와 합침)
    fig, axes = plt.subplots(2, 1, figsize=(20, 16))
    
    # Time axis
    time_axis = np.arange(len(data))
    
    # 1. Signal + True/Pred Anomalies + Anomaly Zone
    axes[0].plot(time_axis, data, 'b-', linewidth=4, label='Signal', alpha=0.9)
    
    # True anomalies - 빨간색 원
    true_anomaly_mask = point_true > 0.5
    if np.any(true_anomaly_mask):
        axes[0].scatter(time_axis[true_anomaly_mask], data[true_anomaly_mask], 
                       color='red', s=120, label='True Anomalies', alpha=0.9, 
                       zorder=5, edgecolors='darkred', linewidth=2)
    
    # Predicted anomalies - 오렌지색 삼각형
    pred_anomaly_mask = point_pred > threshold
    if np.any(pred_anomaly_mask):
        axes[0].scatter(time_axis[pred_anomaly_mask], data[pred_anomaly_mask], 
                       color='orange', s=80, marker='^', label='Pred Anomalies', 
                       alpha=0.8, zorder=4, edgecolors='darkorange', linewidth=2)
    
    # Anomaly zone highlighting
    axes[0].fill_between(time_axis, data.min(), data.max(), 
                        where=(point_pred > threshold), alpha=0.25, color='red', 
                        label='Anomaly Zone')
    
    axes[0].set_title(f'{model_name} - Sample {sample_idx} ({anomaly_type})\n'
                     f'Series: True={series_true:.1f}, Pred={series_pred:.3f}', 
                     fontsize=24, fontweight='bold', pad=20)
    axes[0].set_ylabel('Signal Value', fontsize=20, fontweight='bold')
    axes[0].legend(fontsize=18, loc='upper right')
    axes[0].grid(True, alpha=0.4, linewidth=1.5)
    axes[0].tick_params(labelsize=18)
    
    # 2. Anomaly Score + True Labels + Threshold (2번째와 3번째 합침)
    # Twin axis for better visualization
    ax2_twin = axes[1].twinx()
    
    # Anomaly score - 녹색
    line1 = axes[1].plot(time_axis, point_pred, 'g-', linewidth=4, 
                        label='Anomaly Score', alpha=0.9)
    axes[1].axhline(y=threshold, color='red', linestyle='--', linewidth=4, 
                   label=f'Threshold ({threshold:.3f})', alpha=0.9)
    
    # 임계값을 넘는 영역 강조
    axes[1].fill_between(time_axis, point_pred, threshold, 
                        where=(point_pred > threshold), alpha=0.4, 
                        color='red', label='Anomaly Zone')
    
    # True labels - 보라색 (우측 축)
    line2 = ax2_twin.plot(time_axis, point_true, color='purple', linewidth=4, 
                         label='True Labels', alpha=0.9, linestyle='-.')
    
    axes[1].set_title('Anomaly Score vs True Labels with Threshold', 
                     fontsize=24, fontweight='bold', pad=20)
    axes[1].set_xlabel('Time Steps', fontsize=20, fontweight='bold')
    axes[1].set_ylabel('Anomaly Score', fontsize=20, fontweight='bold', color='green')
    ax2_twin.set_ylabel('True Labels (0=Normal, 1=Anomaly)', 
                       fontsize=20, fontweight='bold', color='purple')
    
    # Y축 범위 설정
    axes[1].set_ylim(-0.05, 1.05)
    ax2_twin.set_ylim(-0.05, 1.05)
    
    # 범례 합치기
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, 
                  fontsize=18, loc='upper left')
    
    axes[1].grid(True, alpha=0.4, linewidth=1.5)
    axes[1].tick_params(labelsize=18, colors='green')
    ax2_twin.tick_params(labelsize=18, colors='purple')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    filename = f'{save_dir}/sample_{sample_idx}_{anomaly_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
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

def train_single_model(model, model_name, train_loader, optimizer, device, epochs, scheduler=None):
    """향상된 단일 모델 훈련 - 웹검색 기반 최적화 적용"""
    print(f"🔥 Training {model_name.upper()} for {epochs} epochs...")
    
    model.train()
    total_loss = 0
    num_batches = 0
    best_loss = float('inf')
    patience = 0
    max_patience = 5  # Early stopping patience
    
    # 하이퍼파라미터 가져오기
    hp = MODEL_HP.get(model_name, MODEL_HP['fallback'])
    warmup_epochs = hp.get('warmup_epochs', 0)
    gradient_clip = hp.get('gradient_clip', 1.0)
    
    # 워밍업 스케줄러
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
    
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        
        # 워밍업 단계
        if epoch < warmup_epochs:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Warmup Epoch {epoch+1}/{warmup_epochs}, LR: {current_lr:.6f}")
        
        for batch_idx, (data, point_labels, series_labels, anomaly_types, info) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # 모델별 loss 계산 (기존 인터페이스 호환)
            if hasattr(model, 'compute_loss'):  # 실제 모델들
                # 모든 실제 모델은 기본 compute_loss 사용
                loss = model.compute_loss(data)
                    
            else:  # Fallback 모델
                pred_scores, reconstruction = model(data)
                
                # 향상된 손실 함수
                recon_loss = F.mse_loss(reconstruction, data)
                
                # 시리즈별 이상 스코어
                series_labels_tensor = series_labels.to(device)
                anomaly_loss = F.binary_cross_entropy_with_logits(pred_scores, series_labels_tensor)
                
                # Point-level 손실도 추가
                point_pred = torch.sigmoid(pred_scores).unsqueeze(1).expand(-1, data.shape[1])
                point_labels_tensor = point_labels.to(device)
                point_loss = F.binary_cross_entropy(point_pred, point_labels_tensor)
                
                # 가중 합계
                loss = 0.4 * recon_loss + 0.4 * anomaly_loss + 0.2 * point_loss
            
            loss.backward()
            
            # 향상된 Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}, LR: {current_lr:.6f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        # 스케줄러 업데이트
        if epoch < warmup_epochs and warmup_epochs > 0:
            warmup_scheduler.step()
        elif scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_epoch_loss)
            else:
                scheduler.step()
        
        # Early stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience = 0
        else:
            patience += 1
            
        print(f"   Epoch {epoch+1} completed. Avg Loss: {avg_epoch_loss:.6f}, Best: {best_loss:.6f}, Patience: {patience}")
        
        # Early stopping 체크
        if patience >= max_patience and epoch > epochs // 2:  # 최소 절반은 훈련
            print(f"   Early stopping triggered at epoch {epoch+1}")
            break
    
    avg_total_loss = total_loss / num_batches
    print(f"✅ {model_name.upper()} training completed. Avg Loss: {avg_total_loss:.6f}, Best Loss: {best_loss:.6f}")
    
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
    """개선된 전체 모델 성능 요약 플롯 - 사용자 요청사항 반영"""
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
    
    # 빨간색-파란색 컬러맵 생성 (높은값=빨간색, 낮은값=파란색)
    def get_color_from_value(values, cmap='RdBu_r'):
        """값에 따른 색상 생성 (높은값=빨간색, 낮은값=파란색)"""
        import matplotlib.cm as cm
        normalized = [(v - min(values)) / (max(values) - min(values)) if max(values) != min(values) else 0.5 for v in values]
        colors = [cm.get_cmap(cmap)(norm) for norm in normalized]
        return colors
    
    # 플롯 생성 (크기 증가)
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    metrics_data = [
        (series_acc, 'Series Accuracy', axes[0, 0]),
        (series_f1, 'Series F1-Score', axes[0, 1]),
        (series_auc, 'Series AUC', axes[0, 2]),
        (point_acc, 'Point Accuracy', axes[1, 0]),
        (point_f1, 'Point F1-Score', axes[1, 1]),
        (point_auc, 'Point AUC', axes[1, 2])
    ]
    
    for data, title, ax in metrics_data:
        # 현재 값 범위에 맞춰 Y축 스케일 조정 (0이 맨 아래에 고정되지 않게)
        min_val = min(data)
        max_val = max(data)
        margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
        y_min = max(0, min_val - margin)  # 0 이하로는 가지 않되, 여백 추가
        y_max = min(1, max_val + margin)  # 1 이상으로는 가지 않되, 여백 추가
        
        # 높은값=빨간색, 낮은값=파란색
        colors = get_color_from_value(data, 'RdBu_r')
        
        bars = ax.bar(models, data, alpha=0.8, color=colors, edgecolor='black', linewidth=2)
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        ax.set_ylim(y_min, y_max)  # 현재 값에 맞춘 스케일
        ax.tick_params(axis='x', rotation=45, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylabel('Score', fontsize=18, fontweight='bold')
        
        # 값 표시 (글자 크기 2배)
        for bar, value in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + margin*0.3,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        # 격자 개선
        ax.grid(True, alpha=0.3, linewidth=1)
    
    plt.suptitle('Model Performance Comparison', fontsize=28, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # final_performance_comparison.png
    filename1 = 'metrics/final_performance_comparison.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight', facecolor='white')
    
    # 추가로 performance_bars.png도 생성
    filename2 = 'metrics/performance_bars.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    
    # enhanced_quick_metrics.png 생성 (빨간색=높은값, 파란색=낮은값)
    create_enhanced_quick_metrics(all_results)
    
    print(f"📊 Summary metrics saved to {filename1} and {filename2}")
    return filename1

def create_enhanced_quick_metrics(all_results):
    """개선된 빠른 메트릭 히트맵 - 빨간색(높은값), 파란색(낮은값)"""
    if not MATPLOTLIB_AVAILABLE:
        return
        
    models = list(all_results.keys())
    
    # 메트릭 데이터 수집
    metrics_matrix = []
    metric_names = ['Series Acc', 'Series F1', 'Series AUC', 'Point Acc', 'Point F1', 'Point AUC']
    
    for model in models:
        row = [
            all_results[model]['series_metrics']['accuracy'],
            all_results[model]['series_metrics']['f1'],
            all_results[model]['series_metrics']['auc'],
            all_results[model]['point_metrics']['accuracy'],
            all_results[model]['point_metrics']['f1'],
            all_results[model]['point_metrics']['auc']
        ]
        metrics_matrix.append(row)
    
    metrics_matrix = np.array(metrics_matrix)
    
    # 히트맵 생성
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 빨간색=높은값, 파란색=낮은값 (RdBu_r 컬러맵)
    im = ax.imshow(metrics_matrix, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
    
    # 축 설정
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metric_names, fontsize=18, fontweight='bold')
    ax.set_yticklabels(models, fontsize=18, fontweight='bold')
    
    # 제목
    ax.set_title('Model Performance Heatmap\n(Red=High, Blue=Low)', 
                fontsize=24, fontweight='bold', pad=30)
    
    # 값 표시
    for i in range(len(models)):
        for j in range(len(metric_names)):
            value = metrics_matrix[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                   color=text_color, fontsize=16, fontweight='bold')
    
    # 컬러바
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Performance Score', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    
    filename = 'metrics/enhanced_quick_metrics.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Enhanced quick metrics saved to {filename}")
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
            # 향상된 optimizer 및 scheduler 생성
            hp = MODEL_HP.get(model_name, MODEL_HP['fallback'])
            
            # 모델별 최적화된 optimizer
            if model_name in ['tracegpt', 'patchtrad']:
                # Transformer 모델들은 AdamW + 웜업 + 코사인 스케줄러
                optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=hp['lr'], 
                    weight_decay=hp.get('weight_decay', 1e-4),
                    betas=(0.9, 0.98),  # Transformer에 최적화
                    eps=1e-6
                )
                
                # 코사인 어닐링 스케줄러
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=hp['epochs'], 
                    eta_min=hp['lr'] * 0.01
                )
                
            elif model_name == 'carla':
                # CARLA는 contrastive learning에 최적화
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=hp['lr'],
                    weight_decay=hp.get('weight_decay', 1e-4),
                    betas=(0.9, 0.999)
                )
                
                # ReduceLROnPlateau 스케줄러 (성능 기반)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=3,
                    min_lr=hp['lr'] * 0.001
                )
                
            elif model_name == 'prodiffad':
                # Diffusion 모델은 더 안정적인 학습률
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=hp['lr'],
                    weight_decay=hp.get('weight_decay', 1e-5),
                    betas=(0.9, 0.999)
                )
                
                # 단계별 감소 스케줄러
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=hp['epochs'] // 3,
                    gamma=0.5
                )
                
            else:  # fallback
                # 기본 설정
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=hp['lr'],
                    weight_decay=hp.get('weight_decay', 1e-4)
                )
                
                scheduler_type = hp.get('scheduler', 'plateau')
                if scheduler_type == 'step':
                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=hp.get('step_size', 10),
                        gamma=hp.get('gamma', 0.5)
                    )
                else:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=0.5,
                        patience=3
                    )
            
            print(f"🔧 Optimizer: {type(optimizer).__name__}, Scheduler: {type(scheduler).__name__}")
            
            # 향상된 훈련
            train_loss = train_single_model(
                model, model_name, train_loader, optimizer, 
                GPU_CONFIG['device'], hp['epochs'], scheduler
            )
            
            # 모델 저장 (state_dict + optimizer + scheduler)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'epoch': hp['epochs'],
                'loss': train_loss,
                'hyperparameters': hp
            }
            torch.save(checkpoint, f'pre_trained/{model_name}_final.pth')
            print(f"💾 Model checkpoint saved to pre_trained/{model_name}_final.pth")
        
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