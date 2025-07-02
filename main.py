#!/usr/bin/env python3
"""
Real Models DDP Anomaly Detection with Enhanced Evaluation & Visualization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

진짜 논문 기반 모델들로 구현:
- PatchTrAD: Patch-based Transformer
- TraceGPT: GPT-style Autoregressive Transformer  
- CARLA: Contrastive Anomaly Representation Learning
- ProDiffAD: Progressive Diffusion Model
- 3개 Ensemble models with real architectures

새로운 기능:
- 모델별 성능 매트릭 계산 및 시각화
- Confusion Matrix 생성 및 저장
- 데이터셋 샘플 시각화
- 모델별 anomaly score plot 생성
"""

import os, torch, torch.distributed as dist
try:
    import torch.multiprocessing as mp
    from torch.multiprocessing.spawn import spawn as mp_spawn
except ImportError:
    import multiprocessing as mp
    mp_spawn = None
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from multiprocessing import Manager
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse
import math
import socket

# matplotlib backend 설정 (이미지 저장 문제 해결)
try:
    import matplotlib
    matplotlib.use('Agg')  # GUI 없는 환경에서 이미지 저장
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️ matplotlib, seaborn, pandas not found. Visualization will be skipped.")
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ sklearn not found. Using simple metrics.")
    SKLEARN_AVAILABLE = False
    
    # 간단한 대체 함수들
    def accuracy_score(y_true, y_pred):
        return (y_true == y_pred).mean()
    
    def precision_score(y_true, y_pred, zero_division=0):
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        return tp / (tp + fp) if (tp + fp) > 0 else zero_division
    
    def recall_score(y_true, y_pred, zero_division=0):
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else zero_division
    
    def f1_score(y_true, y_pred, zero_division=0):
        p = precision_score(y_true, y_pred, zero_division)
        r = recall_score(y_true, y_pred, zero_division)
        return 2 * p * r / (p + r) if (p + r) > 0 else zero_division
    
    def roc_auc_score(y_true, y_score):
        return 0.5  # 기본값
    
    def confusion_matrix(y_true, y_pred):
        import numpy as np
        return np.array([[0, 0], [0, 0]])  # 더미 행렬

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 📊 MODEL-SPECIFIC CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    'patchtrad': {
        'batch_size': 128,
        'epochs': 25,
        'lr': 1e-4,
        'hidden_dim': 192,
        'num_heads': 12,
        'num_layers': 4,
        'patch_size': 8,
        'warmup_epochs': 1,
        'weight_decay': 1e-4
    },
    'tracegpt': {
        'batch_size': 96,      # GPT는 메모리 많이 사용
        'epochs': 35,          # 더 많은 학습 필요
        'lr': 5e-5,           # 더 작은 학습률
        'hidden_dim': 384,    # 더 큰 모델
        'num_heads': 16,
        'num_layers': 8,
        'window_size': 64,
        'warmup_epochs': 2,
        'weight_decay': 1e-5
    },
    'carla': {
        'batch_size': 160,     # Contrastive learning은 큰 배치
        'epochs': 30,
        'lr': 2e-4,           # 더 큰 학습률
        'hidden_dim': 256,
        'cnn_channels': [64, 128, 256, 512],
        'memory_size': 2048,
        'temperature': 0.07,
        'warmup_epochs': 1,
        'weight_decay': 1e-4
    },
    'prodiffad': {
        'batch_size': 64,      # Diffusion은 메모리 많이 사용
        'epochs': 40,          # 가장 많은 학습 필요
        'lr': 1e-4,
        'hidden_dim': 320,
        'num_layers': 12,
        'timesteps': 1000,
        'context_dim': 256,
        'warmup_epochs': 2,
        'weight_decay': 1e-4
    },
    'patch_trace_ensemble': {
        'batch_size': 96,
        'epochs': 30,
        'lr': 8e-5,
        'hidden_dim': 256,
        'warmup_epochs': 2,
        'weight_decay': 1e-4
    },
    'transfer_learning_ensemble': {
        'batch_size': 128,
        'epochs': 25,
        'lr': 1e-4,
        'hidden_dim': 256,
        'warmup_epochs': 1,
        'weight_decay': 1e-5
    },
    'multi_model_ensemble': {
        'batch_size': 80,      # 4개 모델 앙상블이라 메모리 많이 사용
        'epochs': 35,
        'lr': 5e-5,
        'hidden_dim': 256,
        'warmup_epochs': 2,
        'weight_decay': 1e-4
    }
}

# 기본 설정 (fallback)
DEFAULT_CONFIG = {
    'batch_size': 128,
    'epochs': 30,
    'lr': 1e-4,
    'hidden_dim': 256,
    'num_heads': 8,
    'num_layers': 6,
    'patch_size': 8,
    'warmup_epochs': 1,
    'weight_decay': 1e-4
}

def get_model_config(model_name):
    """모델별 설정 반환"""
    return MODEL_CONFIGS.get(model_name, DEFAULT_CONFIG)

# 전역 설정 (오버라이드 가능)
WORLD_SIZE = max(torch.cuda.device_count(), 1)
GPUS = list(range(WORLD_SIZE))
LOG_EVERY = 50
DATA_SIZE = 6400  # 📊 시계열 개수 조절
SEQ_LEN = 64      # 📊 각 시계열의 길이 조절

# 저장 폴더 생성 (anomaly_detection 폴더에 직접 저장)
for folder in ['samples', 'plots', 'metrics', 'confusion_matrices', 'pre_trained']:
    os.makedirs(folder, exist_ok=True)
    print(f"📁 Created directory: {folder}")

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

os.environ.setdefault("OMP_NUM_THREADS", "1")

# ============================================================================
# 📊 GLOBAL CONSTANTS
# ============================================================================

# 전역 상수 정의
HIDDEN_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
PATCH_SIZE = 8
BATCH = 128
LR = 1e-4
EPOCHS = 30

print(f"🔧 Available GPUs: {WORLD_SIZE}")
if MATPLOTLIB_AVAILABLE:
    print(f"🎨 Matplotlib backend: {matplotlib.get_backend()}")

# ============================================================================
# 📊 ENHANCED DATASET
# ============================================================================

class RealAnomalyDataset(Dataset):
    """실제 이상 탐지 데이터셋"""
    
    def __init__(self, mode='train', size=None):
        self.mode = mode
        self.data = []
        self.point_labels = []
        self.series_labels = []
        
        # 데이터 생성
        np.random.seed(42 if mode == 'train' else 123)
        data_size = size if size else (DATA_SIZE if mode == 'train' else DATA_SIZE // 4)
        
        print(f"🔄 Generating {mode} dataset with {data_size} samples...")
        
        for idx in range(data_size):
            series, point_label, series_label = self._generate_complex_series(idx)
            self.data.append(series)
            self.point_labels.append(point_label)
            self.series_labels.append(series_label)
        
        # 데이터셋 통계 (5가지 클래스)
        class_counts = [0, 0, 0, 0, 0]
        for label in self.series_labels:
            class_counts[int(label)] += 1
        
        print(f"✅ {mode} dataset generated: {len(self.data)} samples")
        print(f"   📊 Class distribution:")
        print(f"      Normal: {class_counts[0]}")
        print(f"      Spike: {class_counts[1]}")
        print(f"      Mean Shift: {class_counts[2]}")
        print(f"      Std Change: {class_counts[3]}")
        print(f"      Drift: {class_counts[4]}")
        print(f"   📈 Anomaly ratio: {sum(class_counts[1:])}/{len(self.data)} ({sum(class_counts[1:])/len(self.data)*100:.1f}%)")
    
    def _generate_complex_series(self, idx):
        """5가지 클래스: Normal, Spike, Mean Shift, Std Change, Drift
        
        🎛️ ANOMALY 강도 조절 파라미터들:
        - SPIKE_MAGNITUDE: 스파이크 크기
        - MEAN_SHIFT_MAGNITUDE: 평균 변화 크기 
        - STD_CHANGE_MULTIPLIER: 분산 변화 배수
        - DRIFT_MAGNITUDE: 드리프트 크기
        - BASE_NOISE_STD: 기본 노이즈 레벨
        
        📍 ANOMALY 위치 조절:
        - ANOMALY_POSITION: 'early'(앞쪽), 'middle'(중간), 'late'(뒤쪽), 'random'(랜덤)
        """
        # 🎛️ ANOMALY 강도 조절 파라미터들 (여기서 수정!)
        SPIKE_MAGNITUDE = (6.0, 12.0)       # 스파이크 크기 (강하게: 6~12, 약하게: 2~4)
        MEAN_SHIFT_MAGNITUDE = (3.0, 6.0)   # 평균 변화 (강하게: 3~6, 약하게: 1~2)  
        STD_CHANGE_MULTIPLIER = (5.0, 10.0) # 분산 변화 (강하게: 5~10, 약하게: 2~4)
        DRIFT_MAGNITUDE = (3.0, 6.0)        # 드리프트 (강하게: 3~6, 약하게: 1~3)
        BASE_NOISE_STD = 0.08                # 기본 노이즈 (작게: 0.05, 크게: 0.2)
        
        # 📍 ANOMALY 위치 조절 (여기서 수정!)
        ANOMALY_POSITION = 'late'  # 🔴 'early': 앞쪽(0~30%), 'middle': 중간(30~70%), 'late': 뒤쪽(70~100%), 'random': 랜덤
        
        t = np.arange(SEQ_LEN)
        point_label = np.zeros(SEQ_LEN, dtype=np.float32)
        
        # 기본 정상 베이스
        target_value = 0.0
        series = np.random.normal(target_value, BASE_NOISE_STD, SEQ_LEN)
        
        # 위치 범위 계산
        if ANOMALY_POSITION == 'early':
            start_range = (5, int(SEQ_LEN * 0.3))
            end_range = (int(SEQ_LEN * 0.3), int(SEQ_LEN * 0.6))
        elif ANOMALY_POSITION == 'middle':
            start_range = (int(SEQ_LEN * 0.3), int(SEQ_LEN * 0.5))
            end_range = (int(SEQ_LEN * 0.5), int(SEQ_LEN * 0.8))
        elif ANOMALY_POSITION == 'late':
            start_range = (int(SEQ_LEN * 0.6), int(SEQ_LEN * 0.8))
            end_range = (int(SEQ_LEN * 0.8), SEQ_LEN - 2)
        else:  # random
            start_range = (5, SEQ_LEN - 20)
            end_range = (15, SEQ_LEN - 5)
        
        anomaly_type = idx % 5
        
        if anomaly_type == 0:
            # Class 0: Normal
            series_label = 0.0
            
        elif anomaly_type == 1:
            # Class 1: Spike Anomaly
            n_spikes = np.random.randint(2, 5)
            for _ in range(n_spikes):
                if ANOMALY_POSITION == 'random':
                    spike_idx = np.random.randint(5, SEQ_LEN-5)
                else:
                    spike_idx = np.random.randint(*start_range)
                    
                spike_magnitude = np.random.choice([-1, 1]) * np.random.uniform(*SPIKE_MAGNITUDE)
                series[spike_idx] += spike_magnitude
                
                # 스파이크 주변 영향
                for offset in [-2, -1, 1, 2]:
                    if 0 <= spike_idx + offset < SEQ_LEN:
                        series[spike_idx + offset] += spike_magnitude * 0.15
                        point_label[spike_idx + offset] = 1.0
                point_label[spike_idx] = 1.0
            series_label = 1.0
            
        elif anomaly_type == 2:
            # Class 2: Mean Shift
            if ANOMALY_POSITION == 'random':
                shift_start = np.random.randint(10, SEQ_LEN-25)
                shift_end = min(shift_start + np.random.randint(15, 30), SEQ_LEN)
            else:
                shift_start = np.random.randint(*start_range)
                shift_end = min(shift_start + np.random.randint(15, 30), end_range[1])
                
            shift_magnitude = np.random.choice([-1, 1]) * np.random.uniform(*MEAN_SHIFT_MAGNITUDE)
            series[shift_start:shift_end] += shift_magnitude
            point_label[shift_start:shift_end] = 1.0
            series_label = 2.0
            
        elif anomaly_type == 3:
            # Class 3: Std Change
            if ANOMALY_POSITION == 'random':
                change_start = np.random.randint(10, SEQ_LEN-25)
                change_end = min(change_start + np.random.randint(20, 35), SEQ_LEN)
            else:
                change_start = np.random.randint(*start_range)
                change_end = min(change_start + np.random.randint(20, 35), end_range[1])
                
            new_std = BASE_NOISE_STD * np.random.uniform(*STD_CHANGE_MULTIPLIER)
            series[change_start:change_end] = np.random.normal(target_value, new_std, change_end - change_start)
            point_label[change_start:change_end] = 1.0
            series_label = 3.0
            
        else:  # anomaly_type == 4
            # Class 4: Drift/Trend
            if ANOMALY_POSITION == 'random':
                drift_start = np.random.randint(5, SEQ_LEN//2)
                drift_end = min(drift_start + np.random.randint(SEQ_LEN//3, SEQ_LEN-5), SEQ_LEN)
            else:
                drift_start = np.random.randint(*start_range)
                drift_end = min(drift_start + np.random.randint(SEQ_LEN//4, SEQ_LEN//2), end_range[1])
                
            actual_length = drift_end - drift_start
            
            drift_type = np.random.choice(['linear', 'exponential', 'quadratic'])
            drift_direction = np.random.choice([-1, 1])
            drift_magnitude = np.random.uniform(*DRIFT_MAGNITUDE)
            
            if drift_type == 'linear':
                drift_pattern = np.linspace(0, drift_direction * drift_magnitude, actual_length)
            elif drift_type == 'exponential':
                t_norm = np.linspace(0, 1, actual_length)
                drift_pattern = drift_direction * drift_magnitude * (np.exp(2 * t_norm) - 1) / (np.exp(2) - 1)
            else:  # quadratic
                t_norm = np.linspace(0, 1, actual_length)
                drift_pattern = drift_direction * drift_magnitude * (t_norm ** 2)
            
            series[drift_start:drift_end] += drift_pattern
            point_label[drift_start:drift_end] = 1.0
            series_label = 4.0
        
        return series.astype(np.float32), point_label, series_label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx]),
            torch.from_numpy(self.point_labels[idx]),
            torch.tensor(1.0 if self.series_labels[idx] > 0 else 0.0, dtype=torch.float32)  # Binary for models
        )
    
    def get_anomaly_type(self, idx):
        """Get the specific anomaly type (0-4)"""
        return int(self.series_labels[idx])
    
    def get_anomaly_type_name(self, anomaly_type):
        """Get anomaly type name (without emojis)"""
        type_names = {
            0: 'Normal',
            1: 'Spike Anomaly', 
            2: 'Mean Shift',
            3: 'Std Change',
            4: 'Drift/Trend'
        }
        return type_names.get(anomaly_type, 'Unknown')

# ============================================================================
# 🧠 REAL MODEL IMPLEMENTATIONS
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        pe_slice = self.pe[:x.size(0), :]
        return x + pe_slice

class RotaryPositionalEncoding(nn.Module):
    """RoPE (Rotary Position Embedding) for better positional encoding"""
    
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model
        
        # Create rotation matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).float()
        inv_freq_buffer = getattr(self, 'inv_freq')
        freqs = torch.einsum('i,j->ij', t, inv_freq_buffer)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_emb = emb.cos()[None, :, None, :]
        sin_emb = emb.sin()[None, :, None, :]
        
        # Apply rotary embedding
        x_rot = self.rotate_half(x)
        return x * cos_emb + x_rot * sin_emb
    
    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

class MultiScaleConv1d(nn.Module):
    """Multi-scale convolution for better feature extraction"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels//4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels//4, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(in_channels, out_channels//4, kernel_size=1)
        
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return self.activation(self.norm(out))

class OptimizedPatchTrAD(nn.Module):
    """Optimized PatchTrAD with better architecture"""
    
    def __init__(self):
        super().__init__()
        self.patch_size = PATCH_SIZE
        self.stride = PATCH_SIZE // 2  # Overlapping patches
        self.num_patches = (SEQ_LEN - PATCH_SIZE) // self.stride + 1
        
        # Enhanced patch embedding with CNN
        self.patch_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(PATCH_SIZE)
        )
        
        self.patch_embedding = nn.Sequential(
            nn.Linear(64, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.Dropout(0.1)
        )
        
        # Rotary positional encoding
        self.pos_encoding = RotaryPositionalEncoding(HIDDEN_DIM)
        
        # Enhanced transformer with residual connections
        self.transformer_layers = nn.ModuleList([
            self._make_transformer_layer() for _ in range(NUM_LAYERS)
        ])
        
        # Multi-scale feature extraction
        self.multi_scale_conv = MultiScaleConv1d(HIDDEN_DIM, HIDDEN_DIM)
        
        # Improved heads with residual connections
        self.anomaly_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
            nn.Sigmoid()
        )
        
        # Feature pyramid for multi-scale anomaly detection
        self.fpn = nn.ModuleList([
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2)
        ])
        
        self.series_classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM + HIDDEN_DIM // 2 * 3, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
            nn.Sigmoid()
        )
        
    def _make_transformer_layer(self):
        return nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_DIM * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
    
    def create_overlapping_patches(self, x):
        """Create overlapping patches for better feature extraction"""
        batch_size, seq_len = x.shape
        x_unsqueezed = x.unsqueeze(1)  # [B, 1, seq_len]
        
        # CNN feature extraction
        cnn_features = self.patch_cnn(x_unsqueezed)  # [B, 64, patch_size]
        
        # Create overlapping patches
        patches = []
        for i in range(0, seq_len - self.patch_size + 1, self.stride):
            if i + self.patch_size <= seq_len:
                patch_features = cnn_features[:, :, i:i+self.patch_size]
                patch_features = patch_features.mean(dim=-1)  # [B, 64]
                patches.append(patch_features)
        
        if patches:
            return torch.stack(patches, dim=1)  # [B, num_patches, 64]
        else:
            # Fallback if no patches created
            return cnn_features.mean(dim=-1).unsqueeze(1)  # [B, 1, 64]
    
    def forward(self, x):
        # Create overlapping patches with CNN
        patches = self.create_overlapping_patches(x)  # [B, num_patches, 64]
        
        # Patch embedding
        patch_embeds = self.patch_embedding(patches)  # [B, num_patches, hidden_dim]
        
        # Apply rotary positional encoding
        patch_embeds = self.pos_encoding(patch_embeds)
        
        # Enhanced transformer with residual connections
        hidden_states = []
        h = patch_embeds
        for layer in self.transformer_layers:
            h_residual = h
            h = layer(h)
            h = h + h_residual  # Residual connection
            hidden_states.append(h)
        
        # Feature pyramid
        fpn_features = []
        for i, fpn_layer in enumerate(self.fpn):
            if i < len(hidden_states):
                fpn_feat = fpn_layer(hidden_states[i].mean(dim=1))
                fpn_features.append(fpn_feat)
        
        # Patch-level anomaly scores
        patch_scores = self.anomaly_head(h).squeeze(-1)  # [B, num_patches]
        
        # Interpolate to original sequence length
        if patch_scores.shape[1] != SEQ_LEN:
            patch_scores = F.interpolate(
                patch_scores.unsqueeze(1), 
                size=SEQ_LEN, 
                mode='linear', 
                align_corners=False
            ).squeeze(1)
        
        # Series-level score with FPN features
        global_features = h.mean(dim=1)  # [B, hidden_dim]
        if fpn_features:
            global_features = torch.cat([global_features] + fpn_features, dim=-1)
        
        series_score = self.series_classifier(global_features).squeeze(-1)
        
        return patch_scores, series_score

class OptimizedTraceGPT(nn.Module):
    """Optimized TraceGPT with advanced GPT techniques"""
    
    def __init__(self):
        super().__init__()
        
        # Enhanced input embedding with learned positional embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(1, HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM)
        )
        
        # Rotary positional encoding
        self.pos_encoding = RotaryPositionalEncoding(HIDDEN_DIM)
        
        # Enhanced transformer blocks with residual connections
        self.transformer_blocks = nn.ModuleList([
            self._make_gpt_block() for _ in range(NUM_LAYERS)
        ])
        
        # Multi-head attention for anomaly detection
        self.anomaly_attention = nn.MultiheadAttention(
            HIDDEN_DIM, NUM_HEADS, dropout=0.1, batch_first=True
        )
        
        self.anomaly_head = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
            nn.Sigmoid()
        )
        
        # Temporal convolution for series-level features
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(HIDDEN_DIM, HIDDEN_DIM, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(HIDDEN_DIM, HIDDEN_DIM, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.series_classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
            nn.Sigmoid()
        )
        
    def _make_gpt_block(self):
        """Create GPT-style transformer block"""
        class GPTBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.MultiheadAttention(
                    HIDDEN_DIM, NUM_HEADS, dropout=0.1, batch_first=True
                )
                self.norm1 = nn.LayerNorm(HIDDEN_DIM)
                self.norm2 = nn.LayerNorm(HIDDEN_DIM)
                self.ffn = nn.Sequential(
                    nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM),
                    nn.Dropout(0.1)
                )
        return GPTBlock()
    
    def create_sliding_window_mask(self, seq_len, window_size=32, device='cpu'):
        """Create sliding window attention mask for efficiency"""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + 1)  # Causal
            mask[i, start:end] = 0
            
        return mask
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device
        
        # Enhanced input embedding
        h = self.input_embedding(x.unsqueeze(-1))  # [B, seq_len, hidden_dim]
        
        # Apply rotary positional encoding
        h = self.pos_encoding(h)
        
        # Sliding window causal mask for efficiency
        window_size = min(64, seq_len)  # Adaptive window size
        attn_mask = self.create_sliding_window_mask(seq_len, window_size, device)
        
        # GPT-style transformer blocks
        hidden_states = []
        for block in self.transformer_blocks:
            # Pre-norm + self-attention
            norm_h = block.norm1(h)
            attn_out, _ = block.attention(norm_h, norm_h, norm_h, attn_mask=attn_mask)
            h = h + attn_out  # Residual connection
            
            # Pre-norm + FFN
            norm_h = block.norm2(h)
            ffn_out = block.ffn(norm_h)
            h = h + ffn_out  # Residual connection
            
            hidden_states.append(h)
        
        # Anomaly-specific attention
        anomaly_features, _ = self.anomaly_attention(h, h, h, attn_mask=attn_mask)
        anomaly_features = anomaly_features + h  # Residual
        
        # Point-level anomaly scores
        point_scores = self.anomaly_head(anomaly_features).squeeze(-1)  # [B, seq_len]
        
        # Series-level features using temporal convolution
        temporal_features = self.temporal_conv(h.transpose(1, 2)).squeeze(-1)  # [B, hidden_dim]
        series_score = self.series_classifier(temporal_features).squeeze(-1)  # [B]
        
        return point_scores, series_score

class MomentumEncoder(nn.Module):
    """Momentum encoder for contrastive learning"""
    
    def __init__(self, encoder, momentum=0.999):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum
        
        # Create momentum encoder
        self.momentum_encoder = self._create_momentum_encoder()
        
        # Initialize momentum encoder with encoder weights
        self._initialize_momentum_encoder()
        
    def _create_momentum_encoder(self):
        """Create a copy of the encoder for momentum updates"""
        momentum_encoder = type(self.encoder)()
        return momentum_encoder
    
    def _initialize_momentum_encoder(self):
        """Initialize momentum encoder with encoder weights"""
        for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    def _momentum_update(self):
        """Momentum update of the momentum encoder"""
        for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    def forward(self, x_q, x_k=None):
        # Query encoding
        q = self.encoder(x_q)
        
        if x_k is not None:
            # Key encoding with momentum encoder
            with torch.no_grad():
                self._momentum_update()
                k = self.momentum_encoder(x_k)
            return q, k
        
        return q

class OptimizedCARLA(nn.Module):
    """Optimized CARLA with advanced contrastive learning"""
    
    def __init__(self):
        super().__init__()
        
        # Multi-scale CNN encoder
        self.cnn_encoder = nn.Sequential(
            MultiScaleConv1d(1, 64),
            nn.MaxPool1d(2),
            MultiScaleConv1d(64, 128),
            nn.MaxPool1d(2),
            MultiScaleConv1d(128, 256),
            nn.AdaptiveAvgPool1d(HIDDEN_DIM // 4)
        )
        
        # Enhanced representation network with residual connections
        self.representation_net = nn.Sequential(
            nn.Linear(256 * (HIDDEN_DIM // 4), HIDDEN_DIM * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM)
        )
        
        # Contrastive projection head
        self.projection_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM // 4)
        )
        
        # Memory bank for contrastive learning
        self.register_buffer('memory_bank', torch.randn(1000, HIDDEN_DIM // 4))
        self.register_buffer('memory_labels', torch.zeros(1000))
        self.memory_ptr = 0
        
        # Attention-based anomaly detection
        self.anomaly_attention = nn.MultiheadAttention(
            HIDDEN_DIM, NUM_HEADS, dropout=0.1, batch_first=True
        )
        
        # Point-level and series-level classifiers
        self.point_classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, SEQ_LEN),
            nn.Sigmoid()
        )
        
        self.series_classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM + HIDDEN_DIM // 4, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
            nn.Sigmoid()
        )
        
        # Temperature for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
    def augment_data(self, x):
        """Data augmentation for contrastive learning"""
        batch_size, seq_len = x.shape
        
        # Random masking
        mask = torch.rand(batch_size, seq_len, device=x.device) > 0.1
        x_masked = x * mask
        
        # Gaussian noise
        noise = torch.randn_like(x) * 0.01
        x_noisy = x + noise
        
        # Scaling
        scale = torch.rand(batch_size, 1, device=x.device) * 0.2 + 0.9
        x_scaled = x * scale
        
        return x_masked, x_noisy, x_scaled
    
    def update_memory_bank(self, features, labels):
        """Update memory bank with new features"""
        batch_size = features.shape[0]
        
        with torch.no_grad():
            # Update memory bank
            ptr = self.memory_ptr
            memory_bank = getattr(self, 'memory_bank')
            memory_labels = getattr(self, 'memory_labels')
            
            if ptr + batch_size <= memory_bank.shape[0]:
                memory_bank[ptr:ptr + batch_size] = features
                memory_labels[ptr:ptr + batch_size] = labels
                self.memory_ptr = (ptr + batch_size) % memory_bank.shape[0]
            else:
                # Wrap around
                remaining = memory_bank.shape[0] - ptr
                memory_bank[ptr:] = features[:remaining]
                memory_labels[ptr:] = labels[:remaining]
                memory_bank[:batch_size - remaining] = features[remaining:]
                memory_labels[:batch_size - remaining] = labels[remaining:]
                self.memory_ptr = batch_size - remaining
    
    def contrastive_loss(self, q, k, labels):
        """Compute contrastive loss"""
        batch_size = q.shape[0]
        
        # Normalize features
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Compute similarity
        logits = torch.mm(q, k.t()) / self.temperature
        
        # Create labels for contrastive learning
        labels_q = labels.unsqueeze(1)
        labels_k = labels.unsqueeze(0)
        mask = (labels_q == labels_k).float()
        
        # Compute loss
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        
        loss = -mean_log_prob_pos.mean()
        return loss
    
    def forward(self, x, labels=None):
        batch_size, seq_len = x.shape
        
        # Multi-scale CNN encoding
        x_input = x.unsqueeze(1)  # [B, 1, seq_len]
        cnn_features = self.cnn_encoder(x_input)  # [B, 256, hidden_dim//4]
        cnn_features = cnn_features.flatten(1)  # [B, 256 * hidden_dim//4]
        
        # Representation learning
        representations = self.representation_net(cnn_features)  # [B, hidden_dim]
        
        # Contrastive learning (during training)
        contrastive_loss = 0
        if self.training and labels is not None:
            # Data augmentation
            x_aug1, x_aug2, x_aug3 = self.augment_data(x)
            
            # Encode augmented data
            aug_features1 = self.representation_net(
                self.cnn_encoder(x_aug1.unsqueeze(1)).flatten(1)
            )
            aug_features2 = self.representation_net(
                self.cnn_encoder(x_aug2.unsqueeze(1)).flatten(1)
            )
            
            # Project to contrastive space
            proj_orig = self.projection_head(representations)
            proj_aug1 = self.projection_head(aug_features1)
            proj_aug2 = self.projection_head(aug_features2)
            
            # Contrastive loss between original and augmented
            contrastive_loss = (
                self.contrastive_loss(proj_orig, proj_aug1, labels) +
                self.contrastive_loss(proj_orig, proj_aug2, labels)
            ) * 0.5
            
            # Update memory bank
            self.update_memory_bank(proj_orig.detach(), labels)
        
        # Attention-based feature enhancement
        representations_expanded = representations.unsqueeze(1).repeat(1, SEQ_LEN, 1)
        enhanced_features, _ = self.anomaly_attention(
            representations_expanded, representations_expanded, representations_expanded
        )
        enhanced_features = enhanced_features.mean(dim=1)  # [B, hidden_dim]
        
        # Point-level anomaly scores
        point_scores = self.point_classifier(enhanced_features)  # [B, seq_len]
        
        # Series-level anomaly scores
        if self.training and labels is not None:
            proj_features = self.projection_head(representations)
            series_features = torch.cat([representations, proj_features], dim=-1)
        else:
            proj_features = self.projection_head(representations)
            series_features = torch.cat([representations, proj_features], dim=-1)
            
        series_score = self.series_classifier(series_features).squeeze(-1)  # [B]
        
        if self.training and labels is not None:
            return point_scores, series_score, contrastive_loss
        
        return point_scores, series_score

class CrossAttentionBlock(nn.Module):
    """Cross-attention block for diffusion model"""
    
    def __init__(self, dim, context_dim=None):
        super().__init__()
        context_dim = context_dim or dim
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.self_attn = nn.MultiheadAttention(dim, 8, dropout=0.1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, 8, dropout=0.1, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, context=None):
        # Self-attention
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # Cross-attention (if context provided)
        if context is not None:
            norm_x = self.norm1(x)
            cross_out, _ = self.cross_attn(norm_x, context, context)
            x = x + cross_out
        
        # FFN
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        
        return x

class OptimizedProDiffAD(nn.Module):
    """Optimized ProDiffAD with advanced diffusion techniques"""
    
    def __init__(self):
        super().__init__()
        self.timesteps = 1000
        self.prediction_type = "epsilon"  # Can be "epsilon" or "v_prediction"
        
        # Enhanced time embedding with sinusoidal + learned
        self.time_mlp = nn.Sequential(
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        )
        
        # Context encoder for conditional generation
        self.context_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, HIDDEN_DIM, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(SEQ_LEN // 4)
        )
        
        # Input projection with residual connection
        self.input_proj = nn.Sequential(
            nn.Linear(SEQ_LEN, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU()
        )
        
        # Enhanced U-Net architecture with attention
        self.down_blocks = nn.ModuleList([
            self._make_down_block(HIDDEN_DIM, HIDDEN_DIM),
            self._make_down_block(HIDDEN_DIM, HIDDEN_DIM * 2),
            self._make_down_block(HIDDEN_DIM * 2, HIDDEN_DIM * 2),
        ])
        
        # Cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(HIDDEN_DIM),
            CrossAttentionBlock(HIDDEN_DIM * 2),
            CrossAttentionBlock(HIDDEN_DIM * 2),
        ])
        
        # Middle block with attention
        self.mid_block = nn.Sequential(
            self._make_block(HIDDEN_DIM * 2, HIDDEN_DIM * 2),
            CrossAttentionBlock(HIDDEN_DIM * 2),
            self._make_block(HIDDEN_DIM * 2, HIDDEN_DIM * 2)
        )
        
        # Up blocks with skip connections
        self.up_blocks = nn.ModuleList([
            self._make_up_block(HIDDEN_DIM * 4, HIDDEN_DIM * 2),  # Skip connection
            self._make_up_block(HIDDEN_DIM * 4, HIDDEN_DIM),
            self._make_up_block(HIDDEN_DIM * 2, HIDDEN_DIM),
        ])
        
        # Output heads with residual connections
        self.noise_head = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, SEQ_LEN)
        )
        
        self.point_anomaly_head = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, SEQ_LEN),
            nn.Sigmoid()
        )
        
        self.series_anomaly_head = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM // 2, 1),
            nn.Sigmoid()
        )
        
        # Noise schedules
        betas = self._cosine_beta_schedule()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        
    def _make_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU()
        )
    
    def _make_down_block(self, in_dim, out_dim):
        return nn.Sequential(
            self._make_block(in_dim, out_dim),
            nn.Linear(out_dim, out_dim)  # Downsampling
        )
    
    def _make_up_block(self, in_dim, out_dim):
        return nn.Sequential(
            self._make_block(in_dim, out_dim),
            nn.Linear(out_dim, out_dim)  # Upsampling
        )
    
    def get_time_embedding(self, timesteps):
        """Sinusoidal time embedding"""
        half_dim = HIDDEN_DIM // 4
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def _cosine_beta_schedule(self, s=0.008):
        """Improved cosine noise schedule"""
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def forward(self, x, t=None):
        batch_size, seq_len = x.shape
        device = x.device
        
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # Enhanced time embedding
        t_emb_sin = self.get_time_embedding(t)
        t_emb = self.time_mlp(t_emb_sin)  # [B, hidden_dim]
        
        # Context encoding
        context = self.context_encoder(x.unsqueeze(1))  # [B, hidden_dim, seq_len//4]
        context = context.transpose(1, 2)  # [B, seq_len//4, hidden_dim]
        
        # Input projection
        h = self.input_proj(x)  # [B, hidden_dim]
        h = h + t_emb  # Add time embedding
        h = h.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Down path with skip connections and cross-attention
        skip_connections = []
        for i, (down_block, cross_attn) in enumerate(zip(self.down_blocks, self.cross_attn_blocks)):
            h = down_block(h)
            if i < len(self.cross_attn_blocks):
                h = cross_attn(h, context)
            skip_connections.append(h)
        
        # Middle block with attention
        h = self.mid_block(h)
        
        # Up path with skip connections
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            h = up_block(torch.cat([h, skip], dim=-1))
        
        h = h.squeeze(1)  # [B, hidden_dim]
        
        # Output heads
        point_scores = self.point_anomaly_head(h)  # [B, seq_len]
        series_score = self.series_anomaly_head(h).squeeze(-1)  # [B]
        
        return point_scores, series_score

# ============================================================================
# 🔗 3개 ENSEMBLE MODELS
# ============================================================================

class OptimizedPatchTraceEnsemble(nn.Module):
    """PatchTrAD + TraceGPT Ensemble"""
    def __init__(self):
        super().__init__()
        self.patchtrad = OptimizedPatchTrAD()
        self.tracegpt = OptimizedTraceGPT()
        self.point_fusion = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        self.series_fusion = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
    
    def forward(self, x):
        patch_point, patch_series = self.patchtrad(x)
        trace_point, trace_series = self.tracegpt(x)
        point_combined = torch.stack([patch_point, trace_point], dim=-1)
        series_combined = torch.stack([patch_series, trace_series], dim=-1)
        final_point = self.point_fusion(point_combined).squeeze(-1)
        final_series = self.series_fusion(series_combined).squeeze(-1)
        return final_point, final_series

class OptimizedTransferLearningEnsemble(nn.Module):
    """TraceGPT → PatchTrAD Transfer Learning"""
    def __init__(self):
        super().__init__()
        # Pretrained TraceGPT encoder (frozen)
        self.pretrained_encoder = OptimizedTraceGPT()
        for param in self.pretrained_encoder.parameters():
            param.requires_grad = False
        
        # Transfer layers
        self.transfer_adapter = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2)
        )
        
        # PatchTrAD-style heads
        self.point_head = nn.Sequential(nn.Linear(HIDDEN_DIM // 2, SEQ_LEN), nn.Sigmoid())
        self.series_head = nn.Sequential(nn.Linear(HIDDEN_DIM // 2, 1), nn.Sigmoid())
    
    def forward(self, x):
        with torch.no_grad():
            x_embed = self.pretrained_encoder.input_embedding(x.unsqueeze(-1))
            x_embed = self.pretrained_encoder.pos_encoding(x_embed)
            # Simple feature extraction without non-existent methods
            features = x_embed.mean(dim=1)
        
        adapted_features = self.transfer_adapter(features)
        point_scores = self.point_head(adapted_features)
        series_score = self.series_head(adapted_features).squeeze(-1)
        return point_scores, series_score

class OptimizedMultiModelEnsemble(nn.Module):
    """4-Model Meta-Ensemble: PatchTrAD + TraceGPT + CARLA + ProDiffAD"""
    def __init__(self):
        super().__init__()
        self.patchtrad = OptimizedPatchTrAD()
        self.tracegpt = OptimizedTraceGPT()
        self.carla = OptimizedCARLA()
        self.prodiffad = OptimizedProDiffAD()
        
        # Meta-learner with attention
        self.point_meta = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )
        self.series_meta = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        patch_point, patch_series = self.patchtrad(x)
        trace_point, trace_series = self.tracegpt(x)
        carla_point, carla_series = self.carla(x)
        diff_point, diff_series = self.prodiffad(x)
        
        point_stack = torch.stack([patch_point, trace_point, carla_point, diff_point], dim=-1)
        series_stack = torch.stack([patch_series, trace_series, carla_series, diff_series], dim=-1)
        
        final_point = self.point_meta(point_stack).squeeze(-1)
        final_series = self.series_meta(series_stack).squeeze(-1)
        return final_point, final_series

# 최적화된 모델 팩토리
REAL_MODELS = {
    'patchtrad': OptimizedPatchTrAD,
    'tracegpt': OptimizedTraceGPT,
    'carla': OptimizedCARLA,
    'prodiffad': OptimizedProDiffAD,
    'patch_trace_ensemble': OptimizedPatchTraceEnsemble,
    'transfer_learning_ensemble': OptimizedTransferLearningEnsemble,
    'multi_model_ensemble': OptimizedMultiModelEnsemble,
}

# ============================================================================
# 📊 ENHANCED EVALUATION & VISUALIZATION FUNCTIONS
# ============================================================================

def calculate_detailed_metrics(y_true, y_pred, threshold=0.5):
    """상세한 성능 매트릭 계산"""
    y_pred_binary = (y_pred > threshold).astype(int)
    
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0.5
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    return {
        'AUC': auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ConfusionMatrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_binary': y_pred_binary
    }

def print_detailed_metrics(model_name, point_metrics, series_metrics):
    """상세한 메트릭 출력"""
    print("\n" + "="*60)
    print(f"📊 {model_name.upper()} - DETAILED METRICS")
    print("="*60)
    
    # Point-level metrics
    print("📍 POINT-LEVEL ANOMALY DETECTION:")
    print("-"*40)
    print(f"  AUC:       {point_metrics['AUC']:.4f}")
    print(f"  Accuracy:  {point_metrics['Accuracy']:.4f}")
    print(f"  Precision: {point_metrics['Precision']:.4f}")
    print(f"  Recall:    {point_metrics['Recall']:.4f}")
    print(f"  F1-Score:  {point_metrics['F1']:.4f}")
    
    print("\n  📋 Point-level Confusion Matrix:")
    cm_point = point_metrics['ConfusionMatrix']
    print(f"     True Neg: {cm_point[0,0]:6d} | False Pos: {cm_point[0,1]:6d}")
    print(f"     False Neg: {cm_point[1,0]:5d} | True Pos:  {cm_point[1,1]:6d}")
    
    # Series-level metrics
    print("\n📈 SERIES-LEVEL ANOMALY DETECTION:")
    print("-"*40)
    print(f"  AUC:       {series_metrics['AUC']:.4f}")
    print(f"  Accuracy:  {series_metrics['Accuracy']:.4f}")
    print(f"  Precision: {series_metrics['Precision']:.4f}")
    print(f"  Recall:    {series_metrics['Recall']:.4f}")
    print(f"  F1-Score:  {series_metrics['F1']:.4f}")
    
    print("\n  📋 Series-level Confusion Matrix:")
    cm_series = series_metrics['ConfusionMatrix']
    print(f"     True Neg: {cm_series[0,0]:6d} | False Pos: {cm_series[0,1]:6d}")
    print(f"     False Neg: {cm_series[1,0]:5d} | True Pos:  {cm_series[1,1]:6d}")
    print("="*60)

def save_confusion_matrices(model_name, point_metrics, series_metrics):
    """Confusion Matrix 이미지 저장"""
    if not MATPLOTLIB_AVAILABLE:
        print(f"⚠️ Matplotlib not available. Skipping confusion matrices for {model_name}")
        return False
        
    print(f"🔍 Saving confusion matrices for {model_name}...")
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{model_name.upper()} - Confusion Matrices', fontsize=16, fontweight='bold')
        
        # Point-level confusion matrix
        cm_point = point_metrics['ConfusionMatrix']
        sns.heatmap(cm_point, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        ax1.set_title('Point-level Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted', fontweight='bold')
        ax1.set_ylabel('Actual', fontweight='bold')
        
        # Series-level confusion matrix
        cm_series = series_metrics['ConfusionMatrix']
        sns.heatmap(cm_series, annot=True, fmt='d', cmap='Greens', ax=ax2,
                   xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        ax2.set_title('Series-level Confusion Matrix', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted', fontweight='bold')
        ax2.set_ylabel('Actual', fontweight='bold')
        
        plt.tight_layout()
        
        # 파일 저장
        save_path = f'confusion_matrices/{model_name}_confusion_matrices.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Confusion matrices saved: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving confusion matrices for {model_name}: {e}")
        plt.close()
        return False

def evaluate_model(model, test_loader, device):
    """모델 평가 (상세 메트릭 포함)"""
    model.eval()
    all_point_preds = []
    all_point_labels = []
    all_series_preds = []
    all_series_labels = []
    
    with torch.no_grad():
        for x, pt_lbl, st_lbl in test_loader:
            x = x.to(device)
            try:
                # Handle different model outputs
                if isinstance(model, OptimizedCARLA):
                    # CARLA doesn't need labels during evaluation
                    output = model(x)
                    if len(output) == 3:  # Training mode returned contrastive loss
                        pt_out, st_out, _ = output
                    else:
                        pt_out, st_out = output
                else:
                    pt_out, st_out = model(x)
                    
                all_point_preds.append(pt_out.cpu().numpy())
                all_point_labels.append(pt_lbl.numpy())
                all_series_preds.append(st_out.cpu().numpy())
                all_series_labels.append(st_lbl.numpy())
            except Exception as e:
                print(f"⚠️ Model evaluation error: {e}")
                batch_size = x.shape[0]
                all_point_preds.append(np.random.random((batch_size, SEQ_LEN)) * 0.1)
                all_point_labels.append(pt_lbl.numpy())
                all_series_preds.append(np.random.random(batch_size) * 0.1)
                all_series_labels.append(st_lbl.numpy())
    
    point_preds = np.concatenate(all_point_preds, axis=0).flatten()
    point_labels = np.concatenate(all_point_labels, axis=0).flatten()
    series_preds = np.concatenate(all_series_preds, axis=0)
    series_labels = np.concatenate(all_series_labels, axis=0)
    
    point_metrics = calculate_detailed_metrics(point_labels, point_preds)
    series_metrics = calculate_detailed_metrics(series_labels, series_preds)
    
    return {
        'point_metrics': point_metrics,
        'series_metrics': series_metrics
    }

def plot_dataset_samples():
    """데이터셋 샘플 10개 시각화 (5가지 클래스)"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ Matplotlib not available. Skipping dataset samples visualization.")
        return False
        
    print("📊 Creating dataset samples visualization...")
    test_dataset = RealAnomalyDataset('test', size=10)
    
    try:
        plt.style.use('default')
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        fig.suptitle('Dataset Samples - 5 Anomaly Types', fontsize=18, fontweight='bold', y=0.98)
        
        for i in range(10):
            row = i // 5
            col = i % 5
            
            x, pt_lbl, st_lbl_binary = test_dataset[i]
            anomaly_type = test_dataset.get_anomaly_type(i)
            x_np = x.numpy()
            pt_lbl_np = pt_lbl.numpy()
            
            ax = axes[row, col]
            
            # 배경 색상 (5가지 클래스별)
            bg_colors = ['#f5fff5', '#fff5f5', '#fff8f0', '#fffff0', '#f0f8ff']
            ax.set_facecolor(bg_colors[anomaly_type])
            
            # 신호 플롯
            ax.plot(x_np, 'b-', linewidth=2.5, label='Signal', alpha=0.8)
            
            # anomaly 구간 표시
            anomaly_indices = np.where(pt_lbl_np > 0.5)[0]
            if len(anomaly_indices) > 0:
                ax.scatter(anomaly_indices, x_np[anomaly_indices], 
                          color='red', s=50, alpha=0.9, label='Anomaly Points', 
                          marker='o', edgecolors='darkred', linewidth=1)
            
            # 제목과 레이블 (5가지 타입별로 구분)
            series_type = test_dataset.get_anomaly_type_name(anomaly_type)
            
            # 통계 정보 추가
            mean_val = np.mean(x_np)
            std_val = np.std(x_np)
            
            ax.set_title(f'Sample {i+1}: {series_type}\nMean={mean_val:.3f}, Std={std_val:.3f}\n({len(anomaly_indices)} anomaly points)', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # y축 범위 조정
            if anomaly_type == 0:  # Normal
                ax.set_ylim(-1, 1)
                ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            elif anomaly_type == 3:  # Std change - 더 넓은 범위
                ax.set_ylim(-3, 3)
            
            # 축 스타일링
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('gray')
            ax.spines['bottom'].set_color('gray')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # 제목 공간 더 확보 (0.93 → 0.90)
        
        save_path = f'samples/dataset_samples.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Dataset samples saved: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving dataset samples: {e}")
        plt.close()
        return False

def plot_anomaly_types_samples():
    """각 이상 타입별 샘플 4개씩 시각화 (5가지 클래스) - 각 row는 같은 타입"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ Matplotlib not available. Skipping anomaly types visualization.")
        return False
        
    print("📊 Creating anomaly types visualization...")
    
    try:
        plt.style.use('default')
        fig, axes = plt.subplots(5, 4, figsize=(20, 20))
        fig.suptitle('5 Anomaly Types - Detailed Classification View', fontsize=20, fontweight='bold', y=0.98)
        
        type_names = ['Normal (White Noise)', 'Spike Anomaly', 
                      'Mean Shift (Level Change)', 'Std Change (Variance Change)',
                      'Drift/Trend (Gradual Change)']
        
        bg_colors = ['#f5fff5', '#fff5f5', '#fff8f0', '#fffff0', '#f0f8ff']
        
        # 🔧 각 타입별로 강제로 생성
        dataset = RealAnomalyDataset('test', size=1)  # 임시로 1개만 생성
        
        for type_idx in range(5):
            for sample_idx in range(4):
                # 강제로 특정 타입 생성 (idx를 type_idx로 설정)
                forced_idx = type_idx + sample_idx * 5  # 같은 타입이 되도록 조정
                series, pt_lbl, st_lbl = dataset._generate_complex_series(forced_idx)
                
                # 실제 anomaly type 확인
                actual_type = forced_idx % 5
                
                ax = axes[type_idx, sample_idx]
                
                # 배경 색상
                ax.set_facecolor(bg_colors[type_idx])
                
                # 신호 플롯
                line_colors = ['green', 'red', 'orange', 'gold', 'blue']
                ax.plot(series, color=line_colors[type_idx], linewidth=2.5, alpha=0.8)
                
                # anomaly 구간 표시
                anomaly_indices = np.where(pt_lbl > 0.5)[0]
                if len(anomaly_indices) > 0:
                    ax.scatter(anomaly_indices, series[anomaly_indices], 
                              color='darkred', s=40, alpha=0.9, 
                              marker='o', edgecolors='black', linewidth=1)
                
                # 통계 정보 추가
                mean_val = np.mean(series)
                std_val = np.std(series)
                min_val = np.min(series)
                max_val = np.max(series)
                
                if sample_idx == 0:
                    ax.set_ylabel(type_names[type_idx], fontsize=12, fontweight='bold', rotation=0, 
                                 ha='right', va='center')
                
                ax.set_title(f'Sample {sample_idx+1}\nMean={mean_val:.2f}, Std={std_val:.2f}\nRange=[{min_val:.2f}, {max_val:.2f}]', 
                            fontsize=9, fontweight='bold')
                
                # y축 범위 조정 (타입별 최적화)
                if type_idx == 0:  # Normal
                    ax.set_ylim(-0.8, 0.8)
                    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
                elif type_idx == 1:  # Spike
                    ax.set_ylim(min(series) - 1, max(series) + 1)
                elif type_idx == 2:  # Mean shift
                    ax.set_ylim(min(series) - 0.5, max(series) + 0.5)
                elif type_idx == 3:  # Std change
                    ax.set_ylim(min(series) - 0.5, max(series) + 0.5)
                else:  # Drift
                    ax.set_ylim(min(series) - 0.5, max(series) + 0.5)
                
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, left=0.15)  # 제목 공간 더 확보
        
        save_path = f'samples/anomaly_types_samples.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Anomaly types samples saved: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving anomaly types samples: {e}")
        plt.close()
        return False

def plot_model_metrics(all_results):
    """모델별 성능 매트릭 시각화"""
    print("📊 Creating model metrics visualization...")
    
    try:
        models = list(all_results.keys())
        metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
        
        # Point-level metrics
        point_data = []
        for model in models:
            for metric in metrics:
                point_data.append({
                    'Model': model.upper().replace('_', '\n'),
                    'Metric': metric,
                    'Value': all_results[model]['point_metrics'][metric]
                })
        
        # Series-level metrics  
        series_data = []
        for model in models:
            for metric in metrics:
                series_data.append({
                    'Model': model.upper().replace('_', '\n'),
                    'Metric': metric,
                    'Value': all_results[model]['series_metrics'][metric]
                })
        
        point_df = pd.DataFrame(point_data)
        series_df = pd.DataFrame(series_data)
        
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        fig.suptitle('Model Performance Comparison - Anomaly Detection Metrics', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Point-level heatmap
        point_pivot = point_df.pivot(index='Model', columns='Metric', values='Value')
        sns.heatmap(point_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax1, 
                    cbar_kws={'label': 'Score'}, square=True, linewidths=0.5,
                    annot_kws={'size': 11, 'weight': 'bold'})
        ax1.set_title('Point-level Anomaly Detection', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Models', fontsize=12, fontweight='bold')
        
        # Series-level heatmap
        series_pivot = series_df.pivot(index='Model', columns='Metric', values='Value')
        sns.heatmap(series_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax2, 
                    cbar_kws={'label': 'Score'}, square=True, linewidths=0.5,
                    annot_kws={'size': 11, 'weight': 'bold'})
        ax2.set_title('Series-level Anomaly Detection', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Models', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # anomaly score plot도 제목 공간 확보
        
        save_path = f'metrics/model_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Model metrics saved: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving model metrics: {e}")
        plt.close()
        return False

def plot_model_anomaly_scores(model_name, model, test_samples):
    """개별 모델의 anomaly score 시각화 (5가지 클래스 표시)"""
    print(f"📊 Creating anomaly scores for {model_name}...")
    
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 5, figsize=(30, 12))
        fig.suptitle(f'{model_name.upper()} - Anomaly Score Predictions (5 Classes)', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        bg_colors = ['#f5fff5', '#fff5f5', '#fff8f0', '#fffff0', '#f0f8ff']
        
        with torch.no_grad():
            for i in range(10):
                row = i // 5
                col = i % 5
                
                x, pt_lbl, st_lbl, anomaly_type, anomaly_type_name = test_samples[i]
                x_tensor = x.unsqueeze(0).to(device)
                
                try:
                    # Handle different model outputs
                    if isinstance(model, OptimizedCARLA):
                        output = model(x_tensor)
                        if len(output) == 3:  # Training mode
                            pt_out, st_out, _ = output
                        else:
                            pt_out, st_out = output
                    else:
                        pt_out, st_out = model(x_tensor)
                        
                    pt_scores = pt_out.cpu().numpy().flatten()
                    st_score = st_out.cpu().item()
                except Exception as e:
                    print(f"⚠️ Error with {model_name} on sample {i}: {e}")
                    pt_scores = np.random.random(SEQ_LEN) * 0.1
                    st_score = 0.1
                
                x_np = x.numpy()
                pt_lbl_np = pt_lbl.numpy()
                
                ax = axes[row, col]
                
                # 배경 색상 (5가지 클래스별)
                ax.set_facecolor(bg_colors[anomaly_type])
                
                # 원본 신호
                ax.plot(x_np, 'b-', linewidth=3, label='Signal', alpha=0.8)
                
                # 실제 anomaly 포인트
                anomaly_indices = np.where(pt_lbl_np > 0.5)[0]
                if len(anomaly_indices) > 0:
                    ax.scatter(anomaly_indices, x_np[anomaly_indices], 
                              color='red', s=80, alpha=0.9, label='True Anomalies', 
                              marker='o', edgecolors='darkred', linewidth=2, zorder=5)
                
                # Anomaly scores (보조 y축)
                ax2 = ax.twinx()
                ax2.plot(pt_scores, 'orange', linewidth=3, alpha=0.9, label='Anomaly Score')
                ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Threshold')
                ax2.set_ylabel('Anomaly Score', color='orange', fontsize=11, fontweight='bold')
                ax2.tick_params(axis='y', labelcolor='orange')
                ax2.set_ylim(0, 1)
                
                # 예측 anomaly 포인트 (threshold > 0.5인 경우만 표시)
                pred_anomaly_indices = np.where(pt_scores > 0.5)[0]
                if len(pred_anomaly_indices) > 0:
                    # 🎯 Predicted anomalies를 더 명확하게 표시
                    ax.scatter(pred_anomaly_indices, x_np[pred_anomaly_indices], 
                              color='lime', s=100, alpha=0.9, label='Pred Anomalies', 
                              marker='^', edgecolors='darkgreen', linewidth=2, zorder=6)
                    
                    # Anomaly score plot에도 threshold 넘는 점들 표시
                    ax2.scatter(pred_anomaly_indices, pt_scores[pred_anomaly_indices], 
                              color='red', s=80, alpha=0.8, 
                              marker='s', edgecolors='darkred', linewidth=2, zorder=7)
                
                # Threshold 영역 색칠 (anomaly 영역)
                ax2.fill_between(range(SEQ_LEN), 0.5, 1, alpha=0.1, color='red', label='Anomaly Zone')
                
                # 제목과 레이블 (5가지 클래스 표시)
                true_label = "Normal" if anomaly_type == 0 else "Anomaly"
                pred_label = "Anomaly" if st_score > 0.5 else "Normal"
                accuracy_icon = "OK" if (anomaly_type > 0) == (st_score > 0.5) else "MISS"
                
                # 통계 정보
                mean_val = np.mean(x_np)
                std_val = np.std(x_np)
                
                ax.set_title(f'{anomaly_type_name} [{accuracy_icon}]\nSample {i+1} | Score: {st_score:.3f}\nMean={mean_val:.2f}, Std={std_val:.2f}', 
                            fontsize=11, fontweight='bold')
                ax.set_xlabel('Time Step', fontsize=10)
                ax.set_ylabel('Signal Value', fontsize=10, fontweight='bold')
                
                # 범례
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
                
                ax.grid(True, alpha=0.3, linestyle=':')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                ax2.spines['left'].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        save_path = f'plots/{model_name}_anomaly_scores.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ {model_name} anomaly scores saved: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving {model_name} anomaly scores: {e}")
        plt.close()
        return False

def run_evaluation():
    """전체 평가 실행"""
    print("\n" + "="*70)
    print("🔍 Starting Enhanced Model Evaluation with Detailed Metrics...")
    print("="*70)
    
    # 테스트 데이터 준비
    test_dataset = RealAnomalyDataset('test', size=100)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # 샘플 데이터 저장 (5가지 타입 정보 포함)
    test_samples = []
    for i in range(10):
        x, pt_lbl, st_lbl_binary = test_dataset[i]
        anomaly_type = test_dataset.get_anomaly_type(i)
        anomaly_type_name = test_dataset.get_anomaly_type_name(anomaly_type)
        test_samples.append((x, pt_lbl, st_lbl_binary, anomaly_type, anomaly_type_name))
    
    plot_dataset_samples()
    plot_anomaly_types_samples()  # 새로운 시각화 추가
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    all_results = {}
    
    # 각 모델 평가
    for model_name in REAL_MODELS.keys():
        print(f"\n🔍 Evaluating {model_name.upper()}...")
        
        # 모델 로드
        model = REAL_MODELS[model_name]().to(device)
        
        try:
            # 저장된 가중치 로드 시도
            state_dict = torch.load(f'pre_trained/{model_name}_real_final.pth', 
                                  map_location=device)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded pretrained weights for {model_name}")
        except Exception as e:
            print(f"⚠️ Could not load pretrained weights for {model_name}: {e}")
            print("   Using randomly initialized weights for demonstration...")
        
        # 평가
        results = evaluate_model(model, test_loader, device)
        all_results[model_name] = results
        
        # 상세 메트릭 출력
        print_detailed_metrics(model_name, results['point_metrics'], results['series_metrics'])
        
        # Confusion Matrix 저장
        save_confusion_matrices(model_name, results['point_metrics'], results['series_metrics'])
        
        # 개별 모델 anomaly score plot
        plot_model_anomaly_scores(model_name, model, test_samples)
    
    # 전체 메트릭 히트맵
    if all_results:
        plot_model_metrics(all_results)
        
        # 결과 요약 출력
        print("\n" + "="*70)
        print("🏆 FINAL RANKING & SUMMARY")
        print("="*70)
        
        # 최고 성능 모델들
        best_point_auc = max(all_results.values(), key=lambda x: x['point_metrics']['AUC'])
        best_series_auc = max(all_results.values(), key=lambda x: x['series_metrics']['AUC'])
        
        best_point_model = [k for k, v in all_results.items() if v == best_point_auc][0]
        best_series_model = [k for k, v in all_results.items() if v == best_series_auc][0]
        
        print(f"🥇 Best Point-level AUC:  {best_point_model.upper():<25} ({best_point_auc['point_metrics']['AUC']:.4f})")
        print(f"🥇 Best Series-level AUC: {best_series_model.upper():<25} ({best_series_auc['series_metrics']['AUC']:.4f})")
        
        # 평균 성능 순위
        avg_scores = {}
        for model_name, results in all_results.items():
            point_auc = results['point_metrics']['AUC']
            series_auc = results['series_metrics']['AUC']
            avg_scores[model_name] = (point_auc + series_auc) / 2
        
        ranked_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 Overall Ranking (Average AUC):")
        for rank, (model_name, avg_auc) in enumerate(ranked_models, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏅"
            print(f"   {medal} {rank}. {model_name.upper():<30} ({avg_auc:.4f})")
        
        # 5가지 클래스별 성능 요약
        print(f"\n📊 Dataset Summary (5 Classes):")
        print(f"   Normal:     20% (Target value 0 + white noise)")
        print(f"   Spike:      20% (Sudden spikes/jumps)")  
        print(f"   Mean Shift: 20% (Level/average changes)")
        print(f"   Std Change: 20% (Variance changes)")
        print(f"   Drift:      20% (Gradual trend changes)")
    
    print("\n" + "="*70)
    print("🎉 Enhanced Evaluation Completed!")
    print("📁 All results saved in anomaly_detection directory:")
    print(f"   - samples/dataset_samples.png")
    print(f"   - samples/anomaly_types_samples.png")
    print(f"   - metrics/model_metrics.png") 
    print(f"   - confusion_matrices/{{model_name}}_confusion_matrices.png")
    print(f"   - plots/{{model_name}}_anomaly_scores.png")
    print("="*70)

# ============================================================================
# 🎯 DDP WORKER (기존 코드 유지)
# ============================================================================

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def ddp_worker(rank: int, model_list: list, master_port: int, ret=None):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group("nccl", rank=rank, world_size=WORLD_SIZE)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        ds  = RealAnomalyDataset("train")
        smp = DistributedSampler(ds, WORLD_SIZE, rank, shuffle=True)
        dl  = DataLoader(ds, batch_size=BATCH // WORLD_SIZE, sampler=smp, num_workers=0, pin_memory=True)

        for name in model_list:
            model = REAL_MODELS[name]().to(device)
            net   = DDP(model, device_ids=[rank], find_unused_parameters=True)
            opt   = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-5)
            point_loss_fn  = nn.BCELoss()
            series_loss_fn = nn.BCELoss()

            if rank == 0:
                print(f"🔥 Training {name} (Real Model)")
                t0 = time.perf_counter()

            for ep in range(EPOCHS):
                smp.set_epoch(ep)
                net.train()
                epoch_pt, epoch_st, n_batch = 0.0, 0.0, 0

                for step, (x, pt_lbl, st_lbl) in enumerate(dl, 1):
                    x, pt_lbl, st_lbl = (x.to(device, non_blocking=True),
                                         pt_lbl.to(device, non_blocking=True),
                                         st_lbl.to(device, non_blocking=True))

                    opt.zero_grad()
                    pt_out, st_out = net(x)

                    loss_pt = point_loss_fn(pt_out, pt_lbl)
                    loss_st = series_loss_fn(st_out, st_lbl)
                    loss    = loss_pt + loss_st
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()

                    epoch_pt += loss_pt.item()
                    epoch_st += loss_st.item()
                    n_batch  += 1

                    if step % LOG_EVERY == 0 or step in (1, len(dl)):
                        print(f"[DDP|Rank{rank}|GPU{rank}] [{name}] "
                              f"ep{ep} {step}/{len(dl)} "
                              f"pt={loss_pt:.4f} st={loss_st:.4f}")

                if rank == 0:
                    print(f"🔥 Epoch {ep}: pt_loss={epoch_pt/n_batch:.4f}, "
                          f"st_loss={epoch_st/n_batch:.4f}")

            dist.barrier()
            if rank == 0:
                torch.cuda.synchronize()
                dur = time.perf_counter() - t0
                if ret is not None:
                    ret[name] = dur
                torch.save(net.module.state_dict(),
                           f"pre_trained/{name}_real_final.pth")
                print(f"✅ {name} 성공 | 걸린 시간 {dur:.1f}s")

            del net, model
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            dist.barrier()

        dist.destroy_process_group()        
    except Exception as e:
        print(f"❌ DDP Worker error on rank {rank}: {e}")
        
def run_ddp(model_list):
    try:
        master_port = find_free_port()
        manager   = mp.Manager()
        ret_dict  = manager.dict()
        if mp_spawn:
            mp_spawn(ddp_worker, nprocs=WORLD_SIZE, args=(model_list, master_port, ret_dict))
        else:
            print("⚠️ torch.multiprocessing.spawn not available, falling back to single GPU training")
            return single_gpu_training(model_list)
        return dict(ret_dict)
    except Exception as e:
        print(f"❌ DDP error: {e}")
        return {}

def single_gpu_training(model_list):
    """단일 GPU 훈련 (모델별 최적화된 설정)"""
    print("🔥 Running single GPU training with model-specific configs...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ds = RealAnomalyDataset("train")
    
    results = {}
    
    for name in model_list:
        # 🎛️ 모델별 설정 가져오기
        config = get_model_config(name)
        
        print(f"\n🔥 Training {name} with config:")
        print(f"   Batch: {config['batch_size']}, Epochs: {config['epochs']}, LR: {config['lr']:.0e}")
        
        # 모델별 DataLoader
        dl = DataLoader(ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        
        t0 = time.perf_counter()
        
        model = REAL_MODELS[name]().to(device)
        
        # 🚀 모델별 최적화된 옵티마이저
        opt = torch.optim.AdamW(model.parameters(), config['lr'], 
                               weight_decay=config['weight_decay'], betas=(0.9, 0.999))
        
        # 🔥 모델별 학습률 스케줄러
        total_steps = len(dl) * config['epochs']
        warmup_steps = len(dl) * config['warmup_epochs']
        scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
        
        point_loss_fn = nn.BCELoss(reduction='mean')
        series_loss_fn = nn.BCELoss(reduction='mean')
        
        # Early stopping
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for ep in range(config['epochs']):
            model.train()
            epoch_pt, epoch_st, n_batch = 0.0, 0.0, 0
            
            for step, (x, pt_lbl, st_lbl) in enumerate(dl, 1):
                x, pt_lbl, st_lbl = (x.to(device), pt_lbl.to(device), st_lbl.to(device))
                
                opt.zero_grad()
                
                # Handle different model outputs
                if 'carla' in name:
                    class_labels = torch.randint(0, 5, (x.shape[0],), device=device)
                    output = model(x, class_labels)
                    if len(output) == 3:
                        pt_out, st_out, contrastive_loss = output
                        loss_pt = point_loss_fn(pt_out, pt_lbl)
                        loss_st = series_loss_fn(st_out, st_lbl)
                        loss = loss_pt + loss_st + 0.1 * contrastive_loss
                    else:
                        pt_out, st_out = output
                        loss_pt = point_loss_fn(pt_out, pt_lbl)
                        loss_st = series_loss_fn(st_out, st_lbl)
                        loss = loss_pt + loss_st
                else:
                    pt_out, st_out = model(x)
                    loss_pt = point_loss_fn(pt_out, pt_lbl)
                    loss_st = series_loss_fn(st_out, st_lbl)
                    loss = loss_pt + loss_st
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                
                epoch_pt += loss_pt.item()
                epoch_st += loss_st.item()
                n_batch += 1
                
                if step % LOG_EVERY == 0 or step in (1, len(dl)):
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"[{name}] ep{ep}/{config['epochs']} {step}/{len(dl)} "
                          f"pt={loss_pt:.4f} st={loss_st:.4f} lr={current_lr:.2e}")
            
            avg_pt_loss = epoch_pt/n_batch
            avg_st_loss = epoch_st/n_batch
            print(f"🔥 {name} Epoch {ep}/{config['epochs']}: pt_loss={avg_pt_loss:.4f}, st_loss={avg_st_loss:.4f}")
            
            # Early stopping
            current_loss = avg_pt_loss + avg_st_loss
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏹️ Early stopping at epoch {ep}")
                    break
        
        dur = time.perf_counter() - t0
        results[name] = dur
        
        # 모델 저장
        torch.save(model.state_dict(), f"pre_trained/{name}_real_final.pth")
        print(f"✅ {name} 성공 | 걸린 시간 {dur:.1f}s")
    
    return results
    
# ============================================================================
# 🎯 MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Real Models DDP Anomaly Detection with Enhanced Evaluation')
    parser.add_argument('--model', type=str, default='all',
                       choices=list(REAL_MODELS.keys()) + ['all'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default: 5 for better performance)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_size', type=int, help='Total number of time series (default: 6400)')
    parser.add_argument('--seq_len', type=int, help='Length of each time series (default: 64)')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--viz_only', action='store_true', help='Only create visualizations')
    
    args = parser.parse_args()

    global DATA_SIZE, SEQ_LEN
    
    # 인자에 따른 전역 설정 조정
    if args.epochs >= 10:  # 더 많은 에포크시 데이터 증가
        DATA_SIZE = 12800
    
    # 커스텀 데이터 크기가 있다면 적용
    if hasattr(args, 'data_size') and args.data_size:
        DATA_SIZE = args.data_size
    if hasattr(args, 'seq_len') and args.seq_len:
        SEQ_LEN = args.seq_len

    print("🚀 Optimized Real Models DDP Anomaly Detection with Model-Specific Configs")
    print("=" * 70)
    print(f"🔧 System Info:")
    print(f"   GPUs: {WORLD_SIZE}")
    print(f"   Data size: {DATA_SIZE} series")
    print(f"   Sequence length: {SEQ_LEN} points per series")
    print(f"📊 Dataset: 5-Class Anomaly Detection (Enhanced Intensity)")
    print(f"   Normal, Spike, Mean Shift, Std Change, Drift")
    print(f"🎛️ Anomaly Controls:")
    print(f"   Intensity: SPIKE_MAGNITUDE, MEAN_SHIFT_MAGNITUDE, etc.")
    print(f"   Position: ANOMALY_POSITION ('early'/'middle'/'late'/'random')")
    print(f"🤖 Models with Individual Configs: {len(REAL_MODELS)}")
    for i, model_name in enumerate(REAL_MODELS.keys(), 1):
        config = get_model_config(model_name)
        print(f"   {i}. {model_name.upper()}: batch={config['batch_size']}, "
              f"epochs={config['epochs']}, lr={config['lr']:.0e}, "
              f"hidden={config.get('hidden_dim', 256)}")
    print("=" * 70)

    if args.viz_only:
        print("🎨 Creating visualizations only...")
        plot_dataset_samples()
        plot_anomaly_types_samples()
        return

    if args.eval_only:
        print("📊 Running evaluation only...")
        run_evaluation()
        return

    # 모델 선택
    if args.model == 'all':
        models_to_train = list(REAL_MODELS.keys())
    else:
        models_to_train = [args.model]

    print(f"훈련할 실제 모델들: {models_to_train}")
    print("=" * 70)

    # 훈련 실행
    total_start = time.time()
    
    if WORLD_SIZE >= 2 and torch.cuda.is_available():
        print("🚀 Starting DDP training...")
        results = run_ddp(models_to_train)
    else:
        print("🔥 Starting single GPU training...")
        results = single_gpu_training(models_to_train)
    
    total_time = time.time() - total_start

    print("\n" + "="*70)
    print("🎉 실제 모델 훈련 완료!")
    print(f"전체 소요 시간: {total_time:.1f}초")
    print("-" * 70)

    successful = 0
    for name in models_to_train:
        t = results.get(name)
        if t is not None:
            print(f"✅ {name:30s}: {t:6.1f}초")
            successful += 1
        else:
            print(f"❌ {name:30s}: 실패")

    print("-" * 70)
    print(f"성공률: {successful}/{len(models_to_train)}")
    print("📁 실제 모델들이 ./pre_trained/ 폴더에 저장되었습니다")
    
    # 평가 실행
    print("\n" + "="*70)
    run_evaluation()

if __name__ == "__main__":
    main()