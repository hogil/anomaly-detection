#!/usr/bin/env python3
"""
High-Quality Balanced Dataset Generator
균형잡힌 고품질 데이터셋 생성기 (Normal 80%, Anomaly 20%)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def generate_balanced_dataset(
    n_samples: int = 500,
    seq_len: int = 64,
    normal_ratio: float = 0.8,  # Normal 80%
    noise_level: float = 0.02,  # 낮은 노이즈로 고품질 데이터
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    균형잡힌 고품질 시계열 데이터셋 생성
    
    Args:
        n_samples: 총 샘플 수 (500개)
        seq_len: 시계열 길이 (64)
        normal_ratio: 정상 데이터 비율 (0.8 = 80%)
        noise_level: 노이즈 수준 (0.02로 낮춤)
        random_seed: 재현 가능한 시드
    
    Returns:
        data: [n_samples, seq_len, 1] 형태의 텐서
        labels: [n_samples] 형태의 라벨 (0=Normal, 1=Anomaly)
        types: 각 샘플의 타입 리스트
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    normal_samples = int(n_samples * normal_ratio)
    anomaly_samples = n_samples - normal_samples
    
    data = []
    labels = []
    types = []
    
    print(f"🔄 Generating balanced dataset:")
    print(f"   Total: {n_samples} samples")
    print(f"   Normal: {normal_samples} ({normal_ratio*100:.1f}%)")
    print(f"   Anomaly: {anomaly_samples} ({(1-normal_ratio)*100:.1f}%)")
    print(f"   Sequence length: {seq_len}")
    print(f"   Noise level: {noise_level}")
    
    # ==========================================
    # NORMAL DATA GENERATION (고품질 정상 데이터)
    # ==========================================
    
    for i in range(normal_samples):
        t = np.linspace(0, 1, seq_len)
        
        # 5가지 다양한 정상 패턴 (고품질)
        pattern_type = i % 5
        
        if pattern_type == 0:
            # 선형 증가 트렌드
            base = t * 2.0
            noise = np.random.normal(0, noise_level, seq_len)
            ts = base + noise
            
        elif pattern_type == 1:
            # 지수 성장 트렌드
            base = np.exp(t * 0.8) - 1
            noise = np.random.normal(0, noise_level, seq_len)
            ts = base + noise
            
        elif pattern_type == 2:
            # 로그 성장 트렌드
            base = np.log(1 + t * 5)
            noise = np.random.normal(0, noise_level, seq_len)
            ts = base + noise
            
        elif pattern_type == 3:
            # 2차 함수 트렌드
            base = t ** 2 * 3
            noise = np.random.normal(0, noise_level, seq_len)
            ts = base + noise
            
        else:
            # S-곡선 (시그모이드)
            base = 2 / (1 + np.exp(-10 * (t - 0.5)))
            noise = np.random.normal(0, noise_level, seq_len)
            ts = base + noise
        
        data.append(ts)
        labels.append(0)
        types.append('Normal')
    
    # ==========================================
    # ANOMALY DATA GENERATION (명확한 이상 데이터)
    # ==========================================
    
    anomaly_per_type = anomaly_samples // 4
    
    # 1. Spike Anomalies (강한 스파이크)
    for i in range(anomaly_per_type):
        # 정상 베이스 생성
        t = np.linspace(0, 1, seq_len)
        base = t * 2.0
        noise = np.random.normal(0, noise_level, seq_len)
        ts = base + noise
        
        # 강한 스파이크 추가 (2-4개)
        num_spikes = np.random.randint(1, 4)
        spike_positions = np.random.choice(
            range(seq_len//4, 3*seq_len//4), 
            size=num_spikes, 
            replace=False
        )
        
        for pos in spike_positions:
            # 매우 강한 스파이크 (정상 범위의 5-10배)
            spike_magnitude = np.random.uniform(3.0, 6.0)
            direction = np.random.choice([-1, 1])
            ts[pos] += direction * spike_magnitude
        
        data.append(ts)
        labels.append(1)
        types.append('Spike')
    
    # 2. Mean Shift Anomalies (수준 변화)
    for i in range(anomaly_per_type):
        t = np.linspace(0, 1, seq_len)
        base = t * 2.0
        noise = np.random.normal(0, noise_level, seq_len)
        ts = base + noise
        
        # 강한 수준 변화
        shift_point = np.random.randint(seq_len//4, 3*seq_len//4)
        shift_magnitude = np.random.uniform(2.0, 4.0)
        direction = np.random.choice([-1, 1])
        
        ts[shift_point:] += direction * shift_magnitude
        
        data.append(ts)
        labels.append(1)
        types.append('Mean_Shift')
    
    # 3. Variance Change Anomalies (분산 변화)
    for i in range(anomaly_per_type):
        t = np.linspace(0, 1, seq_len)
        base = t * 2.0
        ts = base.copy()
        
        # 변화 구간 설정
        change_start = np.random.randint(seq_len//4, seq_len//2)
        change_end = np.random.randint(change_start + seq_len//8, 3*seq_len//4)
        
        # 정상 구간: 낮은 노이즈
        ts[:change_start] += np.random.normal(0, noise_level, change_start)
        ts[change_end:] += np.random.normal(0, noise_level, seq_len - change_end)
        
        # 이상 구간: 매우 높은 분산
        high_variance = noise_level * 15  # 15배 증가
        ts[change_start:change_end] += np.random.normal(0, high_variance, change_end - change_start)
        
        data.append(ts)
        labels.append(1)
        types.append('Variance_Change')
    
    # 4. Trend Change Anomalies (트렌드 변화)
    remaining = anomaly_samples - 3 * anomaly_per_type
    for i in range(remaining):
        t = np.linspace(0, 1, seq_len)
        base = t * 2.0
        noise = np.random.normal(0, noise_level, seq_len)
        ts = base + noise
        
        # 급격한 트렌드 변화
        change_point = np.random.randint(seq_len//4, 3*seq_len//4)
        
        # 새로운 트렌드 추가 (매우 급격함)
        new_trend_slope = np.random.uniform(-4.0, 4.0)
        trend_length = seq_len - change_point
        new_trend = np.linspace(0, new_trend_slope, trend_length)
        
        ts[change_point:] += new_trend
        
        data.append(ts)
        labels.append(1)
        types.append('Trend_Change')
    
    # 텐서로 변환
    data = torch.FloatTensor(np.array(data)).unsqueeze(-1)  # [n_samples, seq_len, 1]
    labels = torch.LongTensor(labels)
    
    print(f"✅ Dataset generated successfully:")
    print(f"   Shape: {data.shape}")
    print(f"   Normal samples: {(labels == 0).sum().item()}")
    print(f"   Anomaly samples: {(labels == 1).sum().item()}")
    print(f"   Class distribution: {(labels == 0).sum().item()/(len(labels))*100:.1f}% Normal, {(labels == 1).sum().item()/(len(labels))*100:.1f}% Anomaly")
    
    return data, labels, types

def save_dataset_samples(data: torch.Tensor, labels: torch.Tensor, types: List[str], save_path: str = "samples/"):
    """데이터셋 샘플 시각화 및 저장"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Balanced High-Quality Dataset Samples', fontsize=16)
    
    # Normal 샘플 5개
    normal_indices = (labels == 0).nonzero().flatten()[:5]
    for i, idx in enumerate(normal_indices):
        axes[0, i].plot(data[idx].squeeze(), 'b-', linewidth=2)
        axes[0, i].set_title(f'Normal #{i+1}')
        axes[0, i].grid(True, alpha=0.3)
    
    # Anomaly 샘플 5개 (각 타입별로)
    anomaly_types = ['Spike', 'Mean_Shift', 'Variance_Change', 'Trend_Change']
    for i, anom_type in enumerate(anomaly_types):
        # 해당 타입의 첫 번째 샘플 찾기
        type_indices = [j for j, t in enumerate(types) if t == anom_type]
        if type_indices:
            idx = type_indices[0]
            axes[1, i].plot(data[idx].squeeze(), 'r-', linewidth=2)
            axes[1, i].set_title(f'{anom_type}')
            axes[1, i].grid(True, alpha=0.3)
    
    # 마지막 subplot은 전체 분포
    axes[1, 4].bar(['Normal', 'Anomaly'], 
                   [(labels == 0).sum().item(), (labels == 1).sum().item()],
                   color=['blue', 'red'], alpha=0.7)
    axes[1, 4].set_title('Class Distribution')
    axes[1, 4].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}balanced_dataset_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Dataset samples saved to {save_path}balanced_dataset_samples.png")

if __name__ == "__main__":
    # 테스트 실행
    data, labels, types = generate_balanced_dataset(n_samples=500, seq_len=64)
    save_dataset_samples(data, labels, types) 