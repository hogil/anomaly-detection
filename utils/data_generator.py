#!/usr/bin/env python3
"""
data_generator.py - anomaly 변화량/구간/spike/complex 모두 random하게 생성, point/sample label 0~5
"""
import os
import logging
from typing import Tuple, List
import numpy as np
import torch
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_balanced_dataset(
    n_samples: int = 500,
    seq_len: int = 64,
    normal_ratio: float = 0.8,
    noise_level: float = 0.02,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    반환: (data, point_labels, sample_labels, types)
    - data: (n_samples, seq_len, 1)
    - point_labels: (n_samples, seq_len)  # 각 시점별 0~5 type
    - sample_labels: (n_samples,)  # 시계열별 0~5 type
    - types: (n_samples,)  # 시계열별 anomaly type(str)
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    normal_samples = int(n_samples * normal_ratio)
    anomaly_samples = n_samples - normal_samples
    data: List[np.ndarray] = []
    point_labels: List[np.ndarray] = []
    sample_labels: List[int] = []
    types: List[str] = []
    # Normal: 0
    for _ in range(normal_samples):
        ts = np.ones(seq_len) * 1.0 + np.random.normal(0, noise_level, seq_len)
        data.append(ts)
        point_labels.append(np.zeros(seq_len, dtype=int))
        sample_labels.append(0)
        types.append('Normal')
    anomaly_per_type = anomaly_samples // 4
    # 1. avg change: 1
    for i in range(anomaly_per_type):
        local_seed = random_seed + 1000 + i
        np.random.seed(local_seed)
        ts = np.ones(seq_len) * 1.0 + np.random.normal(0, noise_level, seq_len)
        change_mag = np.random.uniform(0.02, 0.05)  # 변화량 더 낮춤
        change_start = np.random.randint(seq_len*5//6, seq_len*11//12)
        ts[change_start:] = 1.0 + change_mag + np.random.normal(0, noise_level, seq_len-change_start)
        label = np.zeros(seq_len, dtype=int)
        label[change_start:] = 1
        data.append(ts)
        point_labels.append(label)
        sample_labels.append(1)
        types.append('avg change')
    # 2. std change: 2
    for i in range(anomaly_per_type):
        local_seed = random_seed + 2000 + i
        np.random.seed(local_seed)
        ts = np.ones(seq_len) * 1.0 + np.random.normal(0, noise_level, seq_len)
        change_start = np.random.randint(seq_len*5//6, seq_len*11//12)
        max_change_len = seq_len - change_start
        change_len = min(np.random.randint(seq_len//8, seq_len//3), max_change_len)
        std_factor = np.random.uniform(2, 4)
        ts[change_start:change_start+change_len] = 1.0 + np.random.normal(0, noise_level*std_factor, change_len)
        label = np.zeros(seq_len, dtype=int)
        label[change_start:change_start+change_len] = 2
        data.append(ts)
        point_labels.append(label)
        sample_labels.append(2)
        types.append('std change')
    # 3. drift: 3
    for i in range(anomaly_per_type):
        local_seed = random_seed + 3000 + i
        np.random.seed(local_seed)
        ts = np.ones(seq_len) * 1.0 + np.random.normal(0, noise_level, seq_len)
        drift_start = np.random.randint(seq_len*4//6, seq_len*9//12)
        drift_len = seq_len - drift_start
        drift_mag = np.random.uniform(0.07, 0.12)  # 변화량 더 높임
        drift = np.linspace(0, drift_mag, drift_len)
        ts[drift_start:] += drift
        label = np.zeros(seq_len, dtype=int)
        label[drift_start:] = 3
        data.append(ts)
        point_labels.append(label)
        sample_labels.append(3)
        types.append('drift')
    # 4. spike: 4 (항상 시계열 마지막 4개 구간 내에서만 발생)
    remaining = anomaly_samples - 3 * anomaly_per_type
    for i in range(remaining):
        local_seed = random_seed + 4000 + i
        np.random.seed(local_seed)
        ts = np.ones(seq_len) * 1.0 + np.random.normal(0, noise_level, seq_len)
        n_spikes = np.random.randint(2, 5)
        spike_positions = np.random.choice(range(seq_len-4, seq_len), size=n_spikes, replace=False)
        label = np.zeros(seq_len, dtype=int)
        for pos in spike_positions:
            spike_mag = np.random.uniform(0.05, 0.1)
            ts[pos] = 1.0 + spike_mag + np.random.normal(0, noise_level)
            label[pos] = 4
        data.append(ts)
        point_labels.append(label)
        sample_labels.append(4)
        types.append('spike')
    # 5. complex: 5 (여러 anomaly type이 겹칠 수 있음, 가장 큰 type으로 sample label)
    n_complex = max(1, anomaly_samples // 8)
    for i in range(n_complex):
        local_seed = random_seed + 5000 + i
        np.random.seed(local_seed)
        ts = np.ones(seq_len) * 1.0 + np.random.normal(0, noise_level, seq_len)
        label = np.zeros(seq_len, dtype=int)
        # avg change
        change_start = np.random.randint(seq_len*5//6, seq_len*11//12)
        change_mag = np.random.uniform(0.02, 0.07)
        ts[change_start:] = 1.0 + change_mag + np.random.normal(0, noise_level, seq_len-change_start)
        label[change_start:] = np.maximum(label[change_start:], 1)
        # drift
        drift_start = np.random.randint(seq_len*4//6, seq_len*9//12)
        drift_len = seq_len - drift_start
        drift_mag = np.random.uniform(0.04, 0.09)
        drift = np.linspace(0, drift_mag, drift_len)
        ts[drift_start:] += drift
        label[drift_start:] = np.maximum(label[drift_start:], 3)
        # spike
        n_spikes = np.random.randint(2, 5)
        spike_positions = np.random.choice(range(seq_len-4, seq_len), size=n_spikes, replace=False)
        for pos in spike_positions:
            spike_mag = np.random.uniform(0.05, 0.1)
            ts[pos] = 1.0 + spike_mag + np.random.normal(0, noise_level)
            label[pos] = 4
        # sample label: 가장 큰 type(5)
        sample_label = 5 if np.any(label > 0) else 0
        data.append(ts)
        point_labels.append(label)
        sample_labels.append(sample_label)
        types.append('complex')
    data_tensor = torch.FloatTensor(np.array(data)).unsqueeze(-1)
    point_labels_tensor = torch.LongTensor(np.array(point_labels))
    sample_labels_tensor = torch.LongTensor(np.array(sample_labels))
    logger.info(f"Custom anomaly dataset generated: shape={data_tensor.shape}, normal={int((sample_labels_tensor==0).sum())}, anomaly={int((sample_labels_tensor>0).sum())}")
    return data_tensor, point_labels_tensor, sample_labels_tensor, types

def save_dataset_samples(
    data: torch.Tensor,
    point_labels: torch.Tensor,
    types: List[str],
    save_path: str = "samples/"
) -> None:
    os.makedirs(save_path, exist_ok=True)
    try:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Custom Anomaly Type Dataset Samples', fontsize=16)
        normal_indices = (point_labels.sum(dim=1) == 0).nonzero().flatten()[:5]
        for i, idx in enumerate(normal_indices):
            axes[0, i].plot(data[idx].squeeze(), 'b-', linewidth=2)
            axes[0, i].set_title(f'Normal #{i+1}')
            axes[0, i].grid(True, alpha=0.3)
        anomaly_types = ['avg change', 'std change', 'drift', 'spike', 'complex']
        for i, anom_type in enumerate(anomaly_types):
            type_indices = [j for j, t in enumerate(types) if t == anom_type]
            if type_indices:
                idx = type_indices[0]
                ts = data[idx].squeeze().numpy()
                lbl = point_labels[idx].numpy()
                start = 0
                while start < len(ts):
                    cur_label = lbl[start]
                    end = start
                    while end < len(ts) and lbl[end] == cur_label:
                        end += 1
                    color = 'b-' if cur_label == 0 else 'r-'
                    axes[1, i].plot(range(start, end), ts[start:end], color, linewidth=2)
                    start = end
                axes[1, i].set_title(f'{anom_type}')
                axes[1, i].grid(True, alpha=0.3)
        plt.tight_layout()
        save_file = os.path.join(save_path, 'custom_anomaly_type_dataset_samples.png')
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Dataset samples saved to {save_file}")
    except Exception as e:
        logger.error(f"[SAMPLE SAVE ERROR] {e}")

if __name__ == "__main__":
    data, point_labels, sample_labels, types = generate_balanced_dataset(n_samples=500, seq_len=64)
    save_dataset_samples(data, point_labels, types)