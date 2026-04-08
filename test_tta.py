"""
TTA 효과 측정 - 기존 best 모델로 TTA on/off 비교
시간 증가율도 측정
"""
import time, json, sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from train import ChartImageDataset, create_model, evaluate, FocalLoss
from torch.utils.data import DataLoader
from torchvision import transforms

import yaml
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 가장 최근 best binary 모델 로드
import os
best_dir = None
best_abn = 0
for d in sorted(os.listdir('logs')):
    p = f'logs/{d}/best_info.json'
    if not os.path.exists(p):
        continue
    info = json.load(open(p))
    if info.get('hparams',{}).get('mode') != 'binary':
        continue
    bm = info.get('test_metrics', {})
    if 'abnormal' not in bm or bm['normal']['recall'] < 0.7:
        continue
    if bm['abnormal']['recall'] > best_abn:
        best_abn = bm['abnormal']['recall']
        best_dir = d

print(f"Best model: {best_dir} (abn={best_abn:.4f})")

# 데이터 로드
data_dir = Path("data")
sc_df = pd.read_csv(data_dir / "scenarios.csv")
test_df = sc_df[sc_df["split"] == "test"]

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

classes = ['normal', 'abnormal']
test_ds = ChartImageDataset(Path("images"), test_df, classes,
                            transform=val_transform, mode='binary')
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# 모델 로드
model = create_model(2, "convnextv2_tiny.fcmae_ft_in22k_in1k", device,
                     "weights/convnextv2_tiny_pretrained.pth", dropout=0.5)
ckpt = torch.load(f'logs/{best_dir}/best_model.pth', map_location=device, weights_only=True)
model.load_state_dict(ckpt)

criterion = FocalLoss(alpha=[1.0, 1.0], gamma=2.0).to(device)

# 1. 일반 평가
print("\n=== 일반 평가 (no TTA) ===")
t0 = time.time()
_, _, recall1, f1_1, metrics1, _, _ = evaluate(
    model, test_loader, criterion, device, classes,
    desc="Test (no TTA)", tta=False
)
t_normal = time.time() - t0
print(f"  abn_R={metrics1['abnormal']['recall']:.4f}")
print(f"  nor_R={metrics1['normal']['recall']:.4f}")
print(f"  F1   ={f1_1:.4f}")
print(f"  Time  ={t_normal:.2f}s")

# 2. TTA 평가
print("\n=== TTA 평가 (5x) ===")
t0 = time.time()
_, _, recall2, f1_2, metrics2, _, _ = evaluate(
    model, test_loader, criterion, device, classes,
    desc="Test (TTA)", tta=True
)
t_tta = time.time() - t0
print(f"  abn_R={metrics2['abnormal']['recall']:.4f}")
print(f"  nor_R={metrics2['normal']['recall']:.4f}")
print(f"  F1   ={f1_2:.4f}")
print(f"  Time  ={t_tta:.2f}s ({t_tta/t_normal:.1f}x slower)")

# 비교
print(f"\n=== 비교 ===")
print(f"  {'Metric':>10s} | {'No TTA':>8s} | {'TTA':>8s} | {'Diff':>8s}")
print(f"  ----------+----------+----------+--------")
print(f"  {'abn_R':>10s} | {metrics1['abnormal']['recall']:8.4f} | {metrics2['abnormal']['recall']:8.4f} | {metrics2['abnormal']['recall']-metrics1['abnormal']['recall']:+8.4f}")
print(f"  {'nor_R':>10s} | {metrics1['normal']['recall']:8.4f} | {metrics2['normal']['recall']:8.4f} | {metrics2['normal']['recall']-metrics1['normal']['recall']:+8.4f}")
print(f"  {'F1':>10s} | {f1_1:8.4f} | {f1_2:8.4f} | {f1_2-f1_1:+8.4f}")
print(f"  {'Time':>10s} | {t_normal:7.2f}s | {t_tta:7.2f}s | {(t_tta/t_normal-1)*100:+7.1f}%")
