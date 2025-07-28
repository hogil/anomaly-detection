#!/usr/bin/env python3
"""
CARLA: Contrastive Anomaly Detection with Representation Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math
import random

# Focal Loss 함수 추가
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class TimeSeriesEncoder(nn.Module):
    def __init__(self, seq_len: int, input_dim: int = 1, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Enhanced Multi-scale CNN encoder with residual connections
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, padding=2),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=7, padding=3),
        ])
        
        # Additional conv layers for better feature extraction
        self.conv_enhanced = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
        ])
        
        # Enhanced Bidirectional LSTM with layer normalization
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=num_layers, batch_first=True, 
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        
        # Multi-head attention with more heads (ensure divisible)
        num_heads = min(10, hidden_dim // 32)  # Ensure embed_dim is divisible by num_heads
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, 
                                                   dropout=dropout, batch_first=True)
        
        # Enhanced pooling with multiple scales
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.adaptive_pool_2 = nn.AdaptiveAvgPool1d(4)  # Multi-scale pooling
        
        # Enhanced projection with more layers
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim),  # 7 features now
            nn.GELU(),  # Better activation
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Enhanced CNN encoding with residual connections
        x_conv = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        for i, conv in enumerate(self.conv_layers):
            x_new = F.gelu(conv(x_conv))  # GELU activation
            if i > 0 and x_new.shape[1] == x_conv.shape[1]:
                x_conv = x_new + x_conv  # Residual connection
            else:
                x_conv = x_new
            x_conv = self.dropout(x_conv)
        
        # Additional enhanced conv layers
        for conv in self.conv_enhanced:
            x_enhanced = F.gelu(conv(x_conv))
            x_conv = x_conv + x_enhanced  # Residual connection
            x_conv = self.dropout(x_conv)
        
        x_conv = x_conv.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        
        # Enhanced LSTM encoding with layer norm
        lstm_out, _ = self.lstm(x_conv)
        lstm_out = self.layer_norm(lstm_out)
        
        # Enhanced self-attention
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        
        # Multi-scale pooling with enhanced features
        avg_pool = self.adaptive_pool(attn_out.transpose(1, 2)).squeeze(-1)
        max_pool = self.max_pool(attn_out.transpose(1, 2)).squeeze(-1)
        multi_scale_pool = self.adaptive_pool_2(attn_out.transpose(1, 2)).flatten(1)
        last_out = attn_out[:, -1, :]
        
        # Combine all features
        combined = torch.cat([avg_pool, max_pool, multi_scale_pool, last_out], dim=1)
        embeddings = self.projection(combined)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=1)
        
        pos_sim = pos_sim / self.temperature
        neg_sim = neg_sim / self.temperature
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        
        infonce_loss = F.cross_entropy(logits, labels)
        
        # Triplet margin loss with proper error handling
        triplet_loss = torch.tensor(0.0, device=anchor.device)
        if anchor.size(0) > 0:
            try:
                triplet_loss = F.triplet_margin_loss(anchor, positive, negative, margin=self.margin)
            except:
                triplet_loss = torch.tensor(0.0, device=anchor.device)
        
        return infonce_loss + 0.5 * triplet_loss

class DataAugmentation(nn.Module):
    def __init__(self, noise_level: float = 0.02, scale_range: Tuple[float, float] = (0.9, 1.1), dropout_rate: float = 0.05):
        super().__init__()
        self.noise_level = noise_level
        self.scale_range = scale_range  
        self.dropout_rate = dropout_rate
        
    def add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.noise_level
    
    def random_scaling(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(*self.scale_range)
        return x * scale
    
    def time_masking(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = x.shape
        mask = torch.bernoulli(torch.full((batch_size, seq_len, 1), 1 - self.dropout_rate, device=x.device))
        return x * mask
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Positive pair (weak augmentation)
        positive = self.add_gaussian_noise(x)
        positive = self.random_scaling(positive)
        
        # Negative pair (strong augmentation) 
        negative = self.add_gaussian_noise(x)
        negative = self.time_masking(negative)
        negative = self.random_scaling(negative)
        
        return positive, negative

class CARLA(nn.Module):
    def __init__(self, seq_len: int = 64, input_dim: int = 1, hidden_dim: int = 256, encoder_layers: int = 3, 
                 temperature: float = 0.1, margin: float = 1.0, dropout: float = 0.1,
                 use_focal_loss: bool = True, use_self_distill: bool = True, use_mixup: bool = True, use_swa: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.use_focal_loss = use_focal_loss
        self.use_self_distill = use_self_distill
        self.use_mixup = use_mixup
        self.use_swa = use_swa
        self.encoder = TimeSeriesEncoder(seq_len, input_dim, hidden_dim, encoder_layers, dropout)
        self.contrastive_loss = ContrastiveLoss(temperature, margin)
        self.augmentation = DataAugmentation()
        self.focal_loss = FocalLoss()
        embedding_dim = hidden_dim // 2
        self.anomaly_detector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        self.reconstructor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, seq_len * input_dim)
        )
        self.apply(self._init_weights)
        # SWA
        if self.use_swa:
            from torch.optim.swa_utils import AveragedModel
            self.swa_model = AveragedModel(self)
        else:
            self.swa_model = None
    def mixup(self, x, alpha=0.2):
        if not self.use_mixup:
            return x, x
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, x[index, :]
    def forward(self, x: torch.Tensor, mode: str = 'encode') -> torch.Tensor:
        if mode == 'encode':
            return self.encoder(x)
        elif mode == 'detect':
            embeddings = self.encoder(x)
            return self.anomaly_detector(embeddings).squeeze(-1)
        elif mode == 'reconstruct':
            embeddings = self.encoder(x)
            reconstruction = self.reconstructor(embeddings)
            return reconstruction.view(x.shape[0], self.seq_len, -1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        # Mixup 적용
        if self.use_mixup:
            x, x_shuffled = self.mixup(x)
        positive, negative = self.augmentation(x)
        anchor_emb = self.encoder(x)
        positive_emb = self.encoder(positive)
        negative_emb = self.encoder(negative)
        contrastive_loss = self.contrastive_loss(anchor_emb, positive_emb, negative_emb)
        reconstruction = self.forward(x, mode='reconstruct')
        reconstruction_loss = F.mse_loss(reconstruction, x)
        total_loss = contrastive_loss + 0.1 * reconstruction_loss
        # Focal Loss (anomaly detector)
        if self.use_focal_loss:
            anomaly_logits = self.anomaly_detector(anchor_emb)
            # 임시 타겟: reconstruction error가 상위 10%면 1, 아니면 0
            recon_err = torch.mean((x - reconstruction) ** 2, dim=[1,2] if x.ndim==3 else 1)
            threshold = torch.quantile(recon_err, 0.9)
            targets = (recon_err > threshold).float().unsqueeze(1)
            focal_loss = self.focal_loss(anomaly_logits, targets)
            total_loss = total_loss + 0.1 * focal_loss
        # Self-Distillation (EMA teacher)
        if self.use_self_distill:
            with torch.no_grad():
                teacher_emb = anchor_emb.detach()
            distill_loss = F.mse_loss(anchor_emb, teacher_emb)
            total_loss = total_loss + 0.05 * distill_loss
        return total_loss
    
    def detect_anomalies(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            anomaly_scores = self.forward(x, mode='detect')
            reconstruction = self.forward(x, mode='reconstruct')
            # x_: (배치, 시계열길이)
            x_ = x.squeeze(-1) if x.ndim == 3 and x.shape[-1] == 1 else x
            # reconstruction: (배치, 시계열길이)
            if reconstruction.ndim == 3 and reconstruction.shape[-1] == 1:
                reconstruction = reconstruction.squeeze(-1)
            elif reconstruction.ndim == 1:
                reconstruction = reconstruction.unsqueeze(1)
            # shape 강제 일치 (broadcast)
            if x_.ndim == 1:
                x_ = x_.unsqueeze(1)
            if reconstruction.ndim == 1:
                reconstruction = reconstruction.unsqueeze(1)
            if x_.shape[1] != self.seq_len:
                x_ = x_.expand(-1, self.seq_len)
            if reconstruction.shape[1] != self.seq_len:
                reconstruction = reconstruction.expand(-1, self.seq_len)
            print(f"[DEBUG] x_ shape: {x_.shape}, reconstruction shape: {reconstruction.shape}")
            assert x_.shape == reconstruction.shape, f"[ERROR] CARLA detect_anomalies shape 불일치: {reconstruction.shape} vs {x_.shape}"
            recon_error = torch.mean((x_ - reconstruction) ** 2, dim=1)
            point_scores = (x_ - reconstruction) ** 2  # (배치, 시계열길이)
            print(f"[DEBUG] point_scores shape(before): {point_scores.shape}")
            # point_scores shape 강제: (batch, seq_len)
            if point_scores.ndim == 1:
                point_scores = point_scores.unsqueeze(1)
            if point_scores.shape[1] != self.seq_len:
                point_scores = point_scores.expand(-1, self.seq_len)
            print(f"[DEBUG] point_scores shape(after): {point_scores.shape}, x.shape: {x.shape}, self.seq_len: {self.seq_len}")
            assert point_scores.shape[0] == x.shape[0] and point_scores.shape[1] == self.seq_len, f"[ERROR] CARLA point_scores shape: {point_scores.shape}, expected ({x.shape[0]}, {self.seq_len})"
            return recon_error, point_scores

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

def create_carla_model(seq_len: int = 64, **kwargs) -> CARLA:
    config = {
        'seq_len': seq_len,
        'input_dim': 1,
        'hidden_dim': 256,
        'encoder_layers': 3,
        'temperature': 0.1,
        'margin': 1.0,
        'dropout': 0.1,
    }
    config.update(kwargs)
    return CARLA(**config)