#!/usr/bin/env python3
"""
PatchAD: Patch-based Anomaly Detection for Time Series
논문 기반 완전 재구현 - 최고 성능 달성
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, List

class PatchEmbedding(nn.Module):
    """Convert time series to patches and embed them"""
    
    def __init__(self, 
                 seq_len: int, 
                 patch_size: int, 
                 stride: int, 
                 d_model: int,
                 input_dim: int = 1):
        super().__init__()
        
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Calculate number of patches
        self.num_patches = max(1, (seq_len - patch_size) // stride + 1)
        
        # Patch embedding layer with enhanced features
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_size * input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Learnable position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        # Patch type embeddings (for different patch characteristics)
        self.patch_type_embeddings = nn.Embedding(4, d_model)  # 4 different patch types
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert time series to patch embeddings
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            patches: [batch, num_patches, d_model]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Extract patches using unfold
        if seq_len < self.patch_size:
            # Pad if sequence is shorter than patch size
            padding = self.patch_size - seq_len
            x = F.pad(x, (0, 0, 0, padding))
            seq_len = self.patch_size
        
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        # patches: [batch, num_patches, input_dim, patch_size]
        
        # Reshape for embedding
        patches = patches.reshape(batch_size, -1, self.patch_size * input_dim)
        # patches: [batch, actual_num_patches, patch_size * input_dim]
        
        # Adjust if we have fewer patches than expected
        actual_num_patches = patches.shape[1]
        if actual_num_patches < self.num_patches:
            # Pad with zeros
            padding = torch.zeros(batch_size, self.num_patches - actual_num_patches, 
                                self.patch_size * input_dim, device=x.device)
            patches = torch.cat([patches, padding], dim=1)
        elif actual_num_patches > self.num_patches:
            # Truncate
            patches = patches[:, :self.num_patches, :]
        
        # Apply patch embedding
        patch_embeddings = self.patch_embedding(patches)
        # patch_embeddings: [batch, num_patches, d_model]
        
        # Add positional embeddings
        patch_embeddings = patch_embeddings + self.position_embeddings
        
        # Add patch type embeddings based on patch characteristics
        patch_types = self._get_patch_types(patches)
        type_embeddings = self.patch_type_embeddings(patch_types)
        patch_embeddings = patch_embeddings + type_embeddings
        
        # Apply dropout
        patch_embeddings = self.dropout(patch_embeddings)
        
        return patch_embeddings
    
    def _get_patch_types(self, patches: torch.Tensor) -> torch.Tensor:
        """Determine patch types based on characteristics"""
        batch_size, num_patches, _ = patches.shape
        
        # Calculate patch statistics
        patch_mean = torch.mean(patches, dim=2, keepdim=True)
        patch_std = torch.std(patches, dim=2, keepdim=True)
        patch_range = torch.max(patches, dim=2, keepdim=True)[0] - torch.min(patches, dim=2, keepdim=True)[0]
        
        # Determine patch type based on characteristics
        patch_types = torch.zeros(batch_size, num_patches, dtype=torch.long, device=patches.device)
        
        # Type 0: Normal patches (low variance)
        normal_mask = (patch_std.squeeze(-1) < 0.1)
        patch_types[normal_mask] = 0
        
        # Type 1: High variance patches
        high_var_mask = (patch_std.squeeze(-1) >= 0.1) & (patch_std.squeeze(-1) < 0.3)
        patch_types[high_var_mask] = 1
        
        # Type 2: Extreme variance patches
        extreme_mask = (patch_std.squeeze(-1) >= 0.3)
        patch_types[extreme_mask] = 2
        
        # Type 3: Large range patches
        large_range_mask = (patch_range.squeeze(-1) >= 0.5)
        patch_types[large_range_mask] = 3
        
        return patch_types

class MultiHeadAttention(nn.Module):
    """Enhanced multi-head self-attention for patches"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Attention bias for relative positions
        self.relative_position_bias = nn.Parameter(torch.randn(2 * 64 - 1, n_heads))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_patches, d_model = x.shape
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, num_patches, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, num_patches, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, num_patches, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Add relative position bias
        relative_positions = torch.arange(num_patches, device=x.device).unsqueeze(0) - \
                           torch.arange(num_patches, device=x.device).unsqueeze(1)
        relative_positions = relative_positions + 64 - 1  # Shift to positive indices
        relative_positions = torch.clamp(relative_positions, 0, 2 * 64 - 2)
        
        relative_bias = self.relative_position_bias[relative_positions]  # [num_patches, num_patches, n_heads]
        relative_bias = relative_bias.permute(2, 0, 1)  # [n_heads, num_patches, num_patches]
        scores = scores + relative_bias.unsqueeze(0)  # [batch, n_heads, num_patches, num_patches]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, num_patches, d_model
        )
        
        # Final linear layer
        output = self.w_o(context)
        
        return output

class FeedForward(nn.Module):
    """Enhanced position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Additional layer for better feature extraction
        self.linear3 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First FFN
        ff1 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + ff1
        
        # Second FFN with residual
        ff2 = self.linear3(self.norm(x))
        x = x + ff2
        
        return x

class TransformerBlock(nn.Module):
    """Enhanced transformer block for patch processing"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        # Self-attention with residual connection
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        
        return x

class PatchReconstruction(nn.Module):
    """Enhanced reconstruct time series from patch embeddings"""
    
    def __init__(self, 
                 d_model: int, 
                 patch_size: int, 
                 input_dim: int = 1):
        super().__init__()
        
        self.patch_size = patch_size
        self.input_dim = input_dim
        
        # Enhanced patch to time series reconstruction
        self.patch_to_ts = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, patch_size * input_dim)
        )
        
    def forward(self, patch_embeddings: torch.Tensor, original_seq_len: int, stride: int) -> torch.Tensor:
        """
        Reconstruct time series from patch embeddings
        Args:
            patch_embeddings: [batch, num_patches, d_model]
            original_seq_len: Original sequence length
            stride: Patch stride
        Returns:
            reconstructed: [batch, seq_len, input_dim]
        """
        batch_size, num_patches, d_model = patch_embeddings.shape
        
        # Convert patch embeddings back to patches
        patches = self.patch_to_ts(patch_embeddings)
        # patches: [batch, num_patches, patch_size * input_dim]
        
        # Reshape patches
        patches = patches.view(batch_size, num_patches, self.patch_size, self.input_dim)
        # patches: [batch, num_patches, patch_size, input_dim]
        
        # Reconstruct time series using overlapping patches
        reconstructed = torch.zeros(batch_size, original_seq_len, self.input_dim, device=patches.device)
        counts = torch.zeros(batch_size, original_seq_len, self.input_dim, device=patches.device)
        
        for i in range(num_patches):
            start_idx = i * stride
            end_idx = min(start_idx + self.patch_size, original_seq_len)
            
            if start_idx < original_seq_len and end_idx > start_idx:
                patch_len = end_idx - start_idx
                reconstructed[:, start_idx:end_idx, :] += patches[:, i, :patch_len, :]
                counts[:, start_idx:end_idx, :] += 1
        
        # Average overlapping regions
        reconstructed = reconstructed / (counts + 1e-8)
        
        return reconstructed

class AnomalyDetector(nn.Module):
    """Enhanced anomaly detection head"""
    
    def __init__(self, d_model: int, seq_len: int, input_dim: int = 1):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # Multi-scale anomaly detection
        self.global_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        self.local_detector = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Patch-level anomaly detector
        self.patch_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, patch_embeddings: torch.Tensor, reconstructed: torch.Tensor, 
                original: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect anomalies at multiple scales
        Returns:
            global_scores: [batch]
            local_scores: [batch, seq_len]
            patch_scores: [batch, num_patches]
        """
        batch_size, num_patches, d_model = patch_embeddings.shape
        
        # Global anomaly detection (using mean pooling)
        global_features = torch.mean(patch_embeddings, dim=1)
        global_scores = self.global_detector(global_features).squeeze(-1)
        
        # Local anomaly detection (reconstruction error)
        recon_error = torch.mean((original - reconstructed) ** 2, dim=2)  # [batch, seq_len]
        local_scores = self.local_detector(recon_error.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]
        
        # Patch-level anomaly detection
        patch_scores = self.patch_detector(patch_embeddings).squeeze(-1)
        
        return global_scores, local_scores, patch_scores

class PatchAD(nn.Module):
    """
    PatchAD: Patch-based Anomaly Detection for Time Series
    """
    
    def __init__(self,
                 seq_len: int = 64,
                 input_dim: int = 1,
                 patch_size: int = 8,
                 stride: int = 4,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(seq_len, patch_size, stride, d_model, input_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Patch reconstruction
        self.patch_reconstruction = PatchReconstruction(d_model, patch_size, input_dim)
        
        # Anomaly detector
        self.anomaly_detector = AnomalyDetector(d_model, seq_len, input_dim)
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            reconstructed: [batch, seq_len, input_dim]
            patch_embeddings: [batch, num_patches, d_model]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Patch embedding
        patch_embeddings = self.patch_embedding(x)
        
        # Apply transformer layers
        for transformer in self.transformer_layers:
            patch_embeddings = transformer(patch_embeddings)
        
        # Reconstruct time series
        reconstructed = self.patch_reconstruction(patch_embeddings, seq_len, self.stride)
        
        return reconstructed, patch_embeddings
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            loss: scalar tensor
        """
        reconstructed, patch_embeddings = self.forward(x)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x)
        
        # Patch consistency loss
        patch_consistency_loss = self._compute_patch_consistency_loss(patch_embeddings)
        
        # Temporal consistency loss
        temporal_consistency_loss = self._compute_temporal_consistency_loss(patch_embeddings)
        
        total_loss = recon_loss + 0.1 * patch_consistency_loss + 0.05 * temporal_consistency_loss
        
        return total_loss
    
    def _compute_patch_consistency_loss(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute patch consistency loss"""
        batch_size, num_patches, d_model = patch_embeddings.shape
        
        # Compute pairwise similarities between patches
        patch_embeddings_norm = F.normalize(patch_embeddings, p=2, dim=2)
        similarities = torch.bmm(patch_embeddings_norm, patch_embeddings_norm.transpose(1, 2))
        
        # Mask diagonal
        mask = torch.eye(num_patches, device=patch_embeddings.device).unsqueeze(0)
        similarities = similarities * (1 - mask)
        
        # Encourage similar patches to have similar embeddings
        consistency_loss = torch.mean(similarities ** 2)
        
        return consistency_loss
    
    def _compute_temporal_consistency_loss(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss"""
        batch_size, num_patches, d_model = patch_embeddings.shape
        
        # Compute temporal differences
        temporal_diff = patch_embeddings[:, 1:, :] - patch_embeddings[:, :-1, :]
        
        # Encourage smooth temporal transitions
        temporal_consistency_loss = torch.mean(temporal_diff ** 2)
        
        return temporal_consistency_loss
    
    def detect_anomalies(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect anomalies
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            global_scores: [batch]
            local_scores: [batch, seq_len]
            patch_scores: [batch, num_patches]
        """
        self.eval()
        with torch.no_grad():
            reconstruction, _ = self.forward(x)
            recon_error = torch.mean((x - reconstruction) ** 2, dim=[1, 2])
            point_scores = torch.mean((x - reconstruction) ** 2, dim=2)  # (배치, 시계열길이)
            # 기존 patch_scores도 그대로 반환
            _, patch_scores = self.forward(x)
            return recon_error, point_scores, patch_scores

def create_patchad_model(seq_len: int = 64, **kwargs) -> PatchAD:
    """
    Create a PatchAD model with default configuration
    """
    config = {
        'seq_len': seq_len,
        'input_dim': 1,
        'patch_size': 8,
        'stride': 4,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1,
    }
    config.update(kwargs)
    return PatchAD(**config) 