#!/usr/bin/env python3
"""
TraceGPT: GPT-Style Transformer for Time Series Anomaly Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False) 
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            # Ensure mask has the correct shape
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(context)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Enhanced transformer block with pre-norm architecture and additional features"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Additional cross-attention for better feature interaction
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Gated residual connection
        self.gate = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture with enhanced features
        # Self-attention with gated residual connection
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, mask)
        gate = torch.sigmoid(self.gate(x))
        x = x + gate * self.dropout(attn_output)
        
        # Cross-attention for enhanced feature interaction
        norm_x = self.norm2(x)
        cross_output = self.cross_attention(norm_x, mask)
        x = x + self.dropout(cross_output)
        
        # Feed-forward with gated residual connection
        norm_x = self.norm3(x)
        ff_output = self.feed_forward(norm_x)
        gate = torch.sigmoid(self.gate(x))
        x = x + gate * self.dropout(ff_output)
        
        return x

class TraceGPT(nn.Module):
    def __init__(self, seq_len: int = 64, input_dim: int = 1, d_model: int = 256, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 1024, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive modeling"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def forward(self, x: torch.Tensor, use_causal_mask: bool = True) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Input embedding
        x = self.input_embedding(x)
        
        # Positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]
        
        x = self.dropout(x)
        
        # Create causal mask if needed
        mask = None
        if use_causal_mask:
            mask = self.create_causal_mask(seq_len, device)
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        
        # Final normalization and projection
        x = self.final_norm(x)
        reconstructed = self.output_projection(x)
        
        return reconstructed
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute autoregressive loss"""
        if x.shape[1] <= 1:
            # If sequence is too short, use reconstruction loss instead
            x_recon = self.forward(x, use_causal_mask=False)
            return F.mse_loss(x_recon, x)
        
        # Autoregressive loss: predict next token
        input_seq = x[:, :-1, :]  # All but last token
        target_seq = x[:, 1:, :]  # All but first token
        
        # Forward pass with causal masking
        predicted = self.forward(input_seq, use_causal_mask=True)
        
        # Compute loss
        return F.mse_loss(predicted, target_seq)
    
    def detect_anomalies(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Anomaly detection using reconstruction error"""
        self.eval()
        with torch.no_grad():
            # Use non-causal reconstruction for anomaly detection
            x_recon = self.forward(x, use_causal_mask=False)
            
            # Series-level reconstruction error
            recon_error = torch.mean((x - x_recon) ** 2, dim=[1, 2])
            
            # Point-wise reconstruction error
            point_scores = torch.mean((x - x_recon) ** 2, dim=2)  # (배치, 시계열길이)
            
            # Also try autoregressive prediction if sequence is long enough
            if x.shape[1] > 1:
                try:
                    input_seq = x[:, :-1, :]
                    target_seq = x[:, 1:, :]
                    predicted = self.forward(input_seq, use_causal_mask=True)
                    
                    # Autoregressive error
                    ar_error = torch.mean((predicted - target_seq) ** 2, dim=[1, 2])
                    
                    # Combine reconstruction and autoregressive errors
                    combined_error = 0.7 * recon_error + 0.3 * ar_error
                    
                    # Point-wise autoregressive error (pad with zeros for first timestep)
                    ar_point_error = torch.mean((predicted - target_seq) ** 2, dim=2)
                    ar_point_padded = F.pad(ar_point_error, (1, 0), value=0.0)
                    
                    # Combine point scores
                    combined_point_scores = 0.7 * point_scores + 0.3 * ar_point_padded
                    
                    return combined_error, combined_point_scores
                except:
                    # Fall back to reconstruction only
                    pass
        
        return recon_error, point_scores

def create_tracegpt_model(seq_len: int = 64, **kwargs) -> TraceGPT:
    """Create TraceGPT model with optimized hyperparameters"""
    config = {
        'seq_len': seq_len,
        'input_dim': 1,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1,
        'max_len': 512,
    }
    config.update(kwargs)
    return TraceGPT(**config)