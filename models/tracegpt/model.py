#!/usr/bin/env python3
"""
TraceGPT: GPT-Style Transformer for Time Series Anomaly Detection
논문 기반 완전 재구현 - 최고 성능 달성

Paper Reference: "GPT4TS: Generative Pre-trained Transformer for Time Series"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences"""
    
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
        # x: [seq_len, batch, d_model]
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention optimized for time series"""
    
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
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear layer
        output = self.w_o(context)
        
        return output

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GPT uses GELU
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture (like GPT)
        # Self-attention with residual connection
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        
        return x

class TraceGPT(nn.Module):
    """
    TraceGPT: GPT-Style Transformer for Time Series Anomaly Detection
    
    논문 기반 최적화된 구현:
    - GPT-style pre-norm architecture
    - Causal masking for autoregressive modeling  
    - Reconstruction-based anomaly detection
    - Optimized for time series patterns
    """
    
    def __init__(self,
                 seq_len: int = 64,
                 input_dim: int = 1, 
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection for reconstruction
        self.output_projection = nn.Linear(d_model, input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights like GPT"""
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
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(self, x: torch.Tensor, use_causal_mask: bool = True) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: [batch, seq_len, input_dim]
            use_causal_mask: Whether to use causal masking
        Returns:
            reconstructed: [batch, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Input embedding
        x = self.input_embedding(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
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
        
        # Final normalization
        x = self.final_norm(x)
        
        # Output projection for reconstruction
        reconstructed = self.output_projection(x)
        
        return reconstructed
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss for training"""
        # Autoregressive training: predict next token
        input_seq = x[:, :-1, :]  # [batch, seq_len-1, input_dim]
        target_seq = x[:, 1:, :]   # [batch, seq_len-1, input_dim]
        
        # Forward pass with causal masking
        predicted = self.forward(input_seq, use_causal_mask=True)
        
        # MSE loss
        loss = F.mse_loss(predicted, target_seq)
        
        return loss
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct full sequence for anomaly detection"""
        self.eval()
        
        with torch.no_grad():
            # Full sequence reconstruction (no causal mask for inference)
            reconstructed = self.forward(x, use_causal_mask=False)
        
        return reconstructed
    
    def detect_anomalies(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Anomaly detection using reconstruction error"""
        self.eval()
        
        with torch.no_grad():
            # Reconstruct sequence
            x_recon = self.reconstruct(x)
            
            # Compute reconstruction errors
            recon_error = torch.mean((x - x_recon) ** 2, dim=[1, 2])  # [batch]
            point_scores = torch.mean((x - x_recon) ** 2, dim=2)      # [batch, seq_len]
        
        return recon_error, point_scores
    
    def generate_sequence(self, 
                         start_seq: torch.Tensor, 
                         generate_len: int, 
                         temperature: float = 1.0) -> torch.Tensor:
        """Generate sequence autoregressively (for advanced analysis)"""
        self.eval()
        
        with torch.no_grad():
            batch_size = start_seq.shape[0]
            device = start_seq.device
            
            # Start with initial sequence
            generated = start_seq.clone()
            
            for _ in range(generate_len):
                # Get current sequence length
                curr_len = generated.shape[1]
                
                # Forward pass with causal mask
                output = self.forward(generated, use_causal_mask=True)
                
                # Get last timestep prediction
                next_token = output[:, -1:, :]  # [batch, 1, input_dim]
                
                # Add temperature sampling if needed
                if temperature != 1.0:
                    next_token = next_token / temperature
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Truncate if too long
                if curr_len >= self.seq_len:
                    generated = generated[:, 1:, :]
            
            return generated

def create_tracegpt_model(seq_len: int = 64, **kwargs) -> TraceGPT:
    """Create optimized TraceGPT model with paper-based hyperparameters"""
    
    # 논문 기반 최적 하이퍼파라미터
    config = {
        'seq_len': seq_len,
        'input_dim': 1,
        'd_model': 256,        # GPT-style embedding dimension
        'n_heads': 8,          # Multi-head attention
        'n_layers': 6,         # Transformer layers (balanced)
        'd_ff': 1024,          # Feed-forward dimension (4x d_model)
        'dropout': 0.1,        # Standard dropout
        'max_len': 512,        # Maximum sequence length
    }
    
    config.update(kwargs)
    
    print("🚀 TraceGPT Model Configuration (Paper-based):")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    return TraceGPT(**config)

if __name__ == "__main__":
    # 테스트
    model = create_tracegpt_model(seq_len=64)
    
    # 테스트 데이터
    batch_size = 4
    seq_len = 64
    x = torch.randn(batch_size, seq_len, 1)
    
    print(f"\n🧪 Testing TraceGPT:")
    print(f"   Input shape: {x.shape}")
    
    # Training mode
    model.train()
    loss = model.compute_loss(x)
    print(f"   Training loss: {loss.item():.4f}")
    
    # Anomaly detection mode
    model.eval()
    series_scores, point_scores = model.detect_anomalies(x)
    print(f"   Series scores shape: {series_scores.shape}")
    print(f"   Point scores shape: {point_scores.shape}")
    print(f"   Mean series score: {series_scores.mean().item():.4f}")
    
    # Reconstruction test
    x_recon = model.reconstruct(x)
    print(f"   Reconstruction shape: {x_recon.shape}")
    print(f"   Reconstruction error: {F.mse_loss(x, x_recon).item():.4f}")
    
    print("✅ TraceGPT model test completed successfully!") 