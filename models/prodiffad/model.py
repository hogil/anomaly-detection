#!/usr/bin/env python3
"""
ProDiffAD: 간단한 Diffusion Model for Time Series Anomaly Detection
복잡한 UNet 대신 간단한 구조로 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleDenoiseNetwork(nn.Module):
    """간단한 denoising network"""
    
    def __init__(self, 
                 input_dim: int = 1,
                 hidden_dim: int = 128,
                 time_emb_dim: int = 128):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 3, padding=1)
        
        # Enhanced main processing blocks with more layers
        self.blocks = nn.ModuleList([
            self._make_block(hidden_dim, hidden_dim, time_emb_dim),
            self._make_block(hidden_dim, hidden_dim * 2, time_emb_dim),
            self._make_block(hidden_dim * 2, hidden_dim * 2, time_emb_dim),
            self._make_block(hidden_dim * 2, hidden_dim * 2, time_emb_dim),  # Additional layer
            self._make_block(hidden_dim * 2, hidden_dim, time_emb_dim),
            self._make_block(hidden_dim, hidden_dim, time_emb_dim),
            self._make_block(hidden_dim, hidden_dim, time_emb_dim)  # Additional layer
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, input_dim, 3, padding=1)
        )
        
    def _make_block(self, in_ch: int, out_ch: int, time_emb_dim: int):
        """Create a processing block"""
        return nn.ModuleDict({
            'time_mlp': nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch)
            ),
            'conv1': nn.Conv1d(in_ch, out_ch, 3, padding=1),
            'norm1': nn.BatchNorm1d(out_ch),
            'conv2': nn.Conv1d(out_ch, out_ch, 3, padding=1),
            'norm2': nn.BatchNorm1d(out_ch),
            'shortcut': nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        })
    
    def _apply_block(self, x: torch.Tensor, time_emb: torch.Tensor, block: nn.ModuleDict) -> torch.Tensor:
        """Apply a processing block"""
        h = F.silu(block['norm1'](block['conv1'](x)))
        
        # Add time embedding
        time_emb_proj = block['time_mlp'](time_emb)
        h = h + time_emb_proj.unsqueeze(-1)
        
        h = F.silu(block['norm2'](block['conv2'](h)))
        
        return h + block['shortcut'](x)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim, seq_len]
            timesteps: [batch]
        Returns:
            noise: [batch, input_dim, seq_len]
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Input projection
        h = self.input_proj(x)
        
        # Apply blocks
        for block in self.blocks:
            h = self._apply_block(h, time_emb, block)
        
        # Output projection
        return self.output_proj(h)

class ProDiffAD(nn.Module):
    """
    ProDiffAD: 간단한 Diffusion Model for Time Series Anomaly Detection
    """
    
    def __init__(self,
                 seq_len: int = 64,
                 input_dim: int = 1,
                 hidden_dim: int = 128,
                 time_emb_dim: int = 128,
                 num_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps
        
        # Simple denoising network
        self.denoise_net = SimpleDenoiseNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim
        )
        
        # Diffusion schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute diffusion training loss"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.denoise_net(x_noisy, t)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute training loss"""
        # Convert to [batch, input_dim, seq_len] format
        x = x.transpose(1, 2)
        
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        return self.p_losses(x, t)
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Single reverse diffusion step"""
        betas_t = self.betas[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).reshape(-1, 1, 1)
        
        # Predict noise
        predicted_noise = self.denoise_net(x, t)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.betas[t].reshape(-1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
        """Reconstruct input through partial diffusion"""
        # Convert to [batch, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        batch_size = x.shape[0]
        device = x.device
        
        # Add noise to a moderate level
        t = torch.full((batch_size,), min(num_steps, self.num_timesteps - 1), device=device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        
        # Denoise back
        for i in reversed(range(0, min(num_steps, self.num_timesteps))):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x_noisy = self.p_sample(x_noisy, t)
        
        # Convert back to [batch, seq_len, input_dim]
        return x_noisy.transpose(1, 2)
    
    def detect_anomalies(self, x: torch.Tensor, num_reconstructions: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Anomaly detection using reconstruction error"""
        self.eval()
        
        reconstruction_errors = []
        point_errors = []
        
        # Multiple reconstructions for robustness
        for _ in range(num_reconstructions):
            x_recon = self.reconstruct(x, num_steps=50)
            error = torch.mean((x - x_recon) ** 2, dim=[1, 2])  # [batch]
            point_error = torch.mean((x - x_recon) ** 2, dim=2)  # (배치, 시계열길이)
            reconstruction_errors.append(error)
            point_errors.append(point_error)
        
        # Average reconstruction errors
        series_scores = torch.stack(reconstruction_errors).mean(dim=0)
        point_scores = torch.stack(point_errors).mean(dim=0)
        
        return series_scores, point_scores

def create_prodiffad_model(seq_len: int = 64, **kwargs) -> ProDiffAD:
    """Create simple ProDiffAD model"""
    
    config = {
        'seq_len': seq_len,
        'input_dim': 1,
        'hidden_dim': 128,
        'time_emb_dim': 128,
        'num_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
    }
    
    config.update(kwargs)
    return ProDiffAD(**config)