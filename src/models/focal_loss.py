"""Focal Loss for multi-class classification"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)"""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal

        if self.reduction == "none":
            return focal
        return focal.mean()
