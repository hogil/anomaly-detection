"""Asymmetric loss for single-label classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLossSingleLabel(nn.Module):
    """ASL-style loss over softmax probabilities.

    ``gamma_neg`` down-weights easy negative classes more strongly than the
    positive class. ``clip`` optionally limits easy negative confidence.
    """

    def __init__(
        self,
        weight=None,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).to(dtype=logits.dtype)
        anti_targets = 1.0 - targets_one_hot

        pos_prob = probs.clamp(min=self.eps)
        neg_prob = (1.0 - probs).clamp(min=self.eps)
        if self.clip > 0:
            neg_prob = (neg_prob + self.clip).clamp(max=1.0)

        pos_loss = targets_one_hot * torch.log(pos_prob)
        neg_loss = anti_targets * torch.log(neg_prob.clamp(min=self.eps))

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pos_weight = torch.pow(1.0 - pos_prob, self.gamma_pos)
            neg_weight = torch.pow(1.0 - neg_prob, self.gamma_neg)
            pos_loss = pos_loss * pos_weight
            neg_loss = neg_loss * neg_weight

        losses = -(pos_loss + neg_loss).sum(dim=1)
        if self.weight is not None:
            losses = losses * self.weight[targets]

        if self.reduction == "none":
            return losses
        if self.reduction == "sum":
            return losses.sum()
        return losses.mean()
