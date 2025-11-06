import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, reduction: str | None = "mean"):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)  # (batch,)
        p_t = torch.exp(-ce_loss)
        loss = ((1 - p_t) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss