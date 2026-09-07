from __future__ import annotations

import torch
from torch import nn


class SemanticFeatureTokenEncoder(nn.Module):
    """Map each [standardised score, confidence] pair to an identifiable token."""

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.value_projection = nn.Linear(2, d_model)
        self.feature_identity = nn.Parameter(torch.empty(1, n_features, d_model))
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.feature_identity, std=0.02)

    def forward(self, values: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        projected = self.value_projection(values * available.unsqueeze(-1))
        tokens = projected + self.feature_identity
        return self.norm(tokens)
