from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from .price_cnn_transformer import YieldCurveCNNTransformer


class HistoricalPriceSequenceEncoder(nn.Module):
    """Reuse a pre-FOMC price encoder while preserving every daily token."""

    def __init__(self, fitted_model: YieldCurveCNNTransformer, max_days: int) -> None:
        super().__init__()
        self.surface_encoder = copy.deepcopy(fitted_model.surface_encoder)
        self.transformer = copy.deepcopy(fitted_model.transformer)
        self.cls_token = nn.Parameter(fitted_model.cls_token.detach().clone())
        old_position = fitted_model.position_embedding.detach().transpose(1, 2)
        resized = F.interpolate(
            old_position, size=max_days + 1, mode="linear", align_corners=False
        )
        self.position_embedding = nn.Parameter(resized.transpose(1, 2).clone())

    def forward(
        self, price_inputs: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        price_tokens, _ = self.surface_encoder(price_inputs)
        batch = price_inputs.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat([cls, price_tokens], dim=1)
        sequence = sequence + self.position_embedding[:, : sequence.shape[1]]
        cls_valid = torch.ones((batch, 1), dtype=torch.bool, device=valid_mask.device)
        sequence_valid = torch.cat([cls_valid, valid_mask], dim=1)
        encoded = self.transformer(sequence, src_key_padding_mask=~sequence_valid)
        return encoded[:, 0], encoded[:, 1:]
