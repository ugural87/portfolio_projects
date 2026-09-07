from __future__ import annotations

import torch
from torch import nn

from ..config import FusionConfig, PriceModelConfig
from .historical_price_backbone import HistoricalPriceSequenceEncoder
from .price_cnn_transformer import OrderedQuantileHead, YieldCurveCNNTransformer


class PriceOnlyEventModel(nn.Module):
    """Meeting-horizon control that receives no policy or minutes information."""

    def __init__(
        self,
        fitted_historical_backbone: YieldCurveCNNTransformer,
        price_config: PriceModelConfig,
        fusion_config: FusionConfig,
    ) -> None:
        super().__init__()
        self.freeze_price_encoder = fusion_config.freeze_price_encoder
        self.price_encoder = HistoricalPriceSequenceEncoder(
            fitted_historical_backbone, fusion_config.max_intermeeting_days
        )
        d_model = price_config.d_model
        self.direction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(price_config.dropout),
            nn.Linear(d_model // 2, 3),
        )
        self.quantile_head = OrderedQuantileHead(d_model, price_config.dropout)
        for module in (self.direction_head, self.quantile_head):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        if self.freeze_price_encoder:
            for parameter in self.price_encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_price_encoder:
            self.price_encoder.eval()
        return self

    def forward(
        self,
        price_inputs: torch.Tensor,
        valid_mask: torch.Tensor,
        rate_values: torch.Tensor | None = None,
        feature_values: torch.Tensor | None = None,
        feature_available: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.freeze_price_encoder:
            self.price_encoder.eval()
            with torch.no_grad():
                summary, _ = self.price_encoder(price_inputs, valid_mask)
        else:
            summary, _ = self.price_encoder(price_inputs, valid_mask)
        return {
            "direction_logits": self.direction_head(summary),
            "quantiles": self.quantile_head(summary),
        }
