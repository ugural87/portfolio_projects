from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ..config import FusionConfig, PriceModelConfig
from .event_token_encoder import SemanticFeatureTokenEncoder
from .historical_price_backbone import HistoricalPriceSequenceEncoder
from .price_cnn_transformer import OrderedQuantileHead, YieldCurveCNNTransformer


class FomcCrossFusionModel(nn.Module):
    """Bidirectional event/price cross-attention with explicit availability masks."""

    def __init__(
        self,
        fitted_historical_backbone: YieldCurveCNNTransformer,
        price_config: PriceModelConfig,
        fusion_config: FusionConfig,
        n_features: int,
        n_rate_features: int = 2,
        event_mode: Literal["full", "rate_only"] = "full",
    ) -> None:
        super().__init__()
        self.price_config = price_config
        self.fusion_config = fusion_config
        self.event_mode = event_mode
        d_model = price_config.d_model
        if d_model % fusion_config.n_attention_heads:
            raise ValueError("Price d_model must be divisible by fusion attention heads.")
        self.price_encoder = HistoricalPriceSequenceEncoder(
            fitted_historical_backbone, fusion_config.max_intermeeting_days
        )
        self.rate_encoder = nn.Sequential(
            nn.Linear(n_rate_features, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.feature_encoder = SemanticFeatureTokenEncoder(n_features, d_model)
        self.event_to_price = nn.MultiheadAttention(
            d_model,
            fusion_config.n_attention_heads,
            dropout=price_config.dropout,
            batch_first=True,
        )
        self.event_to_price_norm = nn.LayerNorm(d_model)
        self.price_to_event = nn.MultiheadAttention(
            d_model,
            fusion_config.n_attention_heads,
            dropout=price_config.dropout,
            batch_first=True,
        )
        self.price_to_event_norm = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(price_config.dropout),
            nn.LayerNorm(d_model),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(price_config.dropout),
            nn.Linear(d_model // 2, 3),
        )
        self.quantile_head = OrderedQuantileHead(d_model, price_config.dropout)
        self._initialise_new_layers()
        if event_mode == "rate_only":
            for parameter in self.feature_encoder.parameters():
                parameter.requires_grad = False
        if fusion_config.freeze_price_encoder:
            for parameter in self.price_encoder.parameters():
                parameter.requires_grad = False

    def _initialise_new_layers(self) -> None:
        for module in (
            self.rate_encoder,
            self.feature_encoder,
            self.fusion,
            self.direction_head,
            self.quantile_head,
        ):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        for attention in (self.event_to_price, self.price_to_event):
            nn.init.xavier_uniform_(attention.in_proj_weight)
            nn.init.zeros_(attention.in_proj_bias)
            nn.init.xavier_uniform_(attention.out_proj.weight)
            nn.init.zeros_(attention.out_proj.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.fusion_config.freeze_price_encoder:
            self.price_encoder.eval()
        return self

    def forward(
        self,
        price_inputs: torch.Tensor,
        valid_mask: torch.Tensor,
        rate_values: torch.Tensor,
        feature_values: torch.Tensor,
        feature_available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.fusion_config.freeze_price_encoder:
            self.price_encoder.eval()
            with torch.no_grad():
                price_summary, price_tokens = self.price_encoder(price_inputs, valid_mask)
        else:
            price_summary, price_tokens = self.price_encoder(price_inputs, valid_mask)

        rate_token = self.rate_encoder(rate_values).unsqueeze(1)
        if self.event_mode == "rate_only":
            event_tokens = rate_token
            event_valid = torch.ones(
                (price_inputs.shape[0], 1), dtype=torch.bool, device=price_inputs.device
            )
        else:
            feature_tokens = self.feature_encoder(feature_values, feature_available)
            event_tokens = torch.cat([rate_token, feature_tokens], dim=1)
            rate_valid = torch.ones(
                (price_inputs.shape[0], 1), dtype=torch.bool, device=price_inputs.device
            )
            event_valid = torch.cat([rate_valid, feature_available.bool()], dim=1)

        price_context, event_to_price_weights = self.event_to_price(
            query=event_tokens,
            key=price_tokens,
            value=price_tokens,
            key_padding_mask=~valid_mask.bool(),
            need_weights=True,
            average_attn_weights=False,
        )
        # Invalid feature queries cannot be masked by PyTorch MHA, so neutralise them here.
        price_context = price_context * event_valid.unsqueeze(-1)
        event_to_price_weights = event_to_price_weights * event_valid[:, None, :, None]
        enriched_events = self.event_to_price_norm(event_tokens + price_context)

        event_context, price_to_event_weights = self.price_to_event(
            query=price_summary.unsqueeze(1),
            key=enriched_events,
            value=enriched_events,
            key_padding_mask=~event_valid,
            need_weights=True,
            average_attn_weights=False,
        )
        enriched_price = self.price_to_event_norm(
            price_summary + event_context.squeeze(1)
        )
        fused = self.fusion(torch.cat([price_summary, enriched_price], dim=-1))
        return {
            "direction_logits": self.direction_head(fused),
            "quantiles": self.quantile_head(fused),
            "event_to_price_attention": event_to_price_weights,
            "price_to_event_attention": price_to_event_weights,
            "event_valid_mask": event_valid,
            "price_valid_mask": valid_mask.bool(),
            "price_summary": price_summary,
            "enriched_event_tokens": enriched_events,
        }
