from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..config import PriceModelConfig
from ..data.treasury_data import CHANNEL_NAMES, SERIES_IDS


class ConvNormGELU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class YieldSurfaceEncoder(nn.Module):
    """Encode the time-by-maturity yield surface into one token per day."""

    def __init__(self, config: PriceModelConfig) -> None:
        super().__init__()
        self.config = config
        branch_channels = config.cnn_channels // 2
        self.short_branch = ConvNormGELU(
            len(CHANNEL_NAMES), branch_channels, (3, 3), config.dropout
        )
        self.medium_branch = ConvNormGELU(
            len(CHANNEL_NAMES), branch_channels, (7, 3), config.dropout
        )
        self.refine = ConvNormGELU(
            config.cnn_channels, config.cnn_channels, (3, 3), config.dropout
        )
        n_maturities = len(SERIES_IDS)
        if config.pool_mode == "rank1":
            self.maturity_attention = nn.Conv2d(config.cnn_channels, 1, kernel_size=1)
            token_input_dim = config.cnn_channels
        elif config.pool_mode == "multihead":
            self.maturity_attention = nn.Conv2d(
                config.cnn_channels, config.pool_heads, kernel_size=1
            )
            token_input_dim = config.cnn_channels * config.pool_heads
        else:
            # Under flatten pooling no unused pseudo-attention layer is created.
            self.maturity_attention = None
            token_input_dim = config.cnn_channels * n_maturities
        self.token_projection = nn.Sequential(
            nn.Linear(token_input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        local = torch.cat([self.short_branch(inputs), self.medium_branch(inputs)], dim=1)
        features = local + self.refine(local)
        if self.config.pool_mode == "rank1":
            attention = torch.softmax(self.maturity_attention(features), dim=-1)
            pooled = (features * attention).sum(dim=-1).transpose(1, 2)
            attention_out = attention.mean(dim=1)
        elif self.config.pool_mode == "multihead":
            attention = torch.softmax(self.maturity_attention(features), dim=-1)
            pooled = torch.einsum("bctm,bhtm->bthc", features, attention).flatten(2)
            attention_out = attention.mean(dim=1)
        else:
            pooled = features.permute(0, 2, 1, 3).flatten(2)
            attention_out = None
        return self.token_projection(pooled), attention_out


class OrderedQuantileHead(nn.Module):
    """Produce non-crossing q05, q50 and q95 estimates by construction."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.median = nn.Linear(d_model // 2, 1)
        self.widths = nn.Linear(d_model // 2, 2)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden(context)
        median = self.median(hidden)
        widths = F.softplus(self.widths(hidden)) + 1e-4
        return torch.cat(
            [median - widths[:, :1], median, median + widths[:, 1:]], dim=-1
        )


class YieldCurveCNNTransformer(nn.Module):
    def __init__(self, config: PriceModelConfig) -> None:
        super().__init__()
        self.config = config
        self.surface_encoder = YieldSurfaceEncoder(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.lookback + 1, config.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.direction_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 3),
        )
        self.quantile_head = OrderedQuantileHead(config.d_model, config.dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def encode_sequence(
        self, inputs: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        tokens, maturity_attention = self.surface_encoder(inputs)
        batch = inputs.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        sequence = sequence + self.position_embedding[:, : sequence.shape[1]]
        padding_mask = None
        if valid_mask is not None:
            cls_valid = torch.ones((batch, 1), dtype=torch.bool, device=inputs.device)
            padding_mask = ~torch.cat([cls_valid, valid_mask], dim=1)
        encoded = self.transformer(sequence, src_key_padding_mask=padding_mask)
        return encoded[:, 0], encoded[:, 1:], maturity_attention

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        context, _, maturity_attention = self.encode_sequence(inputs)
        return {
            "direction_logits": self.direction_head(context),
            "quantiles": self.quantile_head(context),
            "maturity_attention": maturity_attention,
        }
