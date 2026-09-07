from __future__ import annotations

import torch
from torch.nn import functional as F

from ..config import PriceModelConfig


def pinball_loss(
    predictions: torch.Tensor, targets: torch.Tensor, quantiles: tuple[float, ...]
) -> torch.Tensor:
    errors = targets.unsqueeze(1) - predictions
    q = torch.tensor(quantiles, dtype=predictions.dtype, device=predictions.device)
    return torch.maximum(q * errors, (q - 1.0) * errors).mean()


def joint_forecast_loss(
    outputs: dict[str, torch.Tensor],
    target_class: torch.Tensor,
    target_value: torch.Tensor,
    config: PriceModelConfig,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    classification = F.cross_entropy(
        outputs["direction_logits"], target_class, weight=class_weight
    )
    regression = pinball_loss(outputs["quantiles"], target_value, config.quantiles)
    return classification + config.regression_loss_weight * regression
