from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def audit_fusion_forward(model, batch, n_features: int, n_heads: int) -> dict[str, object]:
    model.eval()
    price, price_mask, rates, semantic, semantic_mask, *_ = batch
    device = next(model.parameters()).device
    outputs = model(
        price.to(device),
        price_mask.to(device),
        rates.to(device),
        semantic.to(device),
        semantic_mask.to(device),
    )
    batch_size, _, n_days, _ = price.shape
    n_event_tokens = 1 if getattr(model, "event_mode", "full") == "rate_only" else 1 + n_features
    expected = {
        "direction_logits": (batch_size, 3),
        "quantiles": (batch_size, 3),
        "event_to_price_attention": (batch_size, n_heads, n_event_tokens, n_days),
        "price_to_event_attention": (batch_size, n_heads, 1, n_event_tokens),
    }
    for key, shape in expected.items():
        if tuple(outputs[key].shape) != shape:
            raise AssertionError(f"{key} shape {tuple(outputs[key].shape)} != {shape}")
    quantiles = outputs["quantiles"].cpu().numpy()
    if not ((quantiles[:, 0] <= quantiles[:, 1]) & (quantiles[:, 1] <= quantiles[:, 2])).all():
        raise AssertionError("Quantile ordering failed.")
    e2p = outputs["event_to_price_attention"].cpu().numpy()
    price_valid = outputs["price_valid_mask"].cpu().numpy().astype(bool)
    for sample in range(batch_size):
        if not np.allclose(e2p[sample, :, :, ~price_valid[sample]], 0.0, atol=1e-7):
            raise AssertionError("Price padding received non-zero cross-attention.")
    if n_event_tokens > 1:
        p2e = outputs["price_to_event_attention"].cpu().numpy()
        event_valid = outputs["event_valid_mask"].cpu().numpy().astype(bool)
        for sample in range(batch_size):
            if not np.allclose(
                e2p[sample, :, ~event_valid[sample], :], 0.0, atol=1e-7
            ):
                raise AssertionError("Unavailable event queries retained non-zero attention.")
            if not np.allclose(p2e[sample, :, :, ~event_valid[sample]], 0.0, atol=1e-7):
                raise AssertionError("Unavailable event tokens received non-zero attention.")
    return {"passed": True, "shapes": expected}


def audit_frozen_price_encoder(model) -> dict[str, object]:
    parameters = list(model.price_encoder.parameters())
    if not parameters or any(parameter.requires_grad for parameter in parameters):
        raise AssertionError("The historical price encoder is not fully frozen.")
    return {
        "passed": True,
        "frozen_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
