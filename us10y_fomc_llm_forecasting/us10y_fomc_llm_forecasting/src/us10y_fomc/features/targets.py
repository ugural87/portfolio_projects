from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import PriceModelConfig
from .price_features import PriceFeatureBundle


@dataclass(frozen=True)
class TargetTransform:
    bundle: PriceFeatureBundle
    config: PriceModelConfig
    raw_target_scale_bp: float = 1.0

    def future_change_bp(self, anchors: np.ndarray) -> np.ndarray:
        return 100.0 * (
            self.bundle.target_yield[anchors + self.config.horizon]
            - self.bundle.target_yield[anchors]
        )

    def scale_at(self, anchors: np.ndarray) -> np.ndarray:
        if not self.config.vol_scaled_target:
            return np.full(len(anchors), self.raw_target_scale_bp, dtype=np.float64)
        scale = self.bundle.horizon_sigma_bp[anchors]
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Non-positive or missing target volatility scale.")
        return scale

    def to_model_units(self, anchors: np.ndarray) -> np.ndarray:
        return self.future_change_bp(anchors) / self.scale_at(anchors)

    def flat_band(self) -> float:
        return (
            self.config.flat_threshold_sigma
            if self.config.vol_scaled_target
            else self.config.flat_threshold_bp
        )

    def to_class(self, values: np.ndarray) -> np.ndarray:
        threshold = self.flat_band()
        return np.where(values < -threshold, 0, np.where(values > threshold, 2, 1)).astype(
            np.int64
        )

