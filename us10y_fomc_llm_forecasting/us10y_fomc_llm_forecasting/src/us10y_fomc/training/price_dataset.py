from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import PriceModelConfig
from ..features.price_features import PriceFeatureBundle
from ..features.targets import TargetTransform


class PriceWindowDataset(Dataset):
    def __init__(
        self,
        bundle: PriceFeatureBundle,
        target: TargetTransform,
        anchors: np.ndarray,
        config: PriceModelConfig,
    ) -> None:
        self.bundle = bundle
        self.target = target
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.config = config

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, item: int):
        anchor = int(self.anchors[item])
        start = anchor - self.config.lookback + 1
        window = np.transpose(self.bundle.cube[start : anchor + 1], (1, 0, 2)).copy()
        value = np.float32(self.target.to_model_units(np.asarray([anchor]))[0])
        target_class = int(self.target.to_class(np.asarray([value]))[0])
        scale = np.float32(self.target.scale_at(np.asarray([anchor]))[0])
        return (
            torch.from_numpy(window),
            torch.tensor(target_class, dtype=torch.long),
            torch.tensor(value, dtype=torch.float32),
            torch.tensor(scale, dtype=torch.float32),
            torch.tensor(anchor, dtype=torch.long),
        )


def eligible_anchors(
    bundle: PriceFeatureBundle,
    config: PriceModelConfig,
    label_before=None,
) -> np.ndarray:
    first = max(bundle.first_complete_row, config.lookback - 1)
    anchors = np.arange(first, len(bundle.panel) - config.horizon, dtype=np.int64)
    finite = np.isfinite(bundle.horizon_sigma_bp[anchors])
    anchors = anchors[finite]
    if label_before is not None:
        label_dates = bundle.panel.index[anchors + config.horizon]
        anchors = anchors[label_dates < label_before]
    if not len(anchors):
        raise RuntimeError("No eligible price anchors remain after causal filtering.")
    return anchors


def sequential_price_splits(
    anchors: np.ndarray, config: PriceModelConfig
) -> dict[str, np.ndarray]:
    n = len(anchors)
    train_end = int(n * config.fractions[0])
    validation_end = int(n * (config.fractions[0] + config.fractions[1]))
    calibration_end = int(n * sum(config.fractions))
    raw = {
        "train": anchors[:train_end],
        "validation": anchors[train_end:validation_end],
        "calibration": anchors[validation_end:calibration_end],
        "test": anchors[calibration_end:],
    }
    names = list(raw)
    for previous, current in zip(names[:-1], names[1:]):
        boundary = raw[previous][-1] + config.horizon
        raw[current] = raw[current][raw[current] - config.lookback + 1 > boundary]
        if not len(raw[current]):
            raise RuntimeError(f"Purging emptied the price {current} split.")
    return raw
