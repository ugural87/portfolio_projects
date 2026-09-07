from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import FusionConfig
from ..data.treasury_data import CHANNEL_NAMES, SERIES_IDS
from ..llm.feature_contract import FOMC_FEATURE_NAMES
from .event_scalers import RATE_COLUMNS, EventScalers
from .price_features import PriceFeatureBundle
from .targets import TargetTransform


class FomcEventDataset(Dataset):
    """One real scheduled-meeting sample with [score, confidence] semantic channels."""

    def __init__(
        self,
        events,
        indices,
        bundle: PriceFeatureBundle,
        target: TargetTransform,
        scalers: EventScalers,
        config: FusionConfig,
    ) -> None:
        self.events = events.iloc[indices].reset_index(drop=True)
        self.bundle = bundle
        self.target = target
        self.scalers = scalers
        self.config = config

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, item: int):
        event = self.events.iloc[item]
        positions = event["chunk_positions"]
        chunk = np.transpose(self.bundle.cube[positions], (1, 0, 2)).astype(np.float32, copy=True)
        padded = np.zeros(
            (len(CHANNEL_NAMES), self.config.max_intermeeting_days, len(SERIES_IDS)),
            dtype=np.float32,
        )
        valid = np.zeros(self.config.max_intermeeting_days, dtype=bool)
        start = self.config.max_intermeeting_days - chunk.shape[1]
        padded[:, start:, :] = chunk
        valid[start:] = True

        masks = np.asarray(
            [event[f"{feature}__available"] for feature in FOMC_FEATURE_NAMES],
            dtype=np.float32,
        )
        scores = np.asarray(
            [
                (event[feature] - self.scalers.feature_mean[feature])
                / self.scalers.feature_std[feature]
                for feature in FOMC_FEATURE_NAMES
            ],
            dtype=np.float32,
        )
        confidence = np.asarray(
            [event[f"{feature}__confidence"] for feature in FOMC_FEATURE_NAMES],
            dtype=np.float32,
        )
        semantic = np.stack([scores, confidence], axis=-1) * masks[:, None]
        rates = (
            (event[list(RATE_COLUMNS)] - self.scalers.rate_mean) / self.scalers.rate_std
        ).to_numpy(dtype=np.float32)

        pre_anchor = int(event.pre_anchor)
        target_position = int(event.target_position)
        delta_bp = 100.0 * (
            self.bundle.target_yield[target_position] - self.bundle.target_yield[pre_anchor]
        )
        scale_bp = float(self.target.scale_at(np.asarray([pre_anchor]))[0])
        label = np.float32(delta_bp / scale_bp)
        target_class = int(self.target.to_class(np.asarray([label]))[0])
        return (
            torch.from_numpy(padded),
            torch.from_numpy(valid),
            torch.from_numpy(rates),
            torch.from_numpy(semantic),
            torch.from_numpy(masks),
            torch.tensor(target_class, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(scale_bp, dtype=torch.float32),
            torch.tensor(item, dtype=torch.long),
        )
