from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from taxi_duration.config import SplitConfig


@dataclass(frozen=True)
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(frame: pd.DataFrame, config: SplitConfig) -> DataSplit:
    if len(frame) < 30:
        raise ValueError("at least 30 valid rows are required")
    train_end = int(len(frame) * config.train_fraction)
    validation_end = int(len(frame) * (config.train_fraction + config.validation_fraction))
    return DataSplit(
        train=frame.iloc[:train_end].copy(),
        validation=frame.iloc[train_end:validation_end].copy(),
        test=frame.iloc[validation_end:].copy(),
    )
