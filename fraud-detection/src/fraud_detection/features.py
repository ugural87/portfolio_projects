from __future__ import annotations

import numpy as np
import pandas as pd


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"Time", "Amount", *[f"V{i}" for i in range(1, 29)]}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    hour = (frame["Time"].to_numpy(dtype=float) / 3600.0) % 24.0
    features = frame[[*[f"V{i}" for i in range(1, 29)], "Amount"]].copy()
    features["LogAmount"] = np.log1p(features["Amount"].clip(lower=0))
    features["HourSin"] = np.sin(2 * np.pi * hour / 24.0)
    features["HourCos"] = np.cos(2 * np.pi * hour / 24.0)
    return features


def target(frame: pd.DataFrame) -> np.ndarray:
    return frame["Class"].to_numpy(dtype=np.int8)

