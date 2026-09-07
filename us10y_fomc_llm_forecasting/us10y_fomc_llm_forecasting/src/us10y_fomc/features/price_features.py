from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import PriceModelConfig
from ..data.treasury_data import CHANNEL_NAMES, SERIES_IDS, TARGET_SERIES


@dataclass(frozen=True)
class PriceFeatureBundle:
    panel: pd.DataFrame
    cube: np.ndarray
    target_yield: np.ndarray
    horizon_sigma_bp: np.ndarray
    first_complete_row: int


def build_raw_feature_cube(panel: pd.DataFrame, config: PriceModelConfig) -> np.ndarray:
    levels = panel.loc[:, SERIES_IDS]
    change_1d = 100.0 * levels.diff(1)
    change_5d = 100.0 * levels.diff(config.horizon)
    realised_vol = (100.0 * levels.diff(1)).rolling(
        config.vol_window, min_periods=config.vol_window
    ).std()
    return np.stack(
        [
            levels.to_numpy(dtype=np.float64),
            change_1d.to_numpy(dtype=np.float64),
            change_5d.to_numpy(dtype=np.float64),
            realised_vol.to_numpy(dtype=np.float64),
        ],
        axis=1,
    )


def causal_rolling_standardise(cube: np.ndarray, config: PriceModelConfig) -> np.ndarray:
    """Use only the trailing window ending at the observed input date."""
    n_rows, n_channels, n_maturities = cube.shape
    flat = pd.DataFrame(cube.reshape(n_rows, n_channels * n_maturities))
    rolling = flat.rolling(config.norm_window, min_periods=config.norm_min_periods)
    mean = rolling.mean()
    std = rolling.std(ddof=0).replace(0.0, np.nan)
    standardized = (flat - mean) / std
    return standardized.to_numpy(dtype=np.float64).reshape(
        n_rows, n_channels, n_maturities
    )


def build_price_feature_bundle(
    panel: pd.DataFrame, config: PriceModelConfig
) -> PriceFeatureBundle:
    missing = set(SERIES_IDS) - set(panel.columns)
    if missing:
        raise ValueError(f"Treasury panel is missing series: {sorted(missing)}")
    raw = build_raw_feature_cube(panel, config)
    standardized = causal_rolling_standardise(raw, config)
    complete = np.isfinite(standardized).all(axis=(1, 2))
    first_complete = int(np.argmax(complete))
    if not complete[first_complete:].all():
        raise ValueError("Feature cube contains interior gaps after its warm-up period.")
    cube = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    target = panel[TARGET_SERIES].to_numpy(dtype=np.float32)
    daily_change_bp = 100.0 * panel[TARGET_SERIES].diff(1)
    horizon_sigma = (
        daily_change_bp.rolling(config.vol_window, min_periods=config.vol_window)
        .std()
        .to_numpy(dtype=np.float64)
        * np.sqrt(config.horizon)
    )
    if cube.shape[1:] != (len(CHANNEL_NAMES), len(SERIES_IDS)):
        raise AssertionError("Unexpected price-feature cube dimensions.")
    return PriceFeatureBundle(panel, cube, target, horizon_sigma, first_complete)

