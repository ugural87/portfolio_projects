from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..llm.feature_contract import FOMC_FEATURE_NAMES


RATE_COLUMNS = ("actual_rate_change_bp", "fred_target_midpoint")


@dataclass(frozen=True)
class EventScalers:
    feature_mean: pd.Series
    feature_std: pd.Series
    rate_mean: pd.Series
    rate_std: pd.Series


def fit_event_scalers(events: pd.DataFrame, train_indices) -> EventScalers:
    train = events.iloc[train_indices]
    location = pd.Series(index=FOMC_FEATURE_NAMES, dtype=float)
    scale = pd.Series(index=FOMC_FEATURE_NAMES, dtype=float)
    for feature in FOMC_FEATURE_NAMES:
        usable = train[f"{feature}__available"].eq(1.0)
        observed = train.loc[usable, feature]
        if observed.empty:
            raise RuntimeError(f"No usable training observations for {feature}.")
        location[feature] = observed.mean()
        scale[feature] = observed.std(ddof=0)
    scale = scale.replace(0.0, 1.0)
    rate_mean = train[list(RATE_COLUMNS)].mean()
    rate_std = train[list(RATE_COLUMNS)].std(ddof=0).replace(0.0, 1.0)
    return EventScalers(location, scale, rate_mean, rate_std)

