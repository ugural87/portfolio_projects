from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from taxi_duration.data.validation import PASSENGERS, PICKUP

INFERENCE_COLUMNS: Final = (PICKUP, "PULocationID", "DOLocationID", PASSENGERS)
# Zone identifiers are nominal: TLC LocationIDs follow the alphabetical order of zone names,
# so their numeric order carries no geography. They are encoded downstream (models.factory).
ZONE_COLUMNS: Final = ("route_id", "pickup_zone_id", "dropoff_zone_id")
ROUTE_MULTIPLIER: Final = 1000
FEATURE_COLUMNS: Final = (
    "route_id",
    "pickup_zone_id",
    "dropoff_zone_id",
    "passenger_count",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "is_rush_hour",
)


class TripFeatureBuilder(BaseEstimator, TransformerMixin):
    """Deterministic features using only fields available at trip start."""

    def fit(self, X: pd.DataFrame, y: object = None) -> TripFeatureBuilder:
        self.feature_names_in_ = np.asarray(INFERENCE_COLUMNS, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(INFERENCE_COLUMNS) - set(X.columns))
        if missing:
            raise ValueError(f"missing inference columns: {missing}")
        pickup = pd.to_datetime(X[PICKUP], errors="raise")
        hour = pickup.dt.hour + pickup.dt.minute / 60.0
        weekday = pickup.dt.dayofweek
        result = pd.DataFrame(index=X.index)
        pickup_zone = pd.to_numeric(X["PULocationID"], errors="raise").astype("int64")
        dropoff_zone = pd.to_numeric(X["DOLocationID"], errors="raise").astype("int64")
        result["route_id"] = pickup_zone * ROUTE_MULTIPLIER + dropoff_zone
        result["pickup_zone_id"] = pickup_zone
        result["dropoff_zone_id"] = dropoff_zone
        result["passenger_count"] = pd.to_numeric(X[PASSENGERS], errors="raise")
        result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        result["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
        result["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
        result["is_weekend"] = (weekday >= 5).astype(int)
        result["is_rush_hour"] = ((hour.between(7, 10)) | (hour.between(16, 20))).astype(int)
        return result[list(FEATURE_COLUMNS)]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(FEATURE_COLUMNS, dtype=object)
