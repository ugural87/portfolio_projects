import numpy as np
import pandas as pd

from taxi_duration.features.builder import FEATURE_COLUMNS, TripFeatureBuilder


def test_features_are_finite_and_leakage_safe(synthetic_trips: pd.DataFrame) -> None:
    transformed = TripFeatureBuilder().fit_transform(synthetic_trips)
    assert tuple(transformed.columns) == FEATURE_COLUMNS
    assert "tpep_dropoff_datetime" not in transformed.columns
    assert "duration_minutes" not in transformed.columns
    assert np.isfinite(transformed.to_numpy()).all()


def test_cyclical_hour_wraps_smoothly() -> None:
    frame = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 12:00"]),
            "PULocationID": [1, 1],
            "DOLocationID": [2, 2],
            "passenger_count": [1, 1],
        }
    )
    result = TripFeatureBuilder().fit_transform(frame)
    assert np.isclose(result.loc[0, "hour_cos"], 1.0)
    assert np.isclose(result.loc[1, "hour_cos"], -1.0)
