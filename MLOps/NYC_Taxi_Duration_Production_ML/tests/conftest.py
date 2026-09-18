from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_trips() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 240
    pickup = pd.date_range("2025-01-01", periods=rows, freq="30min")
    pickup_zone = rng.integers(1, 50, rows)
    dropoff_zone = rng.integers(1, 50, rows)
    passenger = rng.integers(1, 5, rows)
    duration = 6 + np.abs(dropoff_zone - pickup_zone) * 0.45 + pickup.hour * 0.12
    duration += rng.normal(0, 1.0, rows)
    duration = np.clip(duration, 1.1, 60)
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup + pd.to_timedelta(duration, unit="m"),
            "PULocationID": pickup_zone,
            "DOLocationID": dropoff_zone,
            "passenger_count": passenger,
        }
    )


@pytest.fixture()
def aware_pickup() -> datetime:
    return datetime.fromisoformat("2025-01-15T08:30:00-05:00")
