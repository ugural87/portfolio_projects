from pathlib import Path

import numpy as np
import pandas as pd

from taxi_duration.config import load_config
from taxi_duration.monitoring.drift import drift_report
from taxi_duration.monitoring.profile import build_reference_profile


def test_invalid_values_are_not_hidden_by_renormalisation(
    tmp_path: Path, synthetic_trips: pd.DataFrame
) -> None:
    reference = tmp_path / "reference.json"
    build_reference_profile(
        synthetic_trips.assign(
            duration_minutes=(
                synthetic_trips["tpep_dropoff_datetime"] - synthetic_trips["tpep_pickup_datetime"]
            ).dt.total_seconds()
            / 60
        ),
        reference,
    )
    config = load_config("configs/development.yaml").monitoring
    clean_report = drift_report(synthetic_trips, reference, config)
    assert clean_report["status"] == "ok"
    assert set(clean_report["categorical"]) == {"pickup_zone", "dropoff_zone", "route"}
    assert clean_report["categorical"]["route"]["unseen_rate"] == 0.0

    corrupted = synthetic_trips.copy()
    corrupted["passenger_count"] = corrupted["passenger_count"].astype(float)
    corrupted.loc[:23, "passenger_count"] = np.nan  # 10%
    corrupted.loc[24:47, "passenger_count"] = 0  # 10%
    report = drift_report(corrupted, reference, config)

    passenger = report["features"]["passenger_count"]
    assert passenger["data_quality"]["missing_rows"] == 24
    assert passenger["data_quality"]["out_of_range_rows"] == 24
    assert passenger["data_quality"]["invalid_rate"] == 0.2
    assert passenger["status"] == "critical"
    assert report["status"] == "critical"
