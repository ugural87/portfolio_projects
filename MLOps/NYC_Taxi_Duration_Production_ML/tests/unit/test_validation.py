import pandas as pd

from taxi_duration.config import load_config
from taxi_duration.data.validation import validate_and_clean


def test_validation_rejects_impossible_duration(synthetic_trips: pd.DataFrame) -> None:
    bad = synthetic_trips.iloc[[0]].copy()
    bad["tpep_dropoff_datetime"] = bad["tpep_pickup_datetime"]
    frame = pd.concat([synthetic_trips, bad], ignore_index=True)
    clean, report = validate_and_clean(frame, load_config("configs/development.yaml").data)
    assert len(clean) == len(synthetic_trips)
    assert report.rejected_rows == 1


def test_validation_rejects_missing_columns(synthetic_trips: pd.DataFrame) -> None:
    config = load_config("configs/development.yaml").data
    try:
        validate_and_clean(synthetic_trips.drop(columns="PULocationID"), config)
    except ValueError as error:
        assert "PULocationID" in str(error)
    else:
        raise AssertionError("missing schema column should fail")


def test_validation_drops_and_reports_exact_duplicates(synthetic_trips: pd.DataFrame) -> None:
    frame = pd.concat([synthetic_trips, synthetic_trips.iloc[:2]], ignore_index=True)
    clean, report = validate_and_clean(frame, load_config("configs/development.yaml").data)
    assert len(clean) == len(synthetic_trips)
    assert report.duplicate_rows == 2
    assert report.duplicate_rate == 2 / len(frame)
