from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from taxi_duration.cli import default_period, period_window
from taxi_duration.config import load_config
from taxi_duration.data.validation import Period, validate_and_clean
from taxi_duration.training.split import chronological_split


@pytest.mark.parametrize("text", ["2025-1", "2025-01"])
def test_period_parse_normalises_month(text: str) -> None:
    assert str(Period.parse(text)) == "2025-01"


def test_period_from_tlc_filename() -> None:
    assert Period.from_filename(Path("yellow_tripdata_2025-03.parquet")) == Period(2025, 3)
    assert Period.from_filename(Path("sample.parquet")) is None


@pytest.mark.parametrize(
    ("today", "expected"),
    [(date(2026, 9, 15), "2026-07"), (date(2026, 1, 15), "2025-11"), (date(2026, 2, 1), "2025-12")],
)
def test_default_period_lags_two_months(today: date, expected: str) -> None:
    assert str(default_period(today)) == expected


def test_period_window_crosses_year_boundary() -> None:
    assert [str(period) for period in period_window(Period(2025, 1), 3)] == [
        "2024-11",
        "2024-12",
        "2025-01",
    ]


def test_out_of_period_rows_are_removed_and_reported(synthetic_trips: pd.DataFrame) -> None:
    stray = synthetic_trips.head(3).copy()
    stray["tpep_pickup_datetime"] = pd.to_datetime(
        ["2008-12-31 23:00", "2009-01-01 00:10", "2025-02-01 00:05"]
    )
    stray["tpep_dropoff_datetime"] = stray["tpep_pickup_datetime"] + pd.Timedelta(minutes=12)
    frame = pd.concat([synthetic_trips, stray], ignore_index=True)
    config = load_config("configs/development.yaml")

    clean, report = validate_and_clean(frame, config.data, Period(2025, 1))

    assert report.out_of_period_rows == 3
    assert len(clean) == len(synthetic_trips)
    split = chronological_split(clean, config.split)
    assert split.train["tpep_pickup_datetime"].min() >= pd.Timestamp("2025-01-01")
    assert split.test["tpep_pickup_datetime"].max() < pd.Timestamp("2025-02-01")


def test_non_routable_zones_are_rejected_in_training(synthetic_trips: pd.DataFrame) -> None:
    frame = synthetic_trips.copy()
    frame.loc[:4, "DOLocationID"] = 264
    frame.loc[5:9, "PULocationID"] = 265
    _, report = validate_and_clean(frame, load_config("configs/development.yaml").data)
    assert report.rejected_rows == 10
