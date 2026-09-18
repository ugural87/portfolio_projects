from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from taxi_duration.config import DataConfig

PICKUP = "tpep_pickup_datetime"
DROPOFF = "tpep_dropoff_datetime"
PULOCATION = "PULocationID"
DOLOCATION = "DOLocationID"
PASSENGERS = "passenger_count"
REQUIRED_COLUMNS = (PICKUP, DROPOFF, PULOCATION, DOLOCATION, PASSENGERS)

_PERIOD_IN_NAME = re.compile(r"(\d{4})-(\d{2})\.parquet$")


@dataclass(frozen=True)
class Period:
    """One calendar month of source data, in naive New York local time like the TLC files."""

    year: int
    month: int

    def __post_init__(self) -> None:
        if self.year < 2009 or not 1 <= self.month <= 12:
            raise ValueError("period must be year >= 2009 and month 1..12")

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(year=self.year, month=self.month, day=1)

    @property
    def end(self) -> pd.Timestamp:
        return self.start + pd.offsets.MonthBegin(1)

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    def shift(self, months: int) -> Period:
        index = self.year * 12 + self.month - 1 + months
        return Period(index // 12, index % 12 + 1)

    @classmethod
    def parse(cls, text: str) -> Period:
        match = re.fullmatch(r"(\d{4})-(\d{1,2})", text.strip())
        if not match:
            raise ValueError(f"period must look like YYYY-MM, got {text!r}")
        return cls(int(match.group(1)), int(match.group(2)))

    @classmethod
    def from_filename(cls, path: Path) -> Period | None:
        match = _PERIOD_IN_NAME.search(path.name)
        return cls(int(match.group(1)), int(match.group(2))) if match else None


@dataclass(frozen=True)
class ValidationReport:
    input_rows: int
    valid_rows: int
    rejected_rows: int
    rejection_rate: float
    out_of_period_rows: int
    duplicate_rows: int
    period: str | None

    @property
    def out_of_period_rate(self) -> float:
        return self.out_of_period_rows / self.input_rows if self.input_rows else 0.0

    @property
    def duplicate_rate(self) -> float:
        return self.duplicate_rows / self.input_rows if self.input_rows else 0.0


def validate_and_clean(
    raw: pd.DataFrame, config: DataConfig, period: Period | None = None
) -> tuple[pd.DataFrame, ValidationReport]:
    """Apply schema, domain and (optionally) source-period rules.

    Monthly TLC files contain a small number of rows whose pickup timestamp falls outside the
    file's month. Without a period filter those rows sort to the extremes of the chronological
    split and corrupt both the test window and the recorded lineage periods.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    frame = raw[list(REQUIRED_COLUMNS)].copy()
    frame[PICKUP] = pd.to_datetime(frame[PICKUP], errors="coerce")
    frame[DROPOFF] = pd.to_datetime(frame[DROPOFF], errors="coerce")
    frame["duration_minutes"] = (frame[DROPOFF] - frame[PICKUP]).dt.total_seconds() / 60

    in_period = pd.Series(True, index=frame.index)
    if period is not None:
        in_period = (frame[PICKUP] >= period.start) & (frame[PICKUP] < period.end)

    duplicates = frame.duplicated(subset=list(REQUIRED_COLUMNS), keep="first")
    valid = (
        frame[PICKUP].notna()
        & frame[DROPOFF].notna()
        & in_period
        & frame["duration_minutes"].between(
            config.min_duration_minutes, config.max_duration_minutes, inclusive="both"
        )
        & frame[PASSENGERS].between(
            config.min_passenger_count, config.max_passenger_count, inclusive="both"
        )
        & frame[PULOCATION].between(config.valid_zone_min, config.valid_zone_max)
        & frame[DOLOCATION].between(config.valid_zone_min, config.valid_zone_max)
    )
    if config.drop_exact_duplicates:
        valid &= ~duplicates
    cleaned = frame.loc[valid].copy()
    cleaned.sort_values(by=PICKUP, inplace=True, kind="stable")
    cleaned.reset_index(drop=True, inplace=True)
    input_rows = len(frame)
    rejected = input_rows - len(cleaned)
    report = ValidationReport(
        input_rows=input_rows,
        valid_rows=len(cleaned),
        rejected_rows=rejected,
        rejection_rate=rejected / input_rows if input_rows else 0.0,
        out_of_period_rows=int((frame[PICKUP].notna() & ~in_period).sum()),
        duplicate_rows=int(duplicates.sum()),
        period=str(period) if period is not None else None,
    )
    if cleaned.empty:
        raise ValueError("no valid rows remain after validation")
    return cleaned, report
