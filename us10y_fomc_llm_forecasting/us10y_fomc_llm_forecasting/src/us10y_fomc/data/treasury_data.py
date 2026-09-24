from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..progress import progress
from .fred_client import fetch_fred_series, make_fred_session

SERIES = {
    "DGS3MO": 0.25,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS10": 10.0,
    "DGS30": 30.0,
}
SERIES_IDS = tuple(SERIES)
MATURITY_YEARS = np.asarray(tuple(SERIES.values()), dtype=np.float32)
TARGET_SERIES = "DGS10"
CHANNEL_NAMES = ("level", "change_1d", "change_5d", "realised_vol_20d")


def fetch_treasury_panel(
    api_key: str,
    start_date: str,
    series_ids: Sequence[str] = SERIES_IDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_list: list[pd.Series] = []
    audit_rows: list[dict[str, object]] = []
    with make_fred_session() as session:
        for series_id in progress(
            series_ids,
            desc="FRED Treasury series",
            total=len(series_ids),
            unit="series",
        ):
            series = fetch_fred_series(series_id, start_date, api_key, session)
            series_list.append(series)
            valid = series.dropna()
            if valid.empty:
                raise RuntimeError(f"{series_id} contains no valid observations.")
            audit_rows.append(
                {
                    "series": series_id,
                    "first_valid": valid.index.min(),
                    "last_valid": valid.index.max(),
                    "valid_observations": len(valid),
                    "missing_before_intersection": int(series.isna().sum()),
                }
            )
    raw_panel = pd.concat(series_list, axis=1).sort_index()
    panel = raw_panel.dropna(how="any").astype(np.float32)
    if panel.empty or not panel.index.is_monotonic_increasing or not panel.index.is_unique:
        raise RuntimeError("The common Treasury panel is empty or has an invalid index.")
    if (panel <= -5.0).any().any() or (panel >= 25.0).any().any():
        raise ValueError("Treasury-yield range audit failed.")
    audit = pd.DataFrame(audit_rows).set_index("series")
    return panel, audit


def save_treasury_panel(panel: pd.DataFrame, audit: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    panel.to_csv(directory / "treasury_yields.csv", index_label="date")
    audit.to_csv(directory / "treasury_data_audit.csv")


def load_treasury_panel(directory: Path) -> pd.DataFrame:
    path = directory / "treasury_yields.csv"
    panel = pd.read_csv(path, index_col="date", parse_dates=True)
    return panel.astype(np.float32).sort_index()
