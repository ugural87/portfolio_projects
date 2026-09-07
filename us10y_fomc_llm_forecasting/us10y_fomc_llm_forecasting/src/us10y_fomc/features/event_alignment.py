from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FusionConfig


def align_fomc_events(
    events: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    config: FusionConfig,
) -> pd.DataFrame:
    rows = []
    for event in events.itertuples(index=False):
        previous_date = pd.Timestamp(event.previous_meeting_date)
        meeting_date = pd.Timestamp(event.meeting_date)
        positions = np.flatnonzero((price_index > previous_date) & (price_index < meeting_date))
        if not len(positions):
            raise RuntimeError(f"No pre-meeting prices for {meeting_date.date()}.")
        if len(positions) > config.max_intermeeting_days:
            raise RuntimeError(
                f"{meeting_date.date()} has {len(positions)} intermeeting price days; "
                "increase max_intermeeting_days explicitly."
            )
        post = np.flatnonzero(price_index >= meeting_date)
        if len(post) < config.event_horizon_business_days:
            continue
        pre_anchor = int(positions[-1])
        target_position = int(post[config.event_horizon_business_days - 1])
        if not price_index[pre_anchor] < meeting_date <= price_index[target_position]:
            raise AssertionError("Event alignment is not strictly causal.")
        row = event._asdict()
        row.update(
            {
                "chunk_positions": positions.astype(np.int64),
                "chunk_dates": price_index[positions].to_numpy(),
                "chunk_length": len(positions),
                "input_start_date": price_index[int(positions[0])],
                "input_end_date": price_index[pre_anchor],
                "pre_anchor": pre_anchor,
                "target_position": target_position,
                "target_date": price_index[target_position],
            }
        )
        rows.append(row)
    aligned = pd.DataFrame(rows).sort_values("meeting_date").reset_index(drop=True)
    if aligned.empty:
        raise RuntimeError("No FOMC events aligned to the Treasury panel.")
    return aligned


def purge_later_split(
    events: pd.DataFrame, earlier: np.ndarray, later: np.ndarray
) -> np.ndarray:
    if not len(earlier) or not len(later):
        return later
    boundary = events.iloc[earlier].target_date.max()
    return np.asarray(
        [index for index in later if events.iloc[index].input_start_date > boundary], dtype=int
    )


def make_event_splits(
    events: pd.DataFrame, config: FusionConfig
) -> tuple[dict[str, np.ndarray], pd.Timestamp]:
    holdout_cutoff = events.meeting_date.max() - pd.DateOffset(months=config.final_holdout_months)
    development = np.flatnonzero(events.meeting_date < holdout_cutoff)
    final_holdout = np.flatnonzero(events.meeting_date >= holdout_cutoff)
    n = len(development)
    train_end = int(n * config.train_fraction)
    validation_end = int(n * (config.train_fraction + config.validation_fraction))
    calibration_end = int(
        n
        * (
            config.train_fraction
            + config.validation_fraction
            + config.calibration_fraction
        )
    )
    splits = {
        "train": development[:train_end],
        "validation": development[train_end:validation_end],
        "calibration": development[validation_end:calibration_end],
        "test": development[calibration_end:],
        "final_holdout": final_holdout,
    }
    names = list(splits)
    for earlier, later in zip(names[:-1], names[1:]):
        splits[later] = purge_later_split(events, splits[earlier], splits[later])
        if not len(splits[later]):
            raise RuntimeError(f"Purging emptied the {later} split.")
        if not events.iloc[splits[later]].input_start_date.min() > events.iloc[
            splits[earlier]
        ].target_date.max():
            raise AssertionError(f"Temporal overlap remains before {later}.")
    return splits, holdout_cutoff
