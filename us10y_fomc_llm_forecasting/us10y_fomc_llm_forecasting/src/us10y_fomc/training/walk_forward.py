from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FusionConfig


def build_walk_forward_folds(
    events: pd.DataFrame,
    holdout_cutoff: pd.Timestamp,
    config: FusionConfig,
    minimum_train: int = 40,
    minimum_calibration: int = 19,
) -> pd.DataFrame:
    development = events.index[events.meeting_date < holdout_cutoff].to_numpy(dtype=int)
    years = sorted(events.loc[development, "meeting_date"].dt.year.unique())
    rows = []
    fold = 0
    year_cursor = 0
    while year_cursor < len(years):
        start_year = years[year_cursor]
        stop_year = start_year + config.walk_forward_test_years - 1
        test = development[
            events.loc[development, "meeting_date"].dt.year.between(start_year, stop_year).to_numpy()
        ]
        train_pool = development[
            (events.loc[development, "meeting_date"] < pd.Timestamp(start_year, 1, 1)).to_numpy()
        ]
        if not len(test):
            year_cursor += 1
            continue
        validation_size = max(8, int(np.ceil(0.15 * len(train_pool))))
        initial_calibration = max(
            minimum_calibration, int(np.ceil(0.10 * len(train_pool)))
        )
        selected = None
        largest_calibration = len(train_pool) - validation_size - minimum_train
        for calibration_size in range(initial_calibration, largest_calibration + 1):
            train_stop = len(train_pool) - validation_size - calibration_size
            candidate = {
                "train": train_pool[:train_stop],
                "validation": train_pool[
                    train_stop : train_stop + validation_size
                ],
                "calibration": train_pool[train_stop + validation_size :],
                "test": test,
            }
            for previous, current in (
                ("train", "validation"),
                ("validation", "calibration"),
                ("calibration", "test"),
            ):
                boundary = events.loc[candidate[previous], "target_date"].max()
                later = candidate[current]
                candidate[current] = later[
                    events.loc[later, "input_start_date"].to_numpy() > boundary
                ]
            if (
                len(candidate["train"]) >= minimum_train
                and len(candidate["validation"]) > 0
                and len(candidate["calibration"]) >= minimum_calibration
                and len(candidate["test"]) > 0
            ):
                selected = candidate
                break
        if selected is None:
            year_cursor += 1
            continue
        train = selected["train"]
        validation = selected["validation"]
        calibration = selected["calibration"]
        test = selected["test"]
        rows.append(
            {
                "fold": fold,
                "test_start_year": start_year,
                "test_stop_year": stop_year,
                "train_indices": train,
                "validation_indices": validation,
                "calibration_indices": calibration,
                "test_indices": test,
                "n_train": len(train),
                "n_validation": len(validation),
                "n_calibration": len(calibration),
                "n_test": len(test),
            }
        )
        fold += 1
        year_cursor += config.walk_forward_test_years
    if not rows:
        raise RuntimeError("No walk-forward fold satisfies the training/calibration constraints.")
    manifest = pd.DataFrame(rows)
    tested = np.concatenate(manifest.test_indices.to_list())
    if len(np.unique(tested)) != len(tested):
        raise AssertionError("Walk-forward test folds overlap.")
    return manifest
