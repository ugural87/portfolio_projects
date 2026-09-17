from __future__ import annotations

from collections import OrderedDict

import pandas as pd

from .config import SplitConfig


def _advance_past_tie(frame: pd.DataFrame, index: int) -> int:
    index = min(max(index, 1), len(frame) - 1)
    time_value = frame.iloc[index - 1]["Time"]
    while index < len(frame) and frame.iloc[index]["Time"] == time_value:
        index += 1
    return index


def temporal_partitions(frame: pd.DataFrame, config: SplitConfig) -> OrderedDict[str, pd.DataFrame]:
    config.validate()
    ordered = frame.sort_values("Time", kind="mergesort").reset_index(drop=True)
    fractions = [config.train, config.validation, config.calibration, config.policy]
    cutoffs: list[int] = []
    cumulative = 0.0
    for fraction in fractions:
        cumulative += fraction
        cutoffs.append(_advance_past_tie(ordered, int(len(ordered) * cumulative)))
    starts = [0, *cutoffs]
    ends = [*cutoffs, len(ordered)]
    names = ["train", "validation", "calibration", "policy", "test"]
    parts: OrderedDict[str, pd.DataFrame] = OrderedDict()
    for name, start, end in zip(names, starts, ends):
        part = ordered.iloc[start:end].copy().reset_index(drop=True)
        if part.empty or part["Class"].nunique() != 2:
            raise ValueError(f"{name} partition is empty or lacks one class")
        parts[name] = part
    for left, right in zip(names[:-1], names[1:]):
        if parts[left]["Time"].max() >= parts[right]["Time"].min():
            raise AssertionError(f"Temporal overlap between {left} and {right}")
    return parts


def split_summary(parts: OrderedDict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, part in parts.items():
        rows.append({
            "partition": name,
            "rows": len(part),
            "frauds": int(part["Class"].sum()),
            "fraud_rate": float(part["Class"].mean()),
            "start_hour": float(part["Time"].min() / 3600.0),
            "end_hour": float(part["Time"].max() / 3600.0),
        })
    return pd.DataFrame(rows)

