from __future__ import annotations

import pandas as pd


def audit_backbone_cutoff(metadata: dict, first_fomc_meeting_date) -> dict[str, object]:
    last_label = pd.Timestamp(metadata["last_training_label_date"])
    first_meeting = pd.Timestamp(first_fomc_meeting_date)
    passed = last_label < first_meeting
    if not passed:
        raise AssertionError(
            f"Backbone leakage: last label {last_label.date()} is not before "
            f"first FOMC meeting {first_meeting.date()}."
        )
    return {
        "passed": True,
        "last_backbone_label_date": last_label,
        "first_fomc_meeting_date": first_meeting,
    }


def audit_event_splits(events: pd.DataFrame, splits: dict) -> dict[str, object]:
    names = [name for name in ("train", "validation", "calibration", "test", "final_holdout") if name in splits]
    rows = []
    seen = set()
    for name in names:
        indices = list(map(int, splits[name]))
        overlap = seen.intersection(indices)
        if overlap:
            raise AssertionError(f"Event split row overlap in {name}: {sorted(overlap)}")
        seen.update(indices)
        subset = events.iloc[indices]
        rows.append(
            {
                "split": name,
                "n": len(subset),
                "input_start": subset.input_start_date.min(),
                "target_end": subset.target_date.max(),
            }
        )
    for previous, current in zip(rows[:-1], rows[1:]):
        if not current["input_start"] > previous["target_end"]:
            raise AssertionError(
                f"Temporal overlap between {previous['split']} and {current['split']}."
            )
    return {"passed": True, "splits": rows}
