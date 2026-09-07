from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..llm.feature_contract import FOMC_FEATURE_NAMES


EVENT_TOKEN_NAMES = ("rate_decision", *FOMC_FEATURE_NAMES)


def export_attention_tables(
    predictions: dict[str, np.ndarray],
    event_rows: pd.DataFrame,
    output_directory: Path,
    prefix: str,
) -> tuple[Path, Path]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    e2p = predictions["event_to_price_attention"].mean(axis=1)
    p2e = predictions["price_to_event_attention"].mean(axis=1)[:, 0, :]
    valid_event = predictions["event_valid_mask"].astype(bool)
    e2p_rows = []
    p2e_rows = []
    for sample in range(len(event_rows)):
        meeting_date = pd.Timestamp(event_rows.iloc[sample].meeting_date)
        valid_price_count = int(predictions["price_valid_mask"][sample].sum())
        price_dates = pd.to_datetime(event_rows.iloc[sample].chunk_dates)
        if len(price_dates) != valid_price_count:
            raise AssertionError("Attention price mask does not match the aligned date chunk.")
        for token, name in enumerate(EVENT_TOKEN_NAMES[: e2p.shape[1]]):
            if not valid_event[sample, token]:
                continue
            weights = e2p[sample, token, -valid_price_count:]
            for offset, (price_date, weight) in enumerate(
                zip(price_dates, weights), start=-valid_price_count + 1
            ):
                e2p_rows.append(
                    {
                        "meeting_date": meeting_date,
                        "event_token": name,
                        "price_date": price_date,
                        "business_day_offset": offset,
                        "attention": float(weight),
                    }
                )
        for token, name in enumerate(EVENT_TOKEN_NAMES[: p2e.shape[1]]):
            if valid_event[sample, token]:
                p2e_rows.append(
                    {
                        "meeting_date": meeting_date,
                        "event_token": name,
                        "attention": float(p2e[sample, token]),
                    }
                )
    e2p_path = output_directory / f"{prefix}_event_to_price.csv"
    p2e_path = output_directory / f"{prefix}_price_to_event.csv"
    pd.DataFrame(e2p_rows).to_csv(e2p_path, index=False)
    pd.DataFrame(p2e_rows).to_csv(p2e_path, index=False)
    return e2p_path, p2e_path
