from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..progress import progress
from .fred_client import fetch_fred_series, make_fred_session

VULGAR_FRACTIONS = str.maketrans({"¼": "1/4", "½": "1/2", "¾": "3/4"})
NUMBER_PATTERN = r"(?:\d+-\d+/\d+|\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"
RANGE_PATTERNS = (
    re.compile(
        rf"target range for (?:the )?federal funds rate.{{0,100}}?"
        rf"(?:at|of|to)?\s*(?P<lower>{NUMBER_PATTERN})\s*(?:percent)?\s+"
        rf"(?:to|–|-)\s+(?P<upper>{NUMBER_PATTERN})\s*percent",
        re.I | re.S,
    ),
    re.compile(
        rf"(?P<lower>{NUMBER_PATTERN})\s+(?:to|–|-)\s+"
        rf"(?P<upper>{NUMBER_PATTERN})\s*percent\s+target range for "
        rf"(?:the )?federal funds rate",
        re.I,
    ),
)
POINT_PATTERN = re.compile(
    rf"(?:intended\s+|target\s+)?federal funds rate.{{0,160}}?"
    rf"(?:at|of|to)\s+(?:an average of\s+|around\s+|approximately\s+)?"
    rf"(?P<point>{NUMBER_PATTERN})\s*percent",
    re.I | re.S,
)


def parse_rate_number(value: str) -> float:
    value = value.strip().replace("–", "-")
    mixed = re.fullmatch(r"(\d+)[-\s](\d+)/(\d+)", value)
    if mixed:
        return float(mixed.group(1)) + float(mixed.group(2)) / float(mixed.group(3))
    fraction = re.fullmatch(r"(\d+)/(\d+)", value)
    if fraction:
        return float(fraction.group(1)) / float(fraction.group(2))
    return float(value)


def extract_minutes_target_midpoint(text: str) -> float:
    normalized = text.translate(VULGAR_FRACTIONS)
    ranges = [match for pattern in RANGE_PATTERNS for match in pattern.finditer(normalized)]
    if ranges:
        match = max(ranges, key=lambda item: item.start())
        return float(
            np.mean(
                [parse_rate_number(match.group("lower")), parse_rate_number(match.group("upper"))]
            )
        )
    points = list(POINT_PATTERN.finditer(normalized))
    return np.nan if not points else parse_rate_number(points[-1].group("point"))


def build_policy_rate_audit(
    documents: pd.DataFrame, api_key: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_ids = ("DFEDTAR", "DFEDTARL", "DFEDTARU")
    with make_fred_session() as session:
        policy = {
            series_id: fetch_fred_series(series_id, "1993-01-01", api_key, session)
            for series_id in progress(
                series_ids,
                desc="FRED policy-rate series",
                total=len(series_ids),
                unit="series",
            )
        }
    index = pd.date_range(
        min(series.index.min() for series in policy.values()),
        max(series.index.max() for series in policy.values()),
        freq="D",
    )
    point = policy["DFEDTAR"].reindex(index).ffill()
    point.loc[point.index > policy["DFEDTAR"].dropna().index.max()] = np.nan
    midpoint = point.combine_first(
        (policy["DFEDTARL"].reindex(index).ffill() + policy["DFEDTARU"].reindex(index).ffill()) / 2
    )

    rows = []
    for document in progress(
        documents.itertuples(index=False),
        desc="Policy-rate audit",
        total=len(documents),
        unit="meeting",
    ):
        date = pd.Timestamp(document.meeting_date).normalize()
        candidates = pd.date_range(date - pd.Timedelta(days=1), date + pd.Timedelta(days=1))
        changes = midpoint.diff().reindex(candidates).dropna()
        nonzero = changes.loc[changes.abs() > 1e-12]
        if len(nonzero) == 1:
            effective = nonzero.index[0]
        elif len(nonzero) > 1 and date in nonzero.index:
            effective = date
        elif len(nonzero) > 1:
            raise AssertionError(f"Ambiguous FRED decision window around {date.date()}.")
        else:
            effective = date
        minutes_midpoint = extract_minutes_target_midpoint(
            Path(document.text_path).read_text(encoding="utf-8")
        )
        rows.append(
            {
                "meeting_date": date,
                "actual_rate_change_bp": 100.0 * float(changes.get(effective, 0.0)),
                "rate_effective_date": effective,
                "fred_target_midpoint": float(midpoint.asof(effective)),
                "minutes_target_midpoint": minutes_midpoint,
            }
        )
    rates = pd.DataFrame(rows)
    rates["minutes_fred_gap_bp"] = 100.0 * (
        rates.minutes_target_midpoint - rates.fred_target_midpoint
    )
    anchors = {
        pd.Timestamp("2008-12-16"): -87.5,
        pd.Timestamp("2020-04-29"): 0.0,
        pd.Timestamp("2022-06-15"): 75.0,
    }
    lookup = rates.set_index("meeting_date").actual_rate_change_bp
    for date, expected in anchors.items():
        if date not in lookup.index or not np.isclose(lookup.loc[date], expected):
            raise AssertionError(f"Known FRED decision anchor failed: {date.date()}.")
    residual = np.mod(np.abs(rates.actual_rate_change_bp), 12.5)
    if not (np.minimum(residual, 12.5 - residual) < 1e-6).all():
        raise AssertionError("Off-grid FRED target-rate change detected.")
    disagreements = rates.loc[
        rates.minutes_target_midpoint.notna() & rates.minutes_fred_gap_bp.abs().gt(0.5)
    ].copy()
    return rates, disagreements
