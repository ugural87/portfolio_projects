from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from taxi_duration.config import ServingConfig
from taxi_duration.data.validation import DOLOCATION, PASSENGERS, PICKUP, PULOCATION
from taxi_duration.features.builder import INFERENCE_COLUMNS
from taxi_duration.models.predict import predict_duration


def _histogram(values: np.ndarray, bins: np.ndarray) -> dict[str, list[float]]:
    counts, edges = np.histogram(values, bins=bins)
    proportions = counts / max(counts.sum(), 1)
    return {"edges": edges.tolist(), "proportions": proportions.tolist()}


def _categorical_profile(values: pd.Series, top_n: int = 50) -> dict[str, float]:
    normalized = values.astype(str).value_counts(normalize=True)
    top = normalized.head(top_n)
    result = {str(key): float(value) for key, value in top.items()}
    result["__OTHER__"] = max(0.0, 1.0 - sum(result.values()))
    return result


def _routes(frame: pd.DataFrame) -> pd.Series:
    return (
        frame[PULOCATION].astype(int).astype(str) + ">" + frame[DOLOCATION].astype(int).astype(str)
    )


def build_reference_profile(
    frame: pd.DataFrame,
    output: Path,
    *,
    model: Pipeline | None = None,
    serving: ServingConfig | None = None,
) -> dict[str, object]:
    pickup = pd.to_datetime(frame[PICKUP])
    profile: dict[str, object] = {
        "row_count": len(frame),
        "pickup_hour": _histogram(pickup.dt.hour.to_numpy(), np.arange(0, 25, 2)),
        "passenger_count": _histogram(frame[PASSENGERS].to_numpy(), np.arange(0.5, 7.5, 1)),
        "duration_minutes": _histogram(
            frame["duration_minutes"].to_numpy(), np.linspace(1, 120, 13)
        ),
        "categorical": {
            "pickup_zone": _categorical_profile(frame[PULOCATION]),
            "dropoff_zone": _categorical_profile(frame[DOLOCATION]),
            "route": _categorical_profile(_routes(frame), top_n=100),
        },
        "known_routes": sorted(_routes(frame).unique().tolist()),
    }
    if model is not None and serving is not None:
        prediction = predict_duration(model, frame.loc[:, list(INFERENCE_COLUMNS)], serving)
        profile["prediction_minutes"] = _histogram(
            prediction,
            np.linspace(serving.prediction_lower_bound, serving.prediction_upper_bound, 25),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    return profile


def population_stability_index(expected: list[float], actual: list[float]) -> float:
    expected_arr = np.clip(np.asarray(expected, dtype=float), 1e-6, None)
    actual_arr = np.clip(np.asarray(actual, dtype=float), 1e-6, None)
    if expected_arr.shape != actual_arr.shape:
        raise ValueError("expected and actual histograms must have the same number of bins")
    return float(np.sum((actual_arr - expected_arr) * np.log(actual_arr / expected_arr)))
