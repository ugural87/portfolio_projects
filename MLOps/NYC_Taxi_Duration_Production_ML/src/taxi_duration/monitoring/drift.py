from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from taxi_duration.config import MonitoringConfig, ServingConfig
from taxi_duration.data.validation import DOLOCATION, PASSENGERS, PICKUP, PULOCATION
from taxi_duration.features.builder import INFERENCE_COLUMNS
from taxi_duration.models.predict import predict_duration
from taxi_duration.monitoring.profile import population_stability_index


def _profile_values(
    values: np.ndarray, edges: list[float]
) -> tuple[list[float], dict[str, float | int]]:
    """Split a feature into (histogram of valid values, data-quality counts).

    np.histogram silently drops NaN and out-of-range values and the proportions would then be
    renormalised over the survivors, so a batch where a large share of rows is broken would
    still look distribution-stable. Invalid values are therefore counted and reported apart.
    """
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    missing = np.isnan(numeric)
    out_of_range = ~missing & ((numeric < edges[0]) | (numeric > edges[-1]))
    valid = numeric[~missing & ~out_of_range]
    counts, _ = np.histogram(valid, bins=np.asarray(edges))
    total = len(numeric)
    quality: dict[str, float | int] = {
        "rows": total,
        "valid_rows": len(valid),
        "missing_rows": int(missing.sum()),
        "out_of_range_rows": int(out_of_range.sum()),
        "invalid_rate": float((missing.sum() + out_of_range.sum()) / total) if total else 0.0,
    }
    return (counts / max(counts.sum(), 1)).tolist(), quality


def _status(psi: float, invalid_rate: float, config: MonitoringConfig) -> str:
    if psi >= config.psi_critical or invalid_rate > config.max_invalid_rate:
        return "critical"
    if psi >= config.psi_warning:
        return "warning"
    return "ok"


def _categorical_psi(values: pd.Series, expected: dict[str, float]) -> float:
    normalized = values.astype(str).value_counts(normalize=True)
    categories = [name for name in expected if name != "__OTHER__"]
    actual = [float(normalized.get(name, 0.0)) for name in categories]
    actual.append(max(0.0, 1.0 - sum(actual)))
    expected_values = [float(expected[name]) for name in categories]
    expected_values.append(float(expected.get("__OTHER__", 0.0)))
    return population_stability_index(expected_values, actual)


def _valid_inference_rows(current: pd.DataFrame) -> pd.DataFrame:
    frame = current.loc[:, list(INFERENCE_COLUMNS)].copy()
    frame[PICKUP] = pd.to_datetime(frame[PICKUP], errors="coerce")
    for name in (PULOCATION, DOLOCATION, PASSENGERS):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.dropna()


def drift_report(
    current: pd.DataFrame,
    reference_path: Path,
    config: MonitoringConfig,
    *,
    model: Pipeline | None = None,
    serving: ServingConfig | None = None,
) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    pickup = pd.to_datetime(current[PICKUP], errors="coerce")
    features = {
        "pickup_hour": pickup.dt.hour.to_numpy(dtype=float, na_value=np.nan),
        "passenger_count": current[PASSENGERS].to_numpy(dtype=float, na_value=np.nan),
    }
    report: dict[str, Any] = {"row_count": len(current), "features": {}}
    for name, values in features.items():
        expected = reference[name]
        actual, quality = _profile_values(values, expected["edges"])
        psi = population_stability_index(expected["proportions"], actual)
        report["features"][name] = {
            "psi": psi,
            "data_quality": quality,
            "status": _status(psi, float(quality["invalid_rate"]), config),
        }
    valid = _valid_inference_rows(current)
    categorical_values = {
        "pickup_zone": valid[PULOCATION].astype(int).astype(str),
        "dropoff_zone": valid[DOLOCATION].astype(int).astype(str),
        "route": valid[PULOCATION].astype(int).astype(str)
        + ">"
        + valid[DOLOCATION].astype(int).astype(str),
    }
    categorical_report: dict[str, Any] = {}
    for name, categorical_series in categorical_values.items():
        psi = _categorical_psi(categorical_series, reference["categorical"][name])
        categorical_report[name] = {"psi": psi, "status": _status(psi, 0.0, config)}
    known_routes = set(reference["known_routes"])
    route_values = categorical_values["route"]
    unseen_rate = float((~route_values.isin(known_routes)).mean()) if len(route_values) else 1.0
    categorical_report["route"]["unseen_rate"] = unseen_rate
    if unseen_rate > config.max_unseen_route_rate:
        categorical_report["route"]["status"] = "critical"
    report["categorical"] = categorical_report

    if model is not None and serving is not None and "prediction_minutes" in reference:
        prediction = predict_duration(model, valid, serving)
        expected_prediction = reference["prediction_minutes"]
        actual, quality = _profile_values(prediction, expected_prediction["edges"])
        psi = population_stability_index(expected_prediction["proportions"], actual)
        report["prediction_minutes"] = {
            "psi": psi,
            "data_quality": quality,
            "status": _status(psi, float(quality["invalid_rate"]), config),
        }
    statuses = [f["status"] for f in report["features"].values()]
    statuses.extend(f["status"] for f in categorical_report.values())
    if "prediction_minutes" in report:
        statuses.append(report["prediction_minutes"]["status"])
    report["status"] = (
        "critical" if "critical" in statuses else "warning" if "warning" in statuses else "ok"
    )
    return report
