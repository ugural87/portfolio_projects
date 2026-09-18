from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from taxi_duration.data.validation import DOLOCATION, PICKUP, PULOCATION
from taxi_duration.features.builder import ROUTE_MULTIPLIER


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Point-estimate metrics.

    The model estimates the conditional *median* (L1 loss; quantiles are invariant under the
    monotone log1p transform), so for right-skewed durations bias_minutes is expected to be
    negative. It is reported for monitoring, not gated.
    """
    absolute_error = np.abs(y_true.to_numpy() - y_pred)
    return {
        "mae_minutes": float(mean_absolute_error(y_true, y_pred)),
        "rmse_minutes": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "p95_absolute_error_minutes": float(np.quantile(absolute_error, 0.95)),
        "bias_minutes": float(np.mean(y_pred - y_true.to_numpy())),
    }


def _slice_metrics(frame: pd.DataFrame, key: pd.Series) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for value, group in frame.groupby(key, observed=True, sort=True):
        error = group["prediction"] - group["actual"]
        out[str(value)] = {
            "rows": len(group),
            "mae_minutes": float(error.abs().mean()),
            "bias_minutes": float(error.mean()),
        }
    return out


def error_breakdown(
    features: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    seen_routes: set[int] | None = None,
) -> dict[str, Any]:
    """MAE and bias by pickup hour, weekday and (when available) route coverage."""
    frame = pd.DataFrame({"actual": y_true.to_numpy(), "prediction": y_pred}, index=features.index)
    pickup = pd.to_datetime(features[PICKUP])
    report: dict[str, Any] = {
        "by_pickup_hour": _slice_metrics(frame, pickup.dt.hour.rename("hour")),
        "by_weekday": _slice_metrics(frame, pickup.dt.dayofweek.rename("weekday")),
    }
    if seen_routes is not None:
        route = features[PULOCATION].astype("int64") * ROUTE_MULTIPLIER + features[
            DOLOCATION
        ].astype("int64")
        seen = route.isin(seen_routes).map({True: "seen_in_training", False: "unseen_route"})
        report["by_route_coverage"] = _slice_metrics(frame, seen.rename("route_coverage"))
    return report
