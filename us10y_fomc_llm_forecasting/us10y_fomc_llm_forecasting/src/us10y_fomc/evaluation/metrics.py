from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def forecast_metrics(
    labels_model_units: np.ndarray,
    classes: np.ndarray,
    logits: np.ndarray,
    quantiles: np.ndarray,
    scales_bp: np.ndarray,
    conformal_qhat: float = 0.0,
) -> dict[str, float]:
    truth_bp = labels_model_units * scales_bp
    median_bp = quantiles[:, 1] * scales_bp
    lower_bp = (quantiles[:, 0] - conformal_qhat) * scales_bp
    upper_bp = (quantiles[:, 2] + conformal_qhat) * scales_bp
    predicted_class = logits.argmax(axis=1)
    return {
        "n": float(len(truth_bp)),
        "direction_accuracy": float(accuracy_score(classes, predicted_class)),
        "balanced_accuracy": float(balanced_accuracy_score(classes, predicted_class)),
        "macro_f1": float(f1_score(classes, predicted_class, average="macro", zero_division=0)),
        "mae_bp": float(np.mean(np.abs(truth_bp - median_bp))),
        "rmse_bp": float(np.sqrt(np.mean(np.square(truth_bp - median_bp)))),
        "interval_coverage": float(np.mean((truth_bp >= lower_bp) & (truth_bp <= upper_bp))),
        "mean_interval_width_bp": float(np.mean(upper_bp - lower_bp)),
    }


def prediction_frame(predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    scales = predictions["scales"]
    quantiles = predictions["quantiles"]
    return {
        "truth_bp": predictions["labels"] * scales,
        "forecast_bp": quantiles[:, 1] * scales,
        "lower_raw_bp": quantiles[:, 0] * scales,
        "upper_raw_bp": quantiles[:, 2] * scales,
        "true_class": predictions["classes"],
        "predicted_class": predictions["direction_logits"].argmax(axis=1),
    }
