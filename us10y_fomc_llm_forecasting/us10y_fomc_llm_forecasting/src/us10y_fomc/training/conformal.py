from __future__ import annotations

import math

import numpy as np


def conformal_correction(
    labels: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    labels = np.asarray(labels)
    scores = np.maximum(lower - labels, labels - upper)
    n = len(scores)
    if n < 19 and alpha == 0.10:
        raise ValueError("At least 19 calibration observations are required at alpha=0.10.")
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))
