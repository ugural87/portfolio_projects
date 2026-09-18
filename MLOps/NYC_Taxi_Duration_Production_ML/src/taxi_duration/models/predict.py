from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from taxi_duration.config import ServingConfig


def predict_duration(
    model: Pipeline, features: pd.DataFrame, config: ServingConfig
) -> NDArray[np.float64]:
    """Apply the single train/evaluate/serve prediction contract."""
    raw = np.asarray(model.predict(features), dtype=float)
    return cast(
        NDArray[np.float64],
        np.clip(raw, config.prediction_lower_bound, config.prediction_upper_bound),
    )
