import numpy as np
import pandas as pd

from taxi_duration.config import load_config
from taxi_duration.models.predict import predict_duration


class ExtremeModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        del features
        return np.asarray([-20.0, 500.0])


def test_shared_prediction_contract_clips_both_bounds() -> None:
    serving = load_config("configs/development.yaml").serving
    prediction = predict_duration(  # type: ignore[arg-type]
        ExtremeModel(), pd.DataFrame(index=range(2)), serving
    )
    assert prediction.tolist() == [serving.prediction_lower_bound, serving.prediction_upper_bound]
