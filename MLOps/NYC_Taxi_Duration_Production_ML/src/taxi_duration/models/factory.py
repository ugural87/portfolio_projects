from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from taxi_duration.config import AppConfig
from taxi_duration.features.builder import FEATURE_COLUMNS, ZONE_COLUMNS, TripFeatureBuilder


def _zone_encoder(config: AppConfig) -> ColumnTransformer:
    """Encode nominal zone columns.

    ``target``: cross-fitted target encoding (out-of-fold during fit, full-train at inference)
    of the PU-DO route and of each zone. Unseen routes fall back to the training target mean.
    Because the encoder sits inside TransformedTargetRegressor it encodes log1p(duration).

    ``ordinal``: v1.0 behaviour, raw IDs passed through as numbers. Kept as a comparison arm.
    """
    zone_columns = list(ZONE_COLUMNS)
    numeric_columns = [c for c in FEATURE_COLUMNS if c not in ZONE_COLUMNS]
    if config.model.zone_encoding == "target":
        zone_step: Any = TargetEncoder(
            target_type="continuous",
            smooth="auto",
            cv=config.model.target_encoder_cv,
            shuffle=True,
            random_state=config.random_seed,
        )
    else:
        zone_step = "passthrough"
    return ColumnTransformer(
        [("zones", zone_step, zone_columns), ("numeric", "passthrough", numeric_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(config: AppConfig, max_iter: int | None = None) -> Pipeline:
    regressor = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=config.model.learning_rate,
        max_iter=max_iter if max_iter is not None else config.model.max_iter,
        max_leaf_nodes=config.model.max_leaf_nodes,
        l2_regularization=config.model.l2_regularization,
        min_samples_leaf=config.model.min_samples_leaf,
        random_state=config.random_seed,
        # The number of boosting iterations is chosen on the chronological validation segment
        # (training.pipeline.select_iterations), not on an internal random holdout.
        early_stopping=False,
    )
    inner = Pipeline([("encoder", _zone_encoder(config)), ("hgb", regressor)])
    target_model = TransformedTargetRegressor(
        regressor=inner,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    return Pipeline([("features", TripFeatureBuilder()), ("regressor", target_model)])


def seen_routes(model: Pipeline) -> set[int] | None:
    """Routes observed by the fitted target encoder, or None when not target-encoded."""
    try:
        encoder = model.named_steps["regressor"].regressor_.named_steps["encoder"]
        zones = encoder.named_transformers_["zones"]
    except (AttributeError, KeyError):
        return None
    if not isinstance(zones, TargetEncoder):
        return None
    return {int(v) for v in zones.categories_[0]}
