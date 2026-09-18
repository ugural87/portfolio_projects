from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    raw_dir: Path
    processed_dir: Path
    source_url_template: str
    min_duration_minutes: float = 1.0
    max_duration_minutes: float = 120.0
    min_passenger_count: int = 1
    max_passenger_count: int = 6
    valid_zone_min: int = 1
    valid_zone_max: int = 265
    training_window_months: int = Field(default=3, ge=1, le=12)
    source_lookback_months: int = Field(default=6, ge=1, le=24)
    drop_exact_duplicates: bool = True


class SplitConfig(StrictModel):
    train_fraction: float = Field(gt=0, lt=1)
    validation_fraction: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def fractions_leave_test_data(self) -> SplitConfig:
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be < 1")
        return self


class ModelConfig(StrictModel):
    name: str
    zone_encoding: Literal["target", "ordinal"] = "target"
    target_encoder_cv: int = Field(default=5, ge=2)
    max_iter: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_leaf_nodes: int = Field(gt=1)
    l2_regularization: float = Field(ge=0)
    min_samples_leaf: int = Field(gt=0)


class GateConfig(StrictModel):
    max_validation_mae_minutes: float
    max_test_mae_minutes: float
    max_p95_absolute_error_minutes: float
    min_improvement_vs_baseline: float = Field(ge=0)
    max_validation_test_gap: float
    max_regression_vs_champion: float = Field(default=0.01, gt=0, lt=0.2)
    max_rejection_rate: float = Field(default=0.10, ge=0, le=1)
    max_out_of_period_rate: float = Field(default=0.01, ge=0, le=1)
    max_duplicate_rate: float = Field(default=0.01, ge=0, le=1)
    min_valid_rows: int = Field(default=10_000, ge=1)
    champion_confidence_level: float = Field(default=0.95, gt=0.5, lt=1)


class ServingConfig(StrictModel):
    host: str
    port: int
    prediction_lower_bound: float
    prediction_upper_bound: float


class MonitoringConfig(StrictModel):
    reference_path: Path
    psi_warning: float
    psi_critical: float
    max_invalid_rate: float = Field(default=0.05, ge=0, le=1)
    max_unseen_route_rate: float = Field(default=0.20, ge=0, le=1)
    performance_window: int


class AppConfig(StrictModel):
    project_name: str
    environment: str = "development"
    random_seed: int
    timezone: str
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    gates: GateConfig
    serving: ServingConfig
    monitoring: MonitoringConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    parent = payload.pop("extends", None)
    if parent:
        base_path = config_path.parent / str(parent)
        base_payload = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        payload = _deep_merge(base_payload, payload)
    return AppConfig.model_validate(payload)


def canonical_config_json(config: AppConfig) -> str:
    """Canonical, fully resolved configuration used for artifact lineage hashing."""
    return json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
