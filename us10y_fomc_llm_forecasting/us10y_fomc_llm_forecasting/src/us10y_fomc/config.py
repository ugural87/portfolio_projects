from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class PriceModelConfig:
    start_date: str = "1981-09-01"
    lookback: int = 60
    horizon: int = 5
    norm_window: int = 1250
    norm_min_periods: int = 250
    vol_window: int = 20
    vol_scaled_target: bool = True
    flat_threshold_sigma: float = 0.25
    flat_threshold_bp: float = 3.0
    fractions: tuple[float, float, float] = (0.65, 0.15, 0.10)
    pool_mode: str = "flatten"
    pool_heads: int = 8
    d_model: int = 128
    cnn_channels: int = 64
    n_heads: int = 4
    transformer_layers: int = 3
    feedforward_dim: int = 384
    dropout: float = 0.05
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    max_epochs: int = 300
    patience: int = 30
    gradient_clip: float = 1.0
    n_seeds: int = 3
    quantiles: tuple[float, float, float] = (0.05, 0.50, 0.95)
    regression_loss_weight: float = 0.50
    conformal_alpha: float = 0.10

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.pool_mode not in {"flatten", "multihead", "rank1"}:
            raise ValueError("pool_mode must be flatten, multihead, or rank1.")
        if len(self.fractions) != 3 or sum(self.fractions) >= 1.0:
            raise ValueError("fractions must contain train, validation, and calibration shares.")
        if not 0.0 < self.conformal_alpha < 1.0:
            raise ValueError("conformal_alpha must be between zero and one.")
        if self.cnn_channels % 8:
            raise ValueError("cnn_channels must be divisible by the eight GroupNorm groups.")


@dataclass(frozen=True)
class ExtractionConfig:
    start_year: int = 1993
    model_id: str = "gpt-5.6-luna"
    reasoning_effort: str = "medium"
    inference_policy_version: str = "luna-paid-safe-v3.1"
    primary_output_tokens: int = 24_000
    retry_output_tokens: int = 48_000
    minimum_strict_grounding_ratio: float = 0.80
    minimum_usable_feature_count: int = 11
    input_usd_per_million_tokens: float = 0.20
    output_usd_per_million_tokens: float = 1.20
    daily_token_limit: int = 20_000_000
    hard_run_budget_usd: float = 0.0
    budget_scope: str = "production-v1"

    def __post_init__(self) -> None:
        if self.primary_output_tokens <= 0 or self.retry_output_tokens < self.primary_output_tokens:
            raise ValueError("Extraction output-token limits are inconsistent.")
        if not 0.0 <= self.minimum_strict_grounding_ratio <= 1.0:
            raise ValueError("Grounding ratio must be between zero and one.")
        if not 1 <= self.minimum_usable_feature_count <= 14:
            raise ValueError("minimum_usable_feature_count must be between 1 and 14.")


@dataclass(frozen=True)
class FusionConfig:
    max_intermeeting_days: int = 60
    event_horizon_business_days: int = 5
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 250
    patience: int = 35
    warmup_epochs: int = 10
    random_seed: int = 42
    final_holdout_months: int = 12
    walk_forward_test_years: int = 3
    evaluate_final_holdout: bool = False
    n_attention_heads: int = 4
    freeze_price_encoder: bool = True
    train_fraction: float = 0.65
    validation_fraction: float = 0.15
    calibration_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.max_intermeeting_days <= 0 or self.event_horizon_business_days <= 0:
            raise ValueError("Event windows must be positive.")
        if self.n_attention_heads <= 0:
            raise ValueError("n_attention_heads must be positive.")
        fractions = self.train_fraction + self.validation_fraction + self.calibration_fraction
        if not 0.0 < fractions < 1.0:
            raise ValueError("Fusion train/validation/calibration fractions must sum below one.")


@dataclass(frozen=True)
class RuntimeConfig:
    download_fomc_minutes: bool = True
    run_openai_preflight: bool = False
    confirm_api_key: bool = False
    run_paid_extraction: bool = False
    confirm_paid_extraction: bool = False
    retry_quarantined_pairs: bool = False
    confirm_quarantine_retry: bool = False
    max_new_pairs: int = 0
    train_price_benchmark: bool = False
    train_historical_backbone: bool = False
    train_fusion_models: bool = False
    run_walk_forward: bool = False


@dataclass(frozen=True)
class ProjectConfig:
    price: PriceModelConfig
    extraction: ExtractionConfig
    fusion: FusionConfig
    runtime: RuntimeConfig


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def fomc(self) -> Path:
        return self.data / "external" / "fomc"

    @property
    def minutes_text(self) -> Path:
        return self.fomc / "minutes_text"

    @property
    def manifests(self) -> Path:
        return self.fomc / "manifests"

    @property
    def extraction_cache(self) -> Path:
        return self.fomc / "extraction_cache"

    @property
    def quarantine(self) -> Path:
        return self.fomc / "quarantine"

    @property
    def audit_reports(self) -> Path:
        return self.fomc / "audit_reports"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    def ensure(self) -> "ProjectPaths":
        directories = [
            self.data / "raw", self.data / "interim", self.data / "processed",
            self.minutes_text, self.manifests, self.extraction_cache,
            self.quarantine, self.audit_reports,
            self.outputs / "checkpoints" / "price",
            self.outputs / "checkpoints" / "historical_backbone",
            self.outputs / "checkpoints" / "fusion",
            self.outputs / "predictions", self.outputs / "metrics",
            self.outputs / "attention", self.outputs / "figures",
            self.outputs / "tables", self.outputs / "runtime", self.outputs / "reports",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return self


def _load_dataclass(cls: type[T], path: Path) -> T:
    payload: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(cls)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {path.name}: {sorted(unknown)}")
    for name in ("fractions", "quantiles"):
        if name in payload:
            payload[name] = tuple(payload[name])
    return cls(**payload)


def load_project_config(project_root: Path | str) -> ProjectConfig:
    root = Path(project_root).resolve()
    config_dir = root / "configs"
    return ProjectConfig(
        price=_load_dataclass(PriceModelConfig, config_dir / "price_model.toml"),
        extraction=_load_dataclass(ExtractionConfig, config_dir / "fomc_extraction.toml"),
        fusion=_load_dataclass(FusionConfig, config_dir / "fusion_model.toml"),
        runtime=_load_dataclass(RuntimeConfig, config_dir / "runtime.toml"),
    )
