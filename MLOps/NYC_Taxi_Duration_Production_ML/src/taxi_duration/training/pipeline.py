from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from taxi_duration.config import AppConfig, canonical_config_json
from taxi_duration.data.ingest import sha256_file
from taxi_duration.data.validation import PICKUP, Period, validate_and_clean
from taxi_duration.features.builder import INFERENCE_COLUMNS
from taxi_duration.models.artifact import hash_text, load_artifact, new_metadata, save_artifact
from taxi_duration.models.factory import build_model, seen_routes
from taxi_duration.models.predict import predict_duration
from taxi_duration.monitoring.profile import build_reference_profile
from taxi_duration.training.evaluate import error_breakdown, regression_metrics
from taxi_duration.training.gates import (
    ChampionComparison,
    evaluate_release_gates,
    paired_delta_upper_bound,
)
from taxi_duration.training.split import chronological_split

LOGGER = structlog.get_logger(__name__)
TARGET = "duration_minutes"


def _git_sha() -> str:
    explicit = os.getenv("GIT_SHA")
    if explicit:
        return explicit
    head_path = Path(".git/HEAD")
    if not head_path.exists():
        return "unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        reference = Path(".git") / head.removeprefix("ref: ")
        return reference.read_text(encoding="utf-8").strip() if reference.exists() else "unknown"
    return head


def select_iterations(
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    config: AppConfig,
) -> tuple[int, list[float]]:
    """Pick the boosting iteration count that minimises MAE on the chronological validation set.

    Replaces HistGradientBoosting's built-in early stopping, which holds out a *random* 10% of
    the training rows and therefore mixes time periods inside a chronological design.
    """
    target_model = model.named_steps["regressor"]
    inner = target_model.regressor_
    transformed = inner[:-1].transform(model.named_steps["features"].transform(X_validation))
    curve = [
        float(
            mean_absolute_error(
                y_validation,
                np.clip(
                    target_model.inverse_func(stage),
                    config.serving.prediction_lower_bound,
                    config.serving.prediction_upper_bound,
                ),
            )
        )
        for stage in inner.named_steps["hgb"].staged_predict(transformed)
    ]
    return int(np.argmin(curve)) + 1, curve


def evaluate_champion(
    model_path: Path | None,
    metadata_path: Path | None,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    candidate_prediction: np.ndarray,
    config: AppConfig,
) -> ChampionComparison:
    if model_path is None or metadata_path is None:
        return ChampionComparison(status="skipped", detail="no champion supplied")
    if not model_path.exists() or not metadata_path.exists():
        return ChampionComparison(status="skipped", detail="champion artifact not found")
    try:
        champion, metadata = load_artifact(model_path, metadata_path)
        prediction = predict_duration(champion, X_test, config.serving)
    except Exception as exc:  # any failure must block promotion, not silently pass
        return ChampionComparison(status="failed", detail=f"{type(exc).__name__}: {exc}")
    actual = y_test.to_numpy(dtype=float)
    paired_delta = np.abs(actual - candidate_prediction) - np.abs(actual - prediction)
    hour_clusters = pd.to_datetime(X_test[PICKUP]).dt.floor("h").to_numpy()
    try:
        statistic = paired_delta_upper_bound(
            paired_delta, hour_clusters, config.gates.champion_confidence_level
        )
    except ValueError as exc:
        return ChampionComparison(status="failed", detail=f"comparison statistic: {exc}")
    champion_mae = float(mean_absolute_error(y_test, prediction))
    return ChampionComparison(
        status="compared",
        champion_test_mae=champion_mae,
        champion_version=str(metadata.get("model_version")),
        candidate_test_mae=float(mean_absolute_error(y_test, candidate_prediction)),
        mae_delta=statistic.mean,
        mae_delta_upper_confidence=statistic.upper_confidence,
        mae_delta_standard_error=statistic.standard_error,
        noninferiority_margin_minutes=champion_mae * config.gates.max_regression_vs_champion,
        clusters=statistic.clusters,
        confidence_level=config.gates.champion_confidence_level,
    )


def _quality_summary(reports: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    input_rows = sum(int(report["input_rows"]) for report in reports)
    valid_rows = sum(int(report["valid_rows"]) for report in reports)
    rejected_rows = sum(int(report["rejected_rows"]) for report in reports)
    out_of_period_rows = sum(int(report["out_of_period_rows"]) for report in reports)
    duplicate_rows = sum(int(report["duplicate_rows"]) for report in reports)
    denominator = max(input_rows, 1)
    return {
        "input_rows": input_rows,
        "valid_rows": valid_rows,
        "rejected_rows": rejected_rows,
        "rejection_rate": rejected_rows / denominator,
        "out_of_period_rows": out_of_period_rows,
        "out_of_period_rate": out_of_period_rows / denominator,
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": duplicate_rows / denominator,
    }


def train_from_parquet(
    data_path: Path | Sequence[Path],
    config: AppConfig,
    config_path: Path,
    model_path: Path,
    metadata_path: Path,
    metrics_path: Path,
    model_version: str,
    *,
    period: Period | None = None,
    champion_model_path: Path | None = None,
    champion_metadata_path: Path | None = None,
    reference_path: Path | None = None,
) -> dict[str, Any]:
    data_paths = [data_path] if isinstance(data_path, Path) else list(data_path)
    if not data_paths:
        raise ValueError("at least one training data path is required")
    if period is not None and len(data_paths) != 1:
        raise ValueError("an explicit period can only be used with one data file")
    frames: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    periods: list[str | None] = []
    for path in data_paths:
        source_period = period if len(data_paths) == 1 else Period.from_filename(path)
        if len(data_paths) > 1 and source_period is None:
            raise ValueError(f"cannot infer period from rolling-window file: {path}")
        clean_frame, quality_report = validate_and_clean(
            pd.read_parquet(path), config.data, source_period
        )
        frames.append(clean_frame)
        quality_payload = asdict(quality_report)
        quality_payload["out_of_period_rate"] = quality_report.out_of_period_rate
        quality_payload["duplicate_rate"] = quality_report.duplicate_rate
        reports.append(quality_payload)
        periods.append(str(source_period) if source_period else None)
    clean = pd.concat(frames, ignore_index=True).sort_values(PICKUP, kind="stable")
    clean.reset_index(drop=True, inplace=True)
    quality = _quality_summary(reports)
    source_files = {path.name: sha256_file(path) for path in data_paths}
    split = chronological_split(clean, config.split)
    columns = list(INFERENCE_COLUMNS)
    X_train, y_train = split.train.loc[:, columns], split.train[TARGET]
    X_val, y_val = split.validation.loc[:, columns], split.validation[TARGET]
    X_test, y_test = split.test.loc[:, columns], split.test[TARGET]

    probe = build_model(config).fit(X_train, y_train)
    best_iter, curve = select_iterations(probe, X_val, y_val, config)
    model = (
        probe if best_iter == len(curve) else build_model(config, best_iter).fit(X_train, y_train)
    )

    validation_pred = predict_duration(model, X_val, config.serving)
    test_pred = predict_duration(model, X_test, config.serving)

    baseline = DummyRegressor(strategy="median")
    baseline.fit(np.zeros((len(split.train), 1)), y_train)
    baseline_pred = baseline.predict(np.zeros((len(split.validation), 1)))

    validation_metrics = regression_metrics(y_val, validation_pred)
    test_metrics = regression_metrics(y_test, test_pred)
    baseline_mae = regression_metrics(y_val, baseline_pred)["mae_minutes"]
    champion = evaluate_champion(
        champion_model_path, champion_metadata_path, X_test, y_test, test_pred, config
    )
    gate = evaluate_release_gates(
        validation_metrics, test_metrics, baseline_mae, config.gates, champion, quality
    )

    metrics: dict[str, Any] = {
        "data_quality": quality,
        "data_quality_by_file": reports,
        "source_periods": periods,
        "source_files_sha256": source_files,
        "rows": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "periods": {
            "train_start": str(split.train[PICKUP].min()),
            "train_end": str(split.train[PICKUP].max()),
            "validation_start": str(split.validation[PICKUP].min()),
            "validation_end": str(split.validation[PICKUP].max()),
            "test_start": str(split.test[PICKUP].min()),
            "test_end": str(split.test[PICKUP].max()),
        },
        "model": {
            "zone_encoding": config.model.zone_encoding,
            "selected_iterations": best_iter,
            "max_iterations": len(curve),
        },
        "baseline_validation_mae_minutes": baseline_mae,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_breakdown": error_breakdown(X_test, y_test, test_pred, seen_routes(model)),
        "champion": asdict(champion),
        "release_gate": {"passed": gate.passed, "failures": list(gate.failures)},
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if not gate.passed:
        raise RuntimeError("release gates failed: " + "; ".join(gate.failures))

    flat_metrics = {f"validation_{k}": v for k, v in validation_metrics.items()}
    flat_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
    metadata = new_metadata(
        model_version=model_version,
        git_sha=_git_sha(),
        data_sha256=hash_text(json.dumps(source_files, sort_keys=True)),
        config_sha256=hash_text(canonical_config_json(config)),
        metrics=flat_metrics,
    )
    save_artifact(model, metadata, model_path, metadata_path)
    build_reference_profile(
        split.train,
        reference_path or config.monitoring.reference_path,
        model=model,
        serving=config.serving,
    )
    LOGGER.info("training_completed", model_version=model_version, metrics=metrics)
    return metrics


def evaluate_artifact(
    data_path: Path,
    config: AppConfig,
    model_path: Path,
    metadata_path: Path,
    period: Period | None = None,
) -> dict[str, Any]:
    """Score a released artifact on a later source month (true out-of-period evaluation)."""
    period = period or Period.from_filename(data_path)
    model, metadata = load_artifact(model_path, metadata_path)
    clean, quality = validate_and_clean(pd.read_parquet(data_path), config.data, period)
    X, y = clean.loc[:, list(INFERENCE_COLUMNS)], clean[TARGET]
    prediction = predict_duration(model, X, config.serving)
    median_mae = float(np.abs(y - y.median()).mean())
    return {
        "model_version": metadata.get("model_version"),
        "data_quality": asdict(quality),
        "metrics": regression_metrics(y, prediction),
        "oracle_constant_median_mae_minutes": median_mae,
        "breakdown": error_breakdown(X, y, prediction, seen_routes(model)),
    }
