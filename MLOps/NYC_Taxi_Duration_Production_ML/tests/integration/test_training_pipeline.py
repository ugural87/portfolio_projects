import json
from pathlib import Path

import pandas as pd
import pytest

from taxi_duration.config import AppConfig, load_config
from taxi_duration.models.artifact import load_artifact
from taxi_duration.training.pipeline import evaluate_artifact, train_from_parquet


def _config() -> AppConfig:
    config = load_config(Path("configs/development.yaml"))
    config.gates.min_improvement_vs_baseline = 0.0
    config.gates.max_validation_test_gap = 20.0
    return config


def _train(
    tmp_path: Path, data_path: Path | list[Path], name: str, **kwargs: object
) -> dict[str, object]:
    return train_from_parquet(
        data_path,
        _config(),
        Path("configs/development.yaml"),
        tmp_path / name / "model.joblib",
        tmp_path / name / "metadata.json",
        tmp_path / name / "metrics.json",
        f"{name}-v1",
        reference_path=tmp_path / name / "reference_profile.json",
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.integration
def test_training_produces_loadable_artifacts(
    tmp_path: Path, synthetic_trips: pd.DataFrame
) -> None:
    data_path = tmp_path / "trips.parquet"
    synthetic_trips.to_parquet(data_path)
    metrics = _train(tmp_path, data_path, "candidate")

    _, metadata = load_artifact(
        tmp_path / "candidate" / "model.joblib", tmp_path / "candidate" / "metadata.json"
    )
    assert metadata["runtime_versions"]["scikit-learn"]
    assert metadata["feature_contract_version"] == "1.2.0"
    assert (tmp_path / "candidate" / "reference_profile.json").exists()
    model_info = metrics["model"]
    assert isinstance(model_info, dict)
    assert 1 <= model_info["selected_iterations"] <= model_info["max_iterations"]
    champion = metrics["champion"]
    assert isinstance(champion, dict)
    assert champion["status"] == "skipped"
    assert champion["detail"] == "no champion supplied"
    breakdown = metrics["test_breakdown"]
    assert isinstance(breakdown, dict)
    assert "by_route_coverage" in breakdown


@pytest.mark.integration
def test_champion_is_scored_on_candidate_test_window(
    tmp_path: Path, synthetic_trips: pd.DataFrame
) -> None:
    data_path = tmp_path / "trips.parquet"
    synthetic_trips.to_parquet(data_path)
    _train(tmp_path, data_path, "champion")
    metrics = _train(
        tmp_path,
        data_path,
        "candidate",
        champion_model_path=tmp_path / "champion" / "model.joblib",
        champion_metadata_path=tmp_path / "champion" / "metadata.json",
    )
    champion = metrics["champion"]
    assert isinstance(champion, dict)
    assert champion["status"] == "compared"
    assert champion["champion_version"] == "champion-v1"
    # Identical data and seed -> identical predictions: zero delta, inside any positive margin.
    assert champion["mae_delta"] == 0.0
    assert champion["clusters"] > 1
    assert champion["noninferiority_margin_minutes"] > 0
    assert metrics["release_gate"]["passed"]


@pytest.mark.integration
def test_out_of_period_evaluation(tmp_path: Path, synthetic_trips: pd.DataFrame) -> None:
    data_path = tmp_path / "trips.parquet"
    synthetic_trips.to_parquet(data_path)
    _train(tmp_path, data_path, "candidate")
    later = synthetic_trips.copy()
    shift = pd.Timedelta(days=31)
    later["tpep_pickup_datetime"] += shift
    later["tpep_dropoff_datetime"] += shift
    later_path = tmp_path / "later.parquet"
    later.to_parquet(later_path)

    report = evaluate_artifact(
        later_path,
        _config(),
        tmp_path / "candidate" / "model.joblib",
        tmp_path / "candidate" / "metadata.json",
    )
    assert report["metrics"]["mae_minutes"] > 0
    assert set(report["breakdown"]) >= {"by_pickup_hour", "by_weekday", "by_route_coverage"}
    json.dumps(report)  # report must be serialisable


@pytest.mark.integration
def test_rolling_window_records_all_source_periods(
    tmp_path: Path, synthetic_trips: pd.DataFrame
) -> None:
    paths: list[Path] = []
    for offset, period in enumerate(("2025-01", "2025-02", "2025-03")):
        frame = synthetic_trips.copy()
        shift = pd.DateOffset(months=offset)
        frame["tpep_pickup_datetime"] += shift
        frame["tpep_dropoff_datetime"] += shift
        path = tmp_path / f"yellow_tripdata_{period}.parquet"
        frame.to_parquet(path)
        paths.append(path)
    metrics = _train(tmp_path, paths, "rolling")
    assert metrics["source_periods"] == ["2025-01", "2025-02", "2025-03"]
    assert metrics["data_quality"]["valid_rows"] == 3 * len(synthetic_trips)
