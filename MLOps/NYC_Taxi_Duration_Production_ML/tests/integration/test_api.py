import json
from collections.abc import Iterator
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from taxi_duration.config import load_config
from taxi_duration.models.artifact import runtime_versions
from taxi_duration.models.factory import build_model

VALID_PAYLOAD = {
    "pickup_datetime": "2025-01-15T08:30:00-05:00",
    "pickup_location_id": 10,
    "dropoff_location_id": 30,
    "passenger_count": 1,
}


def _write_artifact(
    directory: Path, trips: pd.DataFrame, versions: dict[str, str]
) -> tuple[Path, Path]:
    target = (
        trips["tpep_dropoff_datetime"] - trips["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    model = build_model(load_config("configs/development.yaml"))
    model.fit(trips, target)
    model_path = directory / "model.joblib"
    metadata_path = directory / "metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": "test-v1",
                "feature_contract_version": "1.1.0",
                "runtime_versions": versions,
            }
        )
    )
    return model_path, metadata_path


def _client(monkeypatch: pytest.MonkeyPatch, model_path: Path, metadata_path: Path) -> TestClient:
    monkeypatch.setenv("TAXI_MODEL_PATH", str(model_path))
    monkeypatch.setenv("TAXI_METADATA_PATH", str(metadata_path))
    monkeypatch.setenv("TAXI_CONFIG_PATH", "configs/development.yaml")
    from taxi_duration.serving.app import app

    return TestClient(app)


@pytest.fixture()
def client(
    tmp_path: Path, synthetic_trips: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    paths = _write_artifact(tmp_path, synthetic_trips, runtime_versions())
    with _client(monkeypatch, *paths) as test_client:
        yield test_client


def _counter(client: TestClient, status: str) -> float:
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(f'taxi_prediction_requests_total{{status="{status}"}}'):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


@pytest.mark.integration
def test_prediction_endpoint(client: TestClient) -> None:
    response = client.post("/v1/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert 1 <= response.json()["predicted_duration_minutes"] <= 120
    assert response.json()["feature_contract_version"] == "1.1.0"
    assert client.get("/health/ready").status_code == 200


@pytest.mark.integration
def test_contract_violations_are_counted(client: TestClient) -> None:
    before = _counter(client, "invalid")
    naive_time = {**VALID_PAYLOAD, "pickup_datetime": "2025-01-15T08:30:00"}
    assert client.post("/v1/predict", json=naive_time).status_code == 422
    assert _counter(client, "invalid") == before + 1


@pytest.mark.integration
@pytest.mark.parametrize("zone", [264, 265])
def test_non_routable_zones_are_rejected(client: TestClient, zone: int) -> None:
    payload = {**VALID_PAYLOAD, "dropoff_location_id": zone}
    assert client.post("/v1/predict", json=payload).status_code == 422


@pytest.mark.integration
@pytest.mark.parametrize("library", ["scikit-learn", "numpy", "pandas", "joblib"])
def test_runtime_mismatch_keeps_service_unready(
    tmp_path: Path,
    synthetic_trips: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    library: str,
) -> None:
    versions = {**runtime_versions(), library: "0.0.0"}
    paths = _write_artifact(tmp_path, synthetic_trips, versions)
    with _client(monkeypatch, *paths) as test_client:
        assert test_client.get("/health/live").status_code == 200
        assert test_client.get("/health/ready").status_code == 503
        assert test_client.post("/v1/predict", json=VALID_PAYLOAD).status_code == 503
