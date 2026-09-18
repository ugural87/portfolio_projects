from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd
import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sklearn.pipeline import Pipeline

from taxi_duration.config import AppConfig, load_config
from taxi_duration.logging import configure_logging
from taxi_duration.models.artifact import load_artifact
from taxi_duration.models.predict import predict_duration
from taxi_duration.serving.metrics import LATENCY, PREDICTION, REQUESTS
from taxi_duration.serving.schemas import HealthResponse, PredictionRequest, PredictionResponse

configure_logging(os.getenv("TAXI_LOG_LEVEL", "INFO"))
LOGGER = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    config = load_config(os.getenv("TAXI_CONFIG_PATH", "configs/development.yaml"))
    model_path = Path(os.getenv("TAXI_MODEL_PATH", "artifacts/model.joblib"))
    metadata_path = Path(os.getenv("TAXI_METADATA_PATH", "artifacts/metadata.json"))
    try:
        model, metadata = load_artifact(model_path, metadata_path)
        application.state.model = model
        application.state.metadata = metadata
        application.state.config = config
        application.state.ready = True
        LOGGER.info("model_loaded", model_version=metadata["model_version"])
    except Exception as exc:
        # Any load failure (missing file, runtime mismatch, corrupt pickle) keeps the process
        # alive but not ready, so the orchestrator withholds traffic instead of crash-looping.
        application.state.ready = False
        application.state.load_error = f"{type(exc).__name__}: {exc}"
        LOGGER.error("model_load_failed", error=application.state.load_error)
    yield


app = FastAPI(
    title="NYC Taxi Duration API",
    version="1.2.3",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def count_contract_violations(request: Request, exc: RequestValidationError) -> Response:
    """Count 422s so the error-rate panel also sees contract violations, then use the default."""
    if request.url.path == "/v1/predict":
        REQUESTS.labels(status="invalid").inc()
    return await request_validation_exception_handler(request, exc)


@app.middleware("http")
async def request_context(request: Request, call_next: Any) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(request_id=request_id)
    started = time.perf_counter()
    status_code = 500
    try:
        response = cast(Response, await call_next(request))
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - started
        LOGGER.info(
            "request_completed",
            path=request.url.path,
            status_code=status_code,
            latency_seconds=elapsed,
        )
        structlog.contextvars.clear_contextvars()


@app.get("/health/live", response_model=HealthResponse)
def live() -> HealthResponse:
    return HealthResponse(status="alive")


@app.get("/health/ready", response_model=HealthResponse)
def ready(request: Request) -> HealthResponse:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="model unavailable")
    return HealthResponse(
        status="ready", model_version=str(request.app.state.metadata["model_version"])
    )


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    if not getattr(request.app.state, "ready", False):
        REQUESTS.labels(status="unavailable").inc()
        raise HTTPException(status_code=503, detail="model unavailable")
    started = time.perf_counter()
    try:
        config: AppConfig = request.app.state.config
        local_time = payload.pickup_datetime.astimezone(ZoneInfo(config.timezone)).replace(
            tzinfo=None
        )
        features = pd.DataFrame(
            [
                {
                    "tpep_pickup_datetime": local_time,
                    "PULocationID": payload.pickup_location_id,
                    "DOLocationID": payload.dropoff_location_id,
                    "passenger_count": payload.passenger_count,
                }
            ]
        )
        model: Pipeline = request.app.state.model
        prediction = float(predict_duration(model, features, config.serving)[0])
        REQUESTS.labels(status="success").inc()
        PREDICTION.observe(prediction)
        metadata = request.app.state.metadata
        LOGGER.info(
            "prediction_completed",
            prediction_minutes=prediction,
            model_version=metadata["model_version"],
            pickup_location_id=payload.pickup_location_id,
            dropoff_location_id=payload.dropoff_location_id,
        )
        return PredictionResponse(
            predicted_duration_minutes=round(prediction, 2),
            model_version=str(metadata["model_version"]),
            feature_contract_version=str(metadata["feature_contract_version"]),
        )
    except Exception:
        REQUESTS.labels(status="error").inc()
        LOGGER.exception("prediction_failed")
        raise HTTPException(status_code=500, detail="prediction failed") from None
    finally:
        LATENCY.observe(time.perf_counter() - started)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
