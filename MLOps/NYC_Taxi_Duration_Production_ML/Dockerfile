ARG PYTHON_IMAGE=python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534
FROM ${PYTHON_IMAGE} AS builder
WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
# The same lock used for training: the image must unpickle the model with the exact
# scikit-learn/numpy that produced it (also enforced at load time via metadata).
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
RUN python -m pip wheel --wheel-dir /wheels --require-hashes -r requirements.lock && \
    python -m pip wheel --wheel-dir /wheels --no-deps .

FROM ${PYTHON_IMAGE} AS runtime
ARG APP_UID=10001
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TAXI_CONFIG_PATH=/app/configs/production.yaml \
    TAXI_MODEL_PATH=/app/artifacts/model.joblib \
    TAXI_METADATA_PATH=/app/artifacts/metadata.json
RUN adduser --disabled-password --gecos "" --uid ${APP_UID} appuser
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY configs ./configs
COPY artifacts/model.joblib artifacts/model.joblib
COPY artifacts/metadata.json artifacts/metadata.json
COPY artifacts/reference_profile.json artifacts/reference_profile.json
RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"
# One worker per container. prometheus_client keeps metrics per process, so several workers
# behind one /metrics endpoint return a different worker's counters on each scrape. Horizontal
# scale comes from Kubernetes replicas, each scraped as its own target.
CMD ["uvicorn", "taxi_duration.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
