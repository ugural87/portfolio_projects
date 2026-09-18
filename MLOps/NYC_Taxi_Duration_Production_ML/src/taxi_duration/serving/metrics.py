from prometheus_client import Counter, Histogram

REQUESTS = Counter("taxi_prediction_requests_total", "Prediction requests", labelnames=("status",))
LATENCY = Histogram(
    "taxi_prediction_latency_seconds",
    "Prediction endpoint latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
PREDICTION = Histogram(
    "taxi_predicted_duration_minutes",
    "Predicted trip duration",
    buckets=(1, 5, 10, 15, 20, 30, 45, 60, 90, 120),
)
