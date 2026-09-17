import numpy as np
import pandas as pd

from fraud_detection.config import CostConfig, SplitConfig
from fraud_detection.decision import expected_costs, hourly_capacity_alerts, realized_policy_metrics
from fraud_detection.features import make_features
from fraud_detection.metrics import metric_bundle
from fraud_detection.splitting import temporal_partitions


def synthetic_frame(n=1000):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"Time": np.arange(n, dtype=float) * 60})
    for i in range(1, 29):
        frame[f"V{i}"] = rng.normal(size=n)
    frame["Amount"] = rng.lognormal(3, 1, size=n)
    frame["Class"] = 0
    frame.loc[np.arange(0, n, 50), "Class"] = 1
    return frame


def test_features_are_scoring_time_features():
    X = make_features(synthetic_frame(100))
    assert "Time" not in X
    assert {"Amount", "LogAmount", "HourSin", "HourCos"}.issubset(X.columns)
    assert X.shape == (100, 32)


def test_temporal_partitions_are_disjoint_and_ordered():
    parts = temporal_partitions(synthetic_frame(), SplitConfig())
    names = list(parts)
    assert names == ["train", "validation", "calibration", "policy", "test"]
    for left, right in zip(names[:-1], names[1:]):
        assert parts[left].Time.max() < parts[right].Time.min()
    assert sum(map(len, parts.values())) == 1000


def test_hourly_capacity_is_enforced():
    frame = synthetic_frame(240)
    value = np.arange(len(frame), dtype=float)
    alert = hourly_capacity_alerts(frame, value, daily_capacity=24, positive_only=False)
    hours = (frame.Time // 3600).astype(int)
    assert pd.Series(alert).groupby(hours).sum().max() <= 1


def test_cost_accounting_reconciles():
    config = CostConfig(review_cost=4, legitimate_friction_cost=8, chargeback_fee=15, capture_rate=0.8)
    y = np.array([1, 0, 1, 0])
    amount = np.array([100.0, 20.0, 50.0, 40.0])
    alert = np.array([True, True, False, False])
    result = realized_policy_metrics(y, amount, alert, config)
    expected_realized = (4 + 0.2 * 115) + (4 + 8) + 65
    assert np.isclose(result["realized_cost"], expected_realized)
    assert np.isclose(result["net_savings"], (115 + 65) - expected_realized)


def test_brier_sign_and_pr_lift():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.2, 0.8, 0.99])
    metrics = metric_bundle(y, p)
    assert metrics["brier"] >= 0
    assert metrics["pr_auc_lift"] >= 1


def test_expected_cost_prefers_review_for_high_risk_large_amount():
    config = CostConfig()
    no_alert, alert = expected_costs([0.99], [1000], config)
    assert alert[0] < no_alert[0]

