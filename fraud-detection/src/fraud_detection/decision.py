from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .config import CostConfig
from .modeling import predict_calibrated


def fraud_loss(amount, chargeback_fee: float):
    return np.asarray(amount, dtype=float) + chargeback_fee


def expected_costs(probability, amount, config: CostConfig):
    p = np.asarray(probability, dtype=float)
    loss = fraud_loss(amount, config.chargeback_fee)
    no_alert = p * loss
    alert = (
        config.review_cost
        + (1 - p) * config.legitimate_friction_cost
        + p * (1 - config.capture_rate) * loss
    )
    return no_alert, alert


def hourly_capacity_alerts(frame, value, daily_capacity: int, positive_only=True):
    values = np.asarray(value, dtype=float)
    hour = np.floor(frame["Time"].to_numpy(dtype=float) / 3600.0).astype(int)
    hourly_capacity = max(1, int(np.ceil(daily_capacity / 24)))
    alert = np.zeros(len(frame), dtype=bool)
    for current_hour in np.unique(hour):
        idx = np.flatnonzero(hour == current_hour)
        if positive_only:
            idx = idx[values[idx] > 0]
        if idx.size:
            ranked = idx[np.argsort(-values[idx], kind="mergesort")]
            alert[ranked[:hourly_capacity]] = True
    return alert


def realized_policy_metrics(y_true, amount, alert, config: CostConfig, probability=None):
    y = np.asarray(y_true, dtype=int)
    amount = np.asarray(amount, dtype=float)
    alert = np.asarray(alert, dtype=bool)
    loss = fraud_loss(amount, config.chargeback_fee)
    realized = np.where(
        alert,
        config.review_cost
        + (1 - y) * config.legitimate_friction_cost
        + y * (1 - config.capture_rate) * loss,
        y * loss,
    )
    fraud = y == 1
    approve_all = float((y * loss).sum())
    prevented_loss = float((loss[alert & fraud] * config.capture_rate).sum())
    review_cost_total = float(alert.sum() * config.review_cost)
    friction_total = float(((alert & ~fraud).sum()) * config.legitimate_friction_cost)
    result = {
        "transactions": int(len(y)),
        "alerts": int(alert.sum()),
        "alert_rate": float(alert.mean()),
        "true_fraud_alerts": int((alert & fraud).sum()),
        "false_positive_alerts": int((alert & ~fraud).sum()),
        "precision": float(y[alert].mean()) if alert.any() else 0.0,
        "fraud_count_recall": float(alert[fraud].mean()) if fraud.any() else 0.0,
        "fraud_amount_capture": float(amount[alert & fraud].sum() / amount[fraud].sum()) if amount[fraud].sum() else 0.0,
        "estimated_prevented_fraud_count": float((alert & fraud).sum() * config.capture_rate),
        "gross_prevented_loss": prevented_loss,
        "review_cost_total": review_cost_total,
        "friction_cost_total": friction_total,
        "approve_all_cost": approve_all,
        "realized_cost": float(realized.sum()),
    }
    result["net_savings"] = approve_all - result["realized_cost"]
    result["savings_per_1000_transactions"] = 1000 * result["net_savings"] / len(y)
    result["savings_per_alert"] = result["net_savings"] / max(result["alerts"], 1)
    result["review_roi"] = result["net_savings"] / max(review_cost_total + friction_total, 1e-9)
    if probability is not None:
        p = np.asarray(probability)
        result["mean_alert_probability"] = float(p[alert].mean()) if alert.any() else 0.0
    return result


def best_f1_threshold(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def make_policy_alerts(name, frame, probability, config: CostConfig, f1_threshold: float):
    p = np.asarray(probability, dtype=float)
    amount = frame["Amount"].to_numpy(dtype=float)
    no_alert, alert_cost = expected_costs(p, amount, config)
    avoidable = no_alert - alert_cost
    if name == "approve_all":
        return np.zeros(len(frame), dtype=bool)
    if name == "fixed_0.5":
        return p >= 0.5
    if name == "f1_threshold":
        return p >= f1_threshold
    if name == "probability_capacity":
        return hourly_capacity_alerts(frame, p, config.daily_alert_capacity, positive_only=False)
    if name == "expected_loss_capacity":
        return hourly_capacity_alerts(frame, p * fraud_loss(amount, config.chargeback_fee), config.daily_alert_capacity, positive_only=False)
    if name == "avoidable_value_capacity":
        return hourly_capacity_alerts(frame, avoidable, config.daily_alert_capacity, positive_only=True)
    raise ValueError(f"Unknown policy: {name}")


POLICIES = [
    "approve_all",
    "fixed_0.5",
    "f1_threshold",
    "probability_capacity",
    "expected_loss_capacity",
    "avoidable_value_capacity",
]


def evaluate_policies(frame, probability, config: CostConfig, f1_threshold):
    y = frame["Class"].to_numpy(dtype=int)
    amount = frame["Amount"].to_numpy(dtype=float)
    rows = []
    alerts = {}
    for name in POLICIES:
        alert = make_policy_alerts(name, frame, probability, config, f1_threshold)
        alerts[name] = alert
        rows.append({"policy": name, **realized_policy_metrics(y, amount, alert, config, probability)})
    return pd.DataFrame(rows).sort_values("realized_cost"), alerts


def bootstrap_savings_interval(frame, alert, config: CostConfig, repetitions=500, seed=42):
    y = frame["Class"].to_numpy(dtype=int)
    amount = frame["Amount"].to_numpy(dtype=float)
    alert = np.asarray(alert, dtype=bool)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    savings = []
    for _ in range(repetitions):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        savings.append(realized_policy_metrics(y[idx], amount[idx], alert[idx], config)["net_savings"])
    low, median, high = np.quantile(savings, [0.025, 0.5, 0.975])
    return {"low": float(low), "median": float(median), "high": float(high), "repetitions": repetitions}


def sensitivity_table(frame, probability, base: CostConfig, f1_threshold):
    rows = []
    for capture in [0.50, 0.70, 0.80, 0.90, 0.95]:
        for friction in [2.0, 5.0, 8.0, 15.0, 25.0]:
            config = CostConfig(
                review_cost=base.review_cost,
                legitimate_friction_cost=friction,
                chargeback_fee=base.chargeback_fee,
                capture_rate=capture,
                daily_alert_capacity=base.daily_alert_capacity,
                currency_label=base.currency_label,
            )
            alert = make_policy_alerts("avoidable_value_capacity", frame, probability, config, f1_threshold)
            rows.append({
                "capture_rate": capture,
                "friction_cost": friction,
                **realized_policy_metrics(frame["Class"], frame["Amount"], alert, config, probability),
            })
    return pd.DataFrame(rows)


def run_decision_analysis(model, calibrator, parts, config: CostConfig, artifact_dir, bootstrap_repetitions=500, seed=42):
    artifact_dir = Path(artifact_dir)
    policy_frame = parts["policy"]
    p_policy = predict_calibrated(model, calibrator, policy_frame)
    f1_threshold = best_f1_threshold(policy_frame["Class"], p_policy)
    policy_results, _ = evaluate_policies(policy_frame, p_policy, config, f1_threshold)
    selected_policy = str(policy_results.iloc[0]["policy"])
    policy_results.to_csv(artifact_dir / "policy_selection_results.csv", index=False)

    sensitivity = sensitivity_table(policy_frame, p_policy, config, f1_threshold)
    sensitivity.to_csv(artifact_dir / "policy_sensitivity.csv", index=False)

    test_frame = parts["test"]
    p_test = predict_calibrated(model, calibrator, test_frame)
    test_results, test_alerts = evaluate_policies(test_frame, p_test, config, f1_threshold)
    test_results["selected_on_policy_block"] = test_results["policy"].eq(selected_policy)
    test_results.to_csv(artifact_dir / "final_test_policy_results.csv", index=False)
    np.savez_compressed(
        artifact_dir / "final_test_predictions.npz",
        y=test_frame["Class"].to_numpy(dtype=int),
        probability=p_test,
        amount=test_frame["Amount"].to_numpy(dtype=float),
        time=test_frame["Time"].to_numpy(dtype=float),
        selected_alert=test_alerts[selected_policy],
    )
    interval = bootstrap_savings_interval(
        test_frame,
        test_alerts[selected_policy],
        config,
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    selected_row = test_results.loc[test_results["policy"].eq(selected_policy)].iloc[0].to_dict()
    business_case = {
        "selected_policy": selected_policy,
        "policy_selection_block": "policy",
        "final_evaluation_block": "test",
        "f1_threshold": f1_threshold,
        "cost_assumptions": asdict(config),
        "test_metrics": selected_row,
        "net_savings_95pct_bootstrap_interval": interval,
        "illustrative_savings_per_million_transactions": float(selected_row["savings_per_1000_transactions"] * 1000),
        "claim_scope": "Scenario estimate on the public ULB dataset; not realized savings from an identified bank.",
    }
    (artifact_dir / "business_case.json").write_text(json.dumps(business_case, indent=2), encoding="utf-8")
    policy_config = {
        "selected_policy": selected_policy,
        "f1_threshold": f1_threshold,
        "hourly_capacity": int(np.ceil(config.daily_alert_capacity / 24)),
        "ranking_window": "one elapsed-hour bucket",
        **asdict(config),
    }
    (artifact_dir / "decision_policy.json").write_text(json.dumps(policy_config, indent=2), encoding="utf-8")
    return business_case, policy_results, test_results, sensitivity

