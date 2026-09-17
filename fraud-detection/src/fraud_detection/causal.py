from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CostConfig
from .modeling import predict_calibrated


def standardized_mean_difference(x, treatment, weights=None):
    x = np.asarray(x, dtype=float)
    t = np.asarray(treatment, dtype=int)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)

    def moments(mask):
        wm = w[mask]
        xv = x[mask]
        mean = np.average(xv, weights=wm)
        variance = np.average((xv - mean) ** 2, weights=wm)
        return mean, variance

    m1, v1 = moments(t == 1)
    m0, v0 = moments(t == 0)
    pooled = np.sqrt((v1 + v0) / 2)
    return float((m1 - m0) / pooled) if pooled > 0 else 0.0


def _nuisance_models(seed=42, max_iter=80):
    propensity = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2_000, C=1.0, random_state=seed)),
    ])
    outcome_1 = HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed
    )
    outcome_0 = HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed + 1
    )
    return propensity, outcome_1, outcome_0


def cross_fitted_aipw(X, treatment, outcome, n_splits=3, seed=42, max_iter=80):
    X = np.asarray(X, dtype=float)
    t = np.asarray(treatment, dtype=int)
    y = np.asarray(outcome, dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ps = np.empty(len(y))
    mu1 = np.empty(len(y))
    mu0 = np.empty(len(y))
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, t)):
        propensity, m1, m0 = _nuisance_models(seed + fold * 10, max_iter=max_iter)
        propensity.fit(X[train_idx], t[train_idx])
        m1.fit(X[train_idx][t[train_idx] == 1], y[train_idx][t[train_idx] == 1])
        m0.fit(X[train_idx][t[train_idx] == 0], y[train_idx][t[train_idx] == 0])
        ps[test_idx] = propensity.predict_proba(X[test_idx])[:, 1]
        mu1[test_idx] = m1.predict(X[test_idx])
        mu0[test_idx] = m0.predict(X[test_idx])
    ps_raw = ps.copy()
    ps = np.clip(ps, 0.01, 0.99)
    score = mu1 - mu0 + t * (y - mu1) / ps - (1 - t) * (y - mu0) / (1 - ps)
    return {
        "ate": float(score.mean()),
        "score": score,
        "ps": ps,
        "ps_raw": ps_raw,
        "mu1": mu1,
        "mu0": mu0,
        "gcomp": float(np.mean(mu1 - mu0)),
    }


def bootstrap_aipw_refit(X, treatment, outcome, repetitions=30, seed=43):
    rng = np.random.default_rng(seed)
    n = len(outcome)
    values = []
    for b in range(repetitions):
        idx = rng.integers(0, n, n)
        estimate = cross_fitted_aipw(
            np.asarray(X)[idx], np.asarray(treatment)[idx], np.asarray(outcome)[idx],
            n_splits=2, seed=seed + b, max_iter=40,
        )
        values.append(estimate["ate"])
    low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
    return {"low": float(low), "median": float(median), "high": float(high), "repetitions": repetitions}


def run_causal_demo(model, calibrator, parts, cost: CostConfig, artifact_dir, bootstrap_repetitions=30, seed=42):
    artifact_dir = Path(artifact_dir)
    frame = parts["test"].copy().reset_index(drop=True)
    risk = predict_calibrated(model, calibrator, frame)
    amount = frame["Amount"].to_numpy(dtype=float)
    fraud = frame["Class"].to_numpy(dtype=int)
    hour = (frame["Time"].to_numpy(dtype=float) / 3600.0) % 24
    log_amount = np.log1p(amount)
    z_amount = (log_amount - log_amount.mean()) / (log_amount.std() + 1e-12)
    risk_rank = pd.Series(risk).rank(pct=True).to_numpy()
    rng = np.random.default_rng(seed)

    true_ps = expit(-4.2 + 5.0 * risk_rank + 0.35 * z_amount)
    treatment = rng.binomial(1, true_ps)
    capture_probability = np.clip(0.88 - 0.08 * expit(z_amount), 0.65, 0.90)
    caught = rng.binomial(1, capture_probability) * fraud
    loss = amount + cost.chargeback_fee
    y0 = fraud * loss
    y1 = cost.review_cost + (1 - fraud) * cost.legitimate_friction_cost + fraud * (1 - caught) * loss
    observed = treatment * y1 + (1 - treatment) * y0
    truth = y1 - y0

    X = np.column_stack([
        risk,
        log_amount,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
    ])
    names = ["risk_score", "log_amount", "hour_sin", "hour_cos"]
    result = cross_fitted_aipw(X, treatment, observed, n_splits=3, seed=seed, max_iter=80)
    ps = result["ps"]
    p_t = treatment.mean()
    weights = np.where(treatment == 1, p_t / ps, (1 - p_t) / (1 - ps))
    cap = np.quantile(weights, 0.995)
    weights = np.minimum(weights, cap)
    ess = float(weights.sum() ** 2 / np.sum(weights ** 2))
    iptw_mu1 = np.sum(weights * treatment * observed) / np.sum(weights * treatment)
    iptw_mu0 = np.sum(weights * (1 - treatment) * observed) / np.sum(weights * (1 - treatment))
    iptw = float(iptw_mu1 - iptw_mu0)
    naive = float(observed[treatment == 1].mean() - observed[treatment == 0].mean())

    logit_ps = logit(ps)
    treated_idx = np.flatnonzero(treatment == 1)
    control_idx = np.flatnonzero(treatment == 0)
    nn = NearestNeighbors(n_neighbors=1).fit(logit_ps[control_idx].reshape(-1, 1))
    distance, neighbor = nn.kneighbors(logit_ps[treated_idx].reshape(-1, 1))
    keep = distance.ravel() <= 0.2 * np.std(logit_ps)
    matched_treated = treated_idx[keep]
    matched_control = control_idx[neighbor.ravel()[keep]]
    matching_att = float(np.mean(observed[matched_treated] - observed[matched_control]))

    balance_rows = []
    matched_mask_t = np.zeros(len(frame), dtype=bool)
    matched_mask_c = np.zeros(len(frame), dtype=bool)
    matched_mask_t[matched_treated] = True
    matched_mask_c[matched_control] = True
    for j, name in enumerate(names):
        matched_x = np.concatenate([X[matched_treated, j], X[matched_control, j]])
        matched_t = np.concatenate([np.ones(len(matched_treated)), np.zeros(len(matched_control))])
        balance_rows.append({
            "feature": name,
            "smd_before": standardized_mean_difference(X[:, j], treatment),
            "smd_after_iptw": standardized_mean_difference(X[:, j], treatment, weights),
            "smd_after_matching": standardized_mean_difference(matched_x, matched_t),
        })
    balance = pd.DataFrame(balance_rows)
    balance.to_csv(artifact_dir / "causal_balance.csv", index=False)

    influence_se = float(np.std(result["score"], ddof=1) / np.sqrt(len(result["score"])))
    refit_interval = bootstrap_aipw_refit(X, treatment, observed, repetitions=bootstrap_repetitions, seed=seed + 1)
    estimates = pd.DataFrame([
        {"estimand": "ATE", "estimator": "truth", "estimate": float(truth.mean())},
        {"estimand": "ATE", "estimator": "naive", "estimate": naive},
        {"estimand": "ATE", "estimator": "IPTW", "estimate": iptw},
        {"estimand": "ATE", "estimator": "g-computation", "estimate": result["gcomp"]},
        {"estimand": "ATE", "estimator": "cross-fitted AIPW", "estimate": result["ate"]},
        {"estimand": "ATT", "estimator": "truth", "estimate": float(truth[treatment == 1].mean())},
        {"estimand": "ATT", "estimator": "PS matching", "estimate": matching_att},
    ])
    estimates.to_csv(artifact_dir / "causal_estimates.csv", index=False)
    diagnostics = {
        "semi_synthetic": True,
        "treatment_rate": float(treatment.mean()),
        "true_ate": float(truth.mean()),
        "true_att": float(truth[treatment == 1].mean()),
        "aipw_ate": result["ate"],
        "aipw_influence_function_95pct_ci": [result["ate"] - 1.96 * influence_se, result["ate"] + 1.96 * influence_se],
        "aipw_full_refit_bootstrap_95pct_ci": refit_interval,
        "iptw_effective_sample_size": ess,
        "unclipped_propensity_min": float(result["ps_raw"].min()),
        "unclipped_propensity_max": float(result["ps_raw"].max()),
        "matching_treated_retained": int(keep.sum()),
        "matching_treated_total": int(len(treated_idx)),
        "scope": "Estimator audit with constructed potential outcomes, not an identified effect from the public dataset.",
    }
    (artifact_dir / "causal_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    np.savez_compressed(
        artifact_dir / "causal_plot_data.npz",
        propensity=result["ps_raw"], treatment=treatment, weights=weights,
    )
    return estimates, balance, diagnostics

