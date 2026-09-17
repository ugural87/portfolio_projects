from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metric_bundle(y_true, probability, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    pred = (p >= threshold).astype(int)
    prevalence = float(y.mean())
    brier = float(brier_score_loss(y, p))
    null_brier = prevalence * (1 - prevalence)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "pr_auc_lift": float(average_precision_score(y, p) / prevalence),
        "brier": brier,
        "brier_skill": float(1 - brier / null_brier) if null_brier > 0 else 0.0,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "threshold": float(threshold),
        "alerts": int(pred.sum()),
        "prevalence": prevalence,
    }


def bootstrap_metric_interval(y_true, probability, metric="pr_auc", repetitions=500, seed=42):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    values = []
    for _ in range(repetitions):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        if metric == "pr_auc":
            values.append(average_precision_score(y[idx], p[idx]))
        elif metric == "roc_auc":
            values.append(roc_auc_score(y[idx], p[idx]))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.5, 0.975]))

