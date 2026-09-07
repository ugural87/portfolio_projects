from __future__ import annotations

import numpy as np
from scipy import stats

from ..progress import progress


def diebold_mariano(
    truth: np.ndarray,
    forecast_a: np.ndarray,
    forecast_b: np.ndarray,
    hac_lag: int = 0,
) -> dict[str, float]:
    """Paired absolute-error DM test; event-series default is HAC lag zero."""
    differential = np.abs(truth - forecast_a) - np.abs(truth - forecast_b)
    n = len(differential)
    centered = differential - differential.mean()
    variance = float(np.dot(centered, centered) / n)
    for lag in range(1, min(hac_lag, n - 1) + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        variance += 2.0 * (1.0 - lag / (hac_lag + 1.0)) * covariance
    standard_error = np.sqrt(max(variance, 1e-12) / n)
    statistic = float(differential.mean() / standard_error)
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=max(n - 1, 1)))
    return {
        "dm_statistic": statistic,
        "p_value": p_value,
        "mean_loss_difference": float(differential.mean()),
    }


def fold_block_bootstrap_mae_difference(
    frame,
    forecast_a: str,
    forecast_b: str,
    repetitions: int = 5_000,
    seed: int = 42,
) -> dict[str, float]:
    fold_ids = np.asarray(sorted(frame.fold.unique()))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in progress(
        range(repetitions),
        desc=f"Bootstrap {forecast_a} vs {forecast_b}",
        total=repetitions,
        unit="draw",
        leave=False,
    ):
        chosen = rng.choice(fold_ids, size=len(fold_ids), replace=True)
        sample = [frame.loc[frame.fold == fold] for fold in chosen]
        pooled = __import__("pandas").concat(sample, ignore_index=True)
        truth = pooled.truth_bp.to_numpy()
        difference = np.mean(np.abs(truth - pooled[forecast_a])) - np.mean(
            np.abs(truth - pooled[forecast_b])
        )
        draws.append(difference)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"mean_difference": float(np.mean(draws)), "ci_low": float(low), "ci_high": float(high)}
