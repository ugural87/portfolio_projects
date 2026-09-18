from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from taxi_duration.config import GateConfig


@dataclass(frozen=True)
class PairedDelta:
    mean: float
    standard_error: float
    upper_confidence: float
    clusters: int


def paired_delta_upper_bound(
    delta: np.ndarray, clusters: np.ndarray, confidence_level: float
) -> PairedDelta:
    """One-sided upper confidence bound of mean(delta) with cluster-robust standard error.

    delta = |y - candidate| - |y - champion| per test row (positive = candidate worse).
    Rows in the same cluster (pickup hour) share traffic conditions, so their deltas are
    correlated; an i.i.d. standard error would overstate the evidence. The cluster-robust
    variance of the mean is  G/(G-1) * sum_g (sum_{i in g} (d_i - d_bar))^2 / n^2 .
    """
    values = np.asarray(delta, dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError("at least two paired rows are required")
    mean = float(values.mean())
    residual_sums = pd.Series(values - mean).groupby(np.asarray(clusters)).sum().to_numpy()
    groups = len(residual_sums)
    if groups < 2:
        raise ValueError("at least two clusters are required")
    variance = groups / (groups - 1) * float(np.sum(residual_sums**2)) / n**2
    standard_error = float(np.sqrt(variance))
    z_score = NormalDist().inv_cdf(confidence_level)
    return PairedDelta(mean, standard_error, mean + z_score * standard_error, groups)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class ChampionComparison:
    """Outcome of scoring the current production model on the candidate's test window.

    status: "compared" (champion_test_mae is set), "skipped" (no champion supplied, e.g. the
    first release) or "failed" (a champion was supplied but could not be evaluated).
    """

    status: str
    champion_test_mae: float | None = None
    champion_version: str | None = None
    candidate_test_mae: float | None = None
    mae_delta: float | None = None
    mae_delta_upper_confidence: float | None = None
    mae_delta_standard_error: float | None = None
    noninferiority_margin_minutes: float | None = None
    clusters: int | None = None
    confidence_level: float | None = None
    detail: str | None = None


def evaluate_release_gates(
    validation: dict[str, float],
    test: dict[str, float],
    baseline_validation_mae: float,
    config: GateConfig,
    champion: ChampionComparison | None = None,
    data_quality: Mapping[str, float | int] | None = None,
) -> GateResult:
    failures: list[str] = []
    if validation["mae_minutes"] > config.max_validation_mae_minutes:
        failures.append("validation MAE exceeds limit")
    if test["mae_minutes"] > config.max_test_mae_minutes:
        failures.append("test MAE exceeds limit")
    if test["p95_absolute_error_minutes"] > config.max_p95_absolute_error_minutes:
        failures.append("test p95 absolute error exceeds limit")
    improvement = 1 - validation["mae_minutes"] / baseline_validation_mae
    if improvement < config.min_improvement_vs_baseline:
        failures.append("candidate does not improve sufficiently over median baseline")
    if abs(test["mae_minutes"] - validation["mae_minutes"]) > config.max_validation_test_gap:
        failures.append("validation-to-test MAE gap exceeds limit")
    if champion is not None:
        if champion.status == "failed":
            failures.append(f"champion could not be evaluated: {champion.detail}")
        elif champion.status == "compared":
            # Non-inferiority: the candidate must show, at the configured confidence, that it is
            # not worse than the champion by more than the margin. A margin of zero would turn
            # this into a superiority test that blocks ~95% of equally good retrains.
            if champion.champion_test_mae is None or champion.mae_delta_upper_confidence is None:
                failures.append("champion comparison is incomplete")
            else:
                margin = champion.champion_test_mae * config.max_regression_vs_champion
                if champion.mae_delta_upper_confidence > margin:
                    failures.append(
                        "candidate is not shown to be non-inferior to the production champion"
                    )
    if data_quality is not None:
        if float(data_quality["rejection_rate"]) > config.max_rejection_rate:
            failures.append("data rejection rate exceeds limit")
        if float(data_quality["out_of_period_rate"]) > config.max_out_of_period_rate:
            failures.append("out-of-period row rate exceeds limit")
        if float(data_quality["duplicate_rate"]) > config.max_duplicate_rate:
            failures.append("duplicate row rate exceeds limit")
        if int(data_quality["valid_rows"]) < config.min_valid_rows:
            failures.append("valid row count is below minimum")
    return GateResult(passed=not failures, failures=tuple(failures))
