import numpy as np
import pytest

from taxi_duration.config import load_config
from taxi_duration.training.gates import (
    ChampionComparison,
    evaluate_release_gates,
    paired_delta_upper_bound,
)

GOOD_VALIDATION = {"mae_minutes": 4.0}
GOOD_TEST = {"mae_minutes": 4.5, "p95_absolute_error_minutes": 15.0}


def test_release_gate_passes_good_candidate() -> None:
    config = load_config("configs/development.yaml").gates
    result = evaluate_release_gates(GOOD_VALIDATION, GOOD_TEST, 7.0, config)
    assert result.passed


def test_release_gate_explains_failure() -> None:
    config = load_config("configs/development.yaml").gates
    result = evaluate_release_gates(
        {"mae_minutes": 14.0},
        {"mae_minutes": 20.0, "p95_absolute_error_minutes": 40.0},
        baseline_validation_mae=14.1,
        config=config,
    )
    assert not result.passed
    assert len(result.failures) >= 2


def _compared(champion_mae: float, upper: float) -> ChampionComparison:
    return ChampionComparison(
        status="compared", champion_test_mae=champion_mae, mae_delta_upper_confidence=upper
    )


def test_candidate_not_shown_non_inferior_is_blocked() -> None:
    config = load_config("configs/development.yaml").gates  # margin 1% of champion MAE
    # champion MAE 5.0 -> margin 0.05 min; upper bound 0.08 exceeds it.
    result = evaluate_release_gates(GOOD_VALIDATION, GOOD_TEST, 7.0, config, _compared(5.0, 0.08))
    assert not result.passed
    assert "non-inferior" in result.failures[0]


def test_equal_candidate_within_margin_passes() -> None:
    config = load_config("configs/development.yaml").gates
    # Upper bound slightly above zero (no evidence of improvement) but inside the margin.
    assert evaluate_release_gates(
        GOOD_VALIDATION, GOOD_TEST, 7.0, config, _compared(5.0, 0.03)
    ).passed


def test_incomplete_comparison_blocks_release() -> None:
    config = load_config("configs/development.yaml").gates
    champion = ChampionComparison(status="compared", champion_test_mae=5.0)
    assert not evaluate_release_gates(GOOD_VALIDATION, GOOD_TEST, 7.0, config, champion).passed


def test_zero_margin_is_rejected_by_config() -> None:
    from pydantic import ValidationError

    from taxi_duration.config import GateConfig

    payload = load_config("configs/development.yaml").gates.model_dump()
    payload["max_regression_vs_champion"] = 0.0
    with pytest.raises(ValidationError):
        GateConfig.model_validate(payload)


def test_equally_good_candidates_mostly_pass() -> None:
    """Regression test for the v1.2 superiority behaviour (~95% of equal candidates blocked)."""
    config = load_config("configs/development.yaml").gates
    rng = np.random.default_rng(0)
    champion_mae, n_rows, trials = 7.0, 20_000, 200
    clusters = np.arange(n_rows) // 60  # one-hour blocks of one-minute rows
    passed = superior = 0
    for _ in range(trials):
        hour_effect = np.repeat(rng.normal(0.0, 0.3, n_rows // 60 + 1), 60)[:n_rows]
        delta = hour_effect + rng.normal(0.0, 3.0, n_rows)  # true mean difference = 0
        stat = paired_delta_upper_bound(delta, clusters, config.champion_confidence_level)
        champion = _compared(champion_mae, stat.upper_confidence)
        passed += evaluate_release_gates(GOOD_VALIDATION, GOOD_TEST, 7.0, config, champion).passed
        superior += stat.upper_confidence <= 0.0  # what a zero-margin gate would require
    # Expected pass rate here is ~0.83 (margin 0.07 min vs cluster-robust SE ~0.027 min); a
    # zero-margin (superiority) gate would pass only ~5%. The pass rate rises with test volume.
    assert passed / trials > 0.7
    assert superior / trials < 0.15


def test_clearly_worse_candidates_are_blocked() -> None:
    config = load_config("configs/development.yaml").gates
    rng = np.random.default_rng(1)
    n_rows = 20_000
    clusters = np.arange(n_rows) // 60
    delta = 0.5 + rng.normal(0.0, 3.0, n_rows)  # candidate 0.5 min worse per trip
    stat = paired_delta_upper_bound(delta, clusters, config.champion_confidence_level)
    champion = _compared(7.0, stat.upper_confidence)
    assert not evaluate_release_gates(GOOD_VALIDATION, GOOD_TEST, 7.0, config, champion).passed


def test_cluster_robust_error_exceeds_iid_error_under_hour_correlation() -> None:
    rng = np.random.default_rng(2)
    n_rows = 12_000
    clusters = np.arange(n_rows) // 60
    delta = np.repeat(rng.normal(0.0, 1.0, n_rows // 60), 60) + rng.normal(0.0, 1.0, n_rows)
    clustered = paired_delta_upper_bound(delta, clusters, 0.95).standard_error
    iid = paired_delta_upper_bound(delta, np.arange(n_rows), 0.95).standard_error
    assert clustered > 3 * iid


def test_statistic_requires_two_clusters() -> None:
    with pytest.raises(ValueError, match="two clusters"):
        paired_delta_upper_bound(np.array([0.1, -0.2, 0.3]), np.zeros(3), 0.95)


def test_unevaluable_champion_blocks_release() -> None:
    config = load_config("configs/development.yaml").gates
    champion = ChampionComparison(status="failed", detail="incompatible runtime")
    assert not evaluate_release_gates(GOOD_VALIDATION, GOOD_TEST, 7.0, config, champion).passed


def test_missing_champion_is_bootstrap_not_failure() -> None:
    config = load_config("configs/development.yaml").gates
    champion = ChampionComparison(status="skipped")
    assert evaluate_release_gates(GOOD_VALIDATION, GOOD_TEST, 7.0, config, champion).passed


def test_data_quality_failure_blocks_release() -> None:
    config = load_config("configs/development.yaml").gates
    quality = {
        "rejection_rate": 0.20,
        "out_of_period_rate": 0.0,
        "duplicate_rate": 0.0,
        "valid_rows": 1_000,
    }
    result = evaluate_release_gates(GOOD_VALIDATION, GOOD_TEST, 7.0, config, data_quality=quality)
    assert not result.passed
    assert "data rejection rate exceeds limit" in result.failures
