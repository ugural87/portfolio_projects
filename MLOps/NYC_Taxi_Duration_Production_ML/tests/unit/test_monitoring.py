import pytest

from taxi_duration.monitoring.profile import population_stability_index


def test_psi_is_zero_for_identical_distributions() -> None:
    assert population_stability_index([0.2, 0.8], [0.2, 0.8]) == pytest.approx(0.0)


def test_psi_detects_shift() -> None:
    assert population_stability_index([0.2, 0.8], [0.8, 0.2]) > 1.0
