from __future__ import annotations

import pytest

from us10y_fomc.training.selection import ValidationLossCheckpoint


def test_validation_loss_controls_checkpoint_and_early_stopping() -> None:
    tracker = ValidationLossCheckpoint(patience=3, min_delta=1e-5)

    assert tracker.observe(0, 1.20)
    assert tracker.observe(1, 1.10)
    assert not tracker.observe(2, 1.15)
    assert not tracker.observe(3, 1.12)
    assert not tracker.should_stop
    assert not tracker.observe(4, 1.11)

    assert tracker.should_stop
    assert tracker.best_epoch == 1
    assert tracker.best_loss == pytest.approx(1.10)


def test_balanced_accuracy_cannot_change_loss_checkpoint() -> None:
    tracker = ValidationLossCheckpoint(patience=2)
    balanced_accuracy = [0.40, 0.95, 0.30]
    validation_loss = [1.00, 1.50, 0.90]

    improved = [tracker.observe(epoch, loss) for epoch, loss in enumerate(validation_loss)]

    assert balanced_accuracy[1] == max(balanced_accuracy)
    assert improved == [True, False, True]
    assert tracker.best_epoch == 2
    assert tracker.best_loss == pytest.approx(0.90)


def test_validation_loss_checkpoint_rejects_invalid_state() -> None:
    with pytest.raises(ValueError):
        ValidationLossCheckpoint(patience=0)
    tracker = ValidationLossCheckpoint(patience=2)
    with pytest.raises(ValueError):
        tracker.observe(-1, 1.0)
    with pytest.raises(ValueError):
        tracker.observe(0, float("nan"))
