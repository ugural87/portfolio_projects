from __future__ import annotations

import math
from dataclasses import dataclass

VALIDATION_LOSS_MIN_DELTA = 1e-5


@dataclass
class ValidationLossCheckpoint:
    """Track validation-loss checkpoint selection and early-stopping state."""

    patience: int
    min_delta: float = VALIDATION_LOSS_MIN_DELTA
    best_loss: float = float("inf")
    best_epoch: int | None = None
    stale_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("Early-stopping patience must be positive.")
        if self.min_delta < 0:
            raise ValueError("Early-stopping min_delta must be non-negative.")

    def observe(self, epoch: int, validation_loss: float) -> bool:
        if epoch < 0:
            raise ValueError("Epoch must be non-negative.")
        if not math.isfinite(validation_loss):
            raise ValueError("Validation loss must be finite.")
        improved = validation_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = float(validation_loss)
            self.best_epoch = int(epoch)
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.stale_epochs >= self.patience
