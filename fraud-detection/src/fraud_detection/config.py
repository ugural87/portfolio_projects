from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.55
    validation: float = 0.10
    calibration: float = 0.10
    policy: float = 0.10
    test: float = 0.15

    def validate(self) -> None:
        total = self.train + self.validation + self.calibration + self.policy + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")


@dataclass(frozen=True)
class CostConfig:
    review_cost: float = 4.0
    legitimate_friction_cost: float = 8.0
    chargeback_fee: float = 15.0
    capture_rate: float = 0.80
    daily_alert_capacity: int = 500
    currency_label: str = "CU"


@dataclass(frozen=True)
class RunConfig:
    random_state: int = 42
    fast_mode: bool = True
    drop_exact_duplicates: bool = True
    bootstrap_repetitions: int = 500
    causal_bootstrap_repetitions: int = 30
    deep_epochs: int = 18
    deep_patience: int = 4


@dataclass(frozen=True)
class ProjectConfig:
    split: SplitConfig = SplitConfig()
    cost: CostConfig = CostConfig()
    run: RunConfig = RunConfig()

    @classmethod
    def from_json(cls, path: str | Path) -> "ProjectConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(
            split=SplitConfig(**raw.get("split", {})),
            cost=CostConfig(**raw.get("cost", {})),
            run=RunConfig(**raw.get("run", {})),
        )
        obj.split.validate()
        return obj

    def to_dict(self) -> dict:
        return asdict(self)

