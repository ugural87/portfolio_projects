import pandas as pd
import pytest

from us10y_fomc.validation.leakage import audit_backbone_cutoff


def test_backbone_cutoff_accepts_strictly_earlier_labels() -> None:
    result = audit_backbone_cutoff(
        {"last_training_label_date": "1993-01-29"}, pd.Timestamp("1993-02-03")
    )
    assert result["passed"]


def test_backbone_cutoff_rejects_same_day_or_future_labels() -> None:
    with pytest.raises(AssertionError, match="Backbone leakage"):
        audit_backbone_cutoff(
            {"last_training_label_date": "1993-02-03"}, pd.Timestamp("1993-02-03")
        )
