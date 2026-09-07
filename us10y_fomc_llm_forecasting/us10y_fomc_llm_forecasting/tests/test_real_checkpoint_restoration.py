from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_real_event_model_restores_the_validation_loss_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    required = (
        ROOT / "data" / "raw" / "treasury_yields.csv",
        ROOT / "data" / "processed" / "grounded_fomc_events.csv",
        ROOT / "outputs" / "checkpoints" / "historical_backbone" / "backbone.pt",
    )
    if not all(path.exists() for path in required):
        pytest.skip("Real local data/checkpoint not present; no synthetic substitute is used.")

    from us10y_fomc.config import ProjectPaths, load_project_config
    from us10y_fomc.features.targets import TargetTransform
    from us10y_fomc.models.ablation_models import PriceOnlyEventModel
    from us10y_fomc.training.backbone_training import load_historical_backbone
    from us10y_fomc.training.fusion_training import build_event_loaders, train_event_model
    from us10y_fomc.training.walk_forward import build_walk_forward_folds
    from us10y_fomc.workflow import prepare_aligned_events

    config = load_project_config(ROOT)
    paths = ProjectPaths(ROOT)
    _, bundle, aligned, _, holdout_cutoff = prepare_aligned_events(paths, config)
    first_meeting = pd.Timestamp(aligned.meeting_date.min())
    backbone, _ = load_historical_backbone(required[2], config.price, first_meeting)
    folds = build_walk_forward_folds(aligned, holdout_cutoff, config.fusion)
    fold = folds.iloc[0]
    split_indices = {
        name: fold[f"{name}_indices"] for name in ("train", "validation", "calibration", "test")
    }
    smoke_config = replace(config.fusion, max_epochs=3, patience=4)
    loaders, _ = build_event_loaders(
        aligned,
        split_indices,
        bundle,
        TargetTransform(bundle, config.price),
        smoke_config,
    )
    model = PriceOnlyEventModel(backbone, config.price, smoke_config)
    trained, history = train_event_model(
        model,
        loaders,
        config.price,
        smoke_config,
        seed=smoke_config.random_seed,
        runtime_log=tmp_path / "restoration.jsonl",
        stage="real_checkpoint_restoration",
    )

    assert isinstance(trained, torch.nn.Module)
    assert int(history.selected_checkpoint.sum()) == 1
    selected = history.loc[history.selected_checkpoint].iloc[0]
    assert np.isclose(
        selected.validation,
        history.restored_validation_loss.iloc[-1],
        rtol=1e-5,
        atol=1e-6,
    )
