from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_real_local_artifacts_and_model_contract() -> None:
    torch = pytest.importorskip("torch")
    panel_path = ROOT / "data" / "raw" / "treasury_yields.csv"
    events_path = ROOT / "data" / "processed" / "grounded_fomc_events.csv"
    checkpoint_path = (
        ROOT / "outputs" / "checkpoints" / "historical_backbone" / "backbone.pt"
    )
    if not all(path.exists() for path in (panel_path, events_path, checkpoint_path)):
        pytest.skip("Real local data/checkpoint not present; no synthetic substitute is used.")

    import pandas as pd
    from torch.utils.data import DataLoader

    from us10y_fomc.config import ProjectPaths, load_project_config
    from us10y_fomc.features.event_dataset import FomcEventDataset
    from us10y_fomc.features.event_scalers import fit_event_scalers
    from us10y_fomc.features.targets import TargetTransform
    from us10y_fomc.llm.feature_contract import FOMC_FEATURE_NAMES
    from us10y_fomc.models.cross_attention_fusion import FomcCrossFusionModel
    from us10y_fomc.training.backbone_training import load_historical_backbone
    from us10y_fomc.validation.model_contracts import (
        audit_frozen_price_encoder,
        audit_fusion_forward,
    )
    from us10y_fomc.workflow import prepare_aligned_events

    config = load_project_config(ROOT)
    paths = ProjectPaths(ROOT)
    _, bundle, aligned, splits, _ = prepare_aligned_events(paths, config)
    backbone, _ = load_historical_backbone(
        checkpoint_path, config.price, pd.Timestamp(aligned.meeting_date.min())
    )
    scalers = fit_event_scalers(aligned, splits["train"])
    dataset = FomcEventDataset(
        aligned,
        splits["train"][: min(4, len(splits["train"]))],
        bundle,
        TargetTransform(bundle, config.price),
        scalers,
        config.fusion,
    )
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    model = FomcCrossFusionModel(
        backbone, config.price, config.fusion, len(FOMC_FEATURE_NAMES)
    )
    assert audit_frozen_price_encoder(model)["passed"]
    assert audit_fusion_forward(
        model, batch, len(FOMC_FEATURE_NAMES), config.fusion.n_attention_heads
    )["passed"]
