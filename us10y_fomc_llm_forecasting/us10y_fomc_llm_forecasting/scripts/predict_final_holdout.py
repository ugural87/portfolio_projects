"""Score the sealed final-year holdout with the already-trained fold-7 models.

This does NOT retrain anything and does NOT modify any existing output. It loads
the last walk-forward checkpoints, applies them to the meetings that were kept
outside the reported evaluation, and writes a separate predictions file.

    python scripts/predict_final_holdout.py

Output: outputs/predictions/final_holdout_predictions.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from _bootstrap import PROJECT_ROOT

from us10y_fomc.config import ProjectPaths, load_project_config
from us10y_fomc.features.event_scalers import EventScalers
from us10y_fomc.features.price_features import build_price_feature_bundle
from us10y_fomc.features.event_alignment import align_fomc_events
from us10y_fomc.features.targets import TargetTransform
from us10y_fomc.data.treasury_data import load_treasury_panel
from us10y_fomc.llm.feature_contract import FOMC_FEATURE_NAMES
from us10y_fomc.models.ablation_models import PriceOnlyEventModel
from us10y_fomc.models.cross_attention_fusion import FomcCrossFusionModel
from us10y_fomc.training.backbone_training import load_historical_backbone
from us10y_fomc.evaluation.metrics import prediction_frame
from us10y_fomc.training.fusion_training import build_event_loaders, collect_event_predictions

MODEL_NAMES = ("price_only", "rate_only", "shuffled_text", "fusion")


def _scalers_from_checkpoint(checkpoint: dict) -> EventScalers:
    return EventScalers(
        feature_mean=pd.Series(checkpoint["feature_mean"]),
        feature_std=pd.Series(checkpoint["feature_std"]),
        rate_mean=pd.Series(checkpoint["rate_mean"]),
        rate_std=pd.Series(checkpoint["rate_std"]),
    )


def _build_model(name: str, backbone, price_config, fusion_config):
    if name == "price_only":
        return PriceOnlyEventModel(backbone, price_config, fusion_config)
    if name == "rate_only":
        return FomcCrossFusionModel(
            backbone, price_config, fusion_config, len(FOMC_FEATURE_NAMES),
            event_mode="rate_only",
        )
    return FomcCrossFusionModel(
        backbone, price_config, fusion_config, len(FOMC_FEATURE_NAMES)
    )


if __name__ == "__main__":
    config = load_project_config(PROJECT_ROOT)
    paths = ProjectPaths(PROJECT_ROOT).ensure()

    panel = load_treasury_panel(paths.data / "raw")
    events = pd.read_csv(
        paths.data / "processed" / "grounded_fomc_events.csv",
        parse_dates=["meeting_date", "previous_meeting_date"],
    )
    bundle = build_price_feature_bundle(panel, config.price)
    aligned = align_fomc_events(events, panel.index, config.fusion)
    target = TargetTransform(bundle, config.price)

    cutoff = aligned.meeting_date.max() - pd.DateOffset(
        months=config.fusion.final_holdout_months
    )
    holdout = np.flatnonzero(aligned.meeting_date >= cutoff)
    if not len(holdout):
        raise RuntimeError("The final holdout is empty.")

    # the last walk-forward fold supplies the trained weights, the scalers and qhat
    reported = pd.read_csv(
        paths.outputs / "predictions" / "walk_forward_predictions.csv",
        parse_dates=["meeting_date"],
    )
    last_fold = int(reported.fold.max())
    qhat = float(reported.loc[reported.fold == last_fold, "fold_qhat"].iloc[0])
    last_train_target = reported.loc[reported.fold == last_fold, "meeting_date"].max()

    print(f"Holdout cutoff        : {cutoff:%Y-%m-%d}")
    print(f"Holdout meetings      : {len(holdout)} "
          f"({aligned.meeting_date.iloc[holdout].min():%Y-%m-%d} to "
          f"{aligned.meeting_date.iloc[holdout].max():%Y-%m-%d})")
    print(f"Scoring model         : fold {last_fold} checkpoints, qhat = {qhat:.4f}")
    print(f"Last reported meeting : {last_train_target:%Y-%m-%d}")

    checkpoint_dir = paths.outputs / "checkpoints" / "fusion"
    reference = torch.load(checkpoint_dir / f"fold_{last_fold}_fusion.pt",
                           map_location="cpu", weights_only=False)
    scalers = _scalers_from_checkpoint(reference)

    backbone, _ = load_historical_backbone(
        paths.outputs / "checkpoints" / "historical_backbone" / "backbone.pt",
        config.price,
        pd.Timestamp(aligned.meeting_date.min()),
    )

    splits = {"holdout": holdout}
    loaders, _ = build_event_loaders(
        aligned, splits, bundle, target, config.fusion, scalers=scalers
    )
    shuffled_loaders, _ = build_event_loaders(
        aligned, splits, bundle, target, config.fusion, scalers=scalers,
        shuffle_semantics=True, seed_offset=10_000 * (last_fold + 1),
    )

    base = aligned.iloc[holdout].reset_index(drop=True)
    rows = pd.DataFrame({
        "meeting_date": base.meeting_date,
        "target_date": base.target_date,
    })

    for name in MODEL_NAMES:
        checkpoint = torch.load(checkpoint_dir / f"fold_{last_fold}_{name}.pt",
                                map_location="cpu", weights_only=False)
        model = _build_model(name, backbone, config.price, config.fusion)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        loader = shuffled_loaders if name == "shuffled_text" else loaders
        result = collect_event_predictions(model, loader["holdout"])
        frame = prediction_frame(result)

        rows[f"{name}_bp"] = frame["forecast_bp"]
        rows[f"{name}_class"] = frame["predicted_class"]
        if name == "fusion":
            rows["truth_bp"] = frame["truth_bp"]
            rows["true_class"] = frame["true_class"]
            rows["fusion_lower_bp"] = (result["quantiles"][:, 0] - qhat) * result["scales"]
            rows["fusion_upper_bp"] = (result["quantiles"][:, 2] + qhat) * result["scales"]
            rows["fold_qhat"] = qhat

    destination = paths.outputs / "predictions" / "final_holdout_predictions.csv"
    rows.to_csv(destination, index=False)
    print(f"\nWrote {len(rows)} holdout predictions to {destination}")
    print(rows[["meeting_date", "truth_bp", "fusion_bp", "price_only_bp"]]
      .round({"truth_bp": 2, "fusion_bp": 2, "price_only_bp": 2})
      .to_string(index=False))
