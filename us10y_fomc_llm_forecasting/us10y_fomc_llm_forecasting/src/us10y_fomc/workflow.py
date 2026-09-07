from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from .config import ProjectConfig, ProjectPaths
from .data.fomc_discovery import discover_fomc_minutes
from .data.fomc_download import download_minutes
from .data.fomc_integrity import audit_fomc_documents
from .data.policy_rates import build_policy_rate_audit
from .data.treasury_data import fetch_treasury_panel, load_treasury_panel, save_treasury_panel
from .evaluation.attention_export import export_attention_tables
from .evaluation.metrics import prediction_frame
from .evaluation.statistical_tests import diebold_mariano
from .features.event_alignment import align_fomc_events, make_event_splits
from .features.price_features import build_price_feature_bundle
from .features.targets import TargetTransform
from .llm.extraction import (
    ExtractionPaths,
    build_grounded_event_table,
    extract_minutes_pairs,
)
from .llm.feature_contract import FOMC_FEATURE_NAMES
from .models.ablation_models import PriceOnlyEventModel
from .models.cross_attention_fusion import FomcCrossFusionModel
from .progress import progress
from .training.backbone_training import load_historical_backbone, train_historical_backbone
from .training.common import seed_everything
from .training.conformal import conformal_correction
from .training.fusion_training import (
    build_event_loaders,
    collect_event_predictions,
    save_event_checkpoint,
    train_event_model,
)
from .training.selection import VALIDATION_LOSS_MIN_DELTA
from .training.walk_forward import build_walk_forward_folds
from .validation.leakage import audit_backbone_cutoff, audit_event_splits
from .validation.model_contracts import audit_frozen_price_encoder, audit_fusion_forward


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def download_official_data(
    paths: ProjectPaths,
    config: ProjectConfig,
    fred_key: str,
) -> dict[str, object]:
    paths.ensure()
    panel, treasury_audit = fetch_treasury_panel(fred_key, config.price.start_date)
    save_treasury_panel(panel, treasury_audit, paths.data / "raw")
    manifest, excluded = discover_fomc_minutes(config.extraction.start_year, date.today().year)
    documents = download_minutes(manifest, paths.minutes_text)
    coverage, checks = audit_fomc_documents(
        manifest, excluded, documents, config.extraction.start_year
    )
    manifest.to_csv(paths.manifests / "fomc_manifest.csv", index=False)
    excluded.to_csv(paths.manifests / "fomc_excluded.csv", index=False)
    documents.to_csv(paths.manifests / "fomc_documents.csv", index=False)
    coverage.to_csv(paths.audit_reports / "fomc_year_coverage.csv")
    release_gap = (
        pd.to_datetime(documents.minutes_release_date) - pd.to_datetime(documents.meeting_date)
    ).dt.days
    release_regime = pd.Series(
        np.where(documents.meeting_date.dt.year >= 2005, "2005_onwards", "pre_2005"),
        index=documents.index,
        name="regime",
    )
    (
        pd.DataFrame({"release_gap_days": release_gap, "regime": release_regime})
        .groupby("regime")
        .release_gap_days.agg(["count", "mean", "min", "max"])
        .to_csv(paths.audit_reports / "minutes_release_gap_summary.csv")
    )
    rates, disagreements = build_policy_rate_audit(documents, fred_key)
    rates.to_csv(paths.data / "processed" / "fomc_policy_rates.csv", index=False)
    disagreements.to_csv(paths.audit_reports / "minutes_rate_parser_disagreements.csv", index=False)
    _write_json(
        paths.audit_reports / "policy_rate_audit_summary.json",
        {
            "fred_rate_gate": True,
            "minutes_parser_coverage": float(rates.minutes_target_midpoint.notna().mean()),
            "minutes_fred_disagreements": len(disagreements),
            "minutes_parser_role": "diagnostic_only",
        },
    )
    _write_json(paths.audit_reports / "document_integrity.json", checks)
    return {
        "treasury_rows": len(panel),
        "documents": len(documents),
        "excluded": len(excluded),
        "rate_parser_disagreements": len(disagreements),
    }


def load_official_documents(paths: ProjectPaths) -> pd.DataFrame:
    documents = pd.read_csv(
        paths.manifests / "fomc_documents.csv",
        parse_dates=["meeting_date", "minutes_release_date"],
    )

    def resolve_text_path(value: str) -> str:
        recorded = Path(str(value))
        if recorded.is_file():
            return str(recorded)
        portable = paths.minutes_text / recorded.name
        if portable.is_file():
            return str(portable)
        raise FileNotFoundError(f"FOMC minutes text is missing for manifest path {recorded.name}.")

    documents["text_path"] = documents["text_path"].map(resolve_text_path)
    return documents.sort_values("meeting_date").reset_index(drop=True)


def audit_local_official_data(paths: ProjectPaths, config: ProjectConfig) -> dict[str, object]:
    panel = load_treasury_panel(paths.data / "raw")
    documents = load_official_documents(paths)
    manifest = pd.read_csv(
        paths.manifests / "fomc_manifest.csv",
        parse_dates=["meeting_date", "meeting_start_date", "canonical_minutes_date"],
    )
    excluded_path = paths.manifests / "fomc_excluded.csv"
    excluded = pd.read_csv(excluded_path, parse_dates=["meeting_date"])
    _, document_checks = audit_fomc_documents(
        manifest, excluded, documents, config.extraction.start_year
    )
    rates = pd.read_csv(
        paths.data / "processed" / "fomc_policy_rates.csv",
        parse_dates=["meeting_date", "rate_effective_date"],
    )
    anchors = {
        pd.Timestamp("2008-12-16"): -87.5,
        pd.Timestamp("2020-04-29"): 0.0,
        pd.Timestamp("2022-06-15"): 75.0,
    }
    lookup = rates.set_index("meeting_date").actual_rate_change_bp
    rate_check = all(
        date in lookup.index and np.isclose(float(lookup.loc[date]), expected)
        for date, expected in anchors.items()
    )
    checks = {
        "treasury_panel": bool(len(panel) and panel.index.is_unique),
        "document_integrity": all(document_checks.values()),
        "fred_rate_anchors": rate_check,
    }
    if not all(checks.values()):
        raise AssertionError(f"Local official-data audit failed: {checks}")
    return checks


def run_minutes_extraction(
    paths: ProjectPaths,
    config: ProjectConfig,
    openai_key: str | None,
    pilot_paths: tuple[Path, ...] = (),
):
    audit_local_official_data(paths, config)
    documents = load_official_documents(paths)
    rates = pd.read_csv(
        paths.data / "processed" / "fomc_policy_rates.csv",
        parse_dates=["meeting_date", "rate_effective_date"],
    )
    result = extract_minutes_pairs(
        documents,
        openai_key,
        config.extraction,
        config.runtime,
        ExtractionPaths(
            production_cache=paths.extraction_cache / "luna_production.jsonl",
            quarantine=paths.quarantine / "luna_quarantine.jsonl",
            cost_ledger=paths.audit_reports / "luna_cost_ledger.jsonl",
            token_budget=paths.audit_reports / "luna_token_budget.csv",
        ),
        pilot_paths=pilot_paths,
    )
    if result.ready:
        events = build_grounded_event_table(result, rates)
        events.to_csv(paths.data / "processed" / "grounded_fomc_events.csv", index=False)
    _write_json(
        paths.audit_reports / "extraction_status.json",
        {
            "required_pairs": result.required_pairs,
            "accepted_pairs": len(result.cached_records),
            "missing_pairs": list(result.missing_dates),
            "ready": result.ready,
            "spent_usd": result.spent_usd,
        },
    )
    return result


def prepare_aligned_events(paths: ProjectPaths, config: ProjectConfig):
    panel = load_treasury_panel(paths.data / "raw")
    events = pd.read_csv(
        paths.data / "processed" / "grounded_fomc_events.csv",
        parse_dates=["meeting_date", "previous_meeting_date"],
    )
    bundle = build_price_feature_bundle(panel, config.price)
    aligned = align_fomc_events(events, panel.index, config.fusion)
    aligned.to_pickle(paths.data / "processed" / "aligned_fomc_events.pkl")
    splits, holdout_cutoff = make_event_splits(aligned, config.fusion)
    audit_event_splits(aligned, splits)
    return panel, bundle, aligned, splits, holdout_cutoff


def ensure_historical_backbone(
    paths: ProjectPaths,
    config: ProjectConfig,
    bundle,
    first_fomc_meeting_date,
):
    checkpoint = paths.outputs / "checkpoints" / "historical_backbone" / "backbone.pt"
    if not checkpoint.exists() and config.runtime.train_historical_backbone:
        metadata = train_historical_backbone(
            bundle,
            first_fomc_meeting_date,
            config.price,
            checkpoint,
            paths.outputs / "runtime" / "historical_backbone.jsonl",
        )
    elif not checkpoint.exists():
        raise RuntimeError(
            "Historical backbone checkpoint is missing. Set train_historical_backbone=true."
        )
    backbone, metadata = load_historical_backbone(checkpoint, config.price, first_fomc_meeting_date)
    audit = audit_backbone_cutoff(metadata, first_fomc_meeting_date)
    _write_json(paths.audit_reports / "backbone_leakage_audit.json", audit)
    return backbone, metadata


def _prediction_rows(
    aligned: pd.DataFrame,
    test_indices: np.ndarray,
    fold: int,
    predictions: dict[str, dict[str, np.ndarray]],
    qhat: float,
) -> pd.DataFrame:
    base = aligned.iloc[test_indices].reset_index(drop=True)
    rows = pd.DataFrame(
        {
            "fold": fold,
            "meeting_date": base.meeting_date,
            "target_date": base.target_date,
        }
    )
    for name, result in predictions.items():
        frame = prediction_frame(result)
        if name == "fusion":
            rows["truth_bp"] = frame["truth_bp"]
            rows["true_class"] = frame["true_class"]
            rows["fusion_lower_bp"] = (result["quantiles"][:, 0] - qhat) * result["scales"]
            rows["fusion_upper_bp"] = (result["quantiles"][:, 2] + qhat) * result["scales"]
            rows["fold_qhat"] = qhat
        rows[f"{name}_bp"] = frame["forecast_bp"]
        rows[f"{name}_class"] = frame["predicted_class"]
    return rows


def run_walk_forward_study(
    paths: ProjectPaths,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.runtime.run_walk_forward or not config.runtime.train_fusion_models:
        raise RuntimeError("Enable both train_fusion_models and run_walk_forward explicitly.")
    _, bundle, aligned, _, holdout_cutoff = prepare_aligned_events(paths, config)
    first_meeting = pd.Timestamp(aligned.meeting_date.min())
    backbone, backbone_metadata = ensure_historical_backbone(paths, config, bundle, first_meeting)
    target = TargetTransform(bundle, config.price)
    folds = build_walk_forward_folds(aligned, holdout_cutoff, config.fusion)
    folds.drop(columns=[c for c in folds.columns if c.endswith("_indices")]).to_csv(
        paths.outputs / "tables" / "walk_forward_manifest.csv", index=False
    )
    development_count = int((aligned.meeting_date < holdout_cutoff).sum())
    projected_coverage = int(folds.n_test.sum()) / development_count
    if projected_coverage <= 0.60:
        raise AssertionError(
            f"Projected walk-forward OOS coverage is only {projected_coverage:.1%}."
        )

    walk_forward_log = paths.outputs / "runtime" / "walk_forward.jsonl"
    if walk_forward_log.exists():
        walk_forward_log.unlink()

    pooled = []
    fold_metrics = []
    fold_progress = progress(
        folds.itertuples(index=False),
        desc="Walk-forward folds",
        total=len(folds),
        unit="fold",
    )
    for fold_row in fold_progress:
        fold = int(fold_row.fold)
        fold_progress.set_postfix(
            fold=f"{fold + 1}/{len(folds)}",
            years=f"{fold_row.test_start_year}-{fold_row.test_stop_year}",
        )
        split_indices = {
            "train": fold_row.train_indices,
            "validation": fold_row.validation_indices,
            "calibration": fold_row.calibration_indices,
            "test": fold_row.test_indices,
        }
        loaders, scalers = build_event_loaders(
            aligned, split_indices, bundle, target, config.fusion
        )
        shuffled_loaders, _ = build_event_loaders(
            aligned,
            split_indices,
            bundle,
            target,
            config.fusion,
            scalers=scalers,
            shuffle_semantics=True,
            seed_offset=10_000 * (fold + 1),
        )
        fold_seed = config.fusion.random_seed + fold
        seed_everything(fold_seed)
        full_template = FomcCrossFusionModel(
            backbone, config.price, config.fusion, len(FOMC_FEATURE_NAMES)
        )
        models = {
            "price_only": PriceOnlyEventModel(backbone, config.price, config.fusion),
            "rate_only": FomcCrossFusionModel(
                backbone,
                config.price,
                config.fusion,
                len(FOMC_FEATURE_NAMES),
                event_mode="rate_only",
            ),
            "shuffled_text": copy.deepcopy(full_template),
            "fusion": copy.deepcopy(full_template),
        }
        real_contract_batch = next(iter(loaders["train"]))
        audit_frozen_price_encoder(models["fusion"])
        audit_fusion_forward(
            models["fusion"],
            real_contract_batch,
            len(FOMC_FEATURE_NAMES),
            config.fusion.n_attention_heads,
        )
        if any(
            parameter.requires_grad
            for parameter in models["rate_only"].feature_encoder.parameters()
        ):
            raise AssertionError("Rate-only model did not disable its text encoder.")
        predictions = {}
        model_progress = progress(
            models.items(),
            desc=f"Fold {fold + 1}/{len(folds)} models",
            total=len(models),
            unit="model",
            leave=False,
        )
        for name, model in model_progress:
            model_progress.set_postfix(model=name)
            active_loaders = shuffled_loaders if name == "shuffled_text" else loaders
            trained, history = train_event_model(
                model,
                active_loaders,
                config.price,
                config.fusion,
                seed=fold_seed,
                runtime_log=walk_forward_log,
                stage=f"fold_{fold}_{name}",
            )
            history.to_csv(
                paths.outputs / "metrics" / f"fold_{fold}_{name}_training.csv", index=False
            )
            selected_history = history.loc[history.selected_checkpoint]
            if len(selected_history) != 1:
                raise AssertionError("Training history must identify exactly one checkpoint epoch.")
            selected_epoch = selected_history.iloc[0]
            predictions[name] = collect_event_predictions(trained, active_loaders["test"])
            save_event_checkpoint(
                paths.outputs / "checkpoints" / "fusion" / f"fold_{fold}_{name}.pt",
                trained,
                name,
                config.price,
                config.fusion,
                scalers,
                {
                    "fold": fold,
                    "backbone_last_training_label_date": backbone_metadata[
                        "last_training_label_date"
                    ],
                    "test_start_year": fold_row.test_start_year,
                    "test_stop_year": fold_row.test_stop_year,
                    "checkpoint_selection_metric": "validation_joint_loss",
                    "checkpoint_min_delta": VALIDATION_LOSS_MIN_DELTA,
                    "early_stopping_patience": config.fusion.patience,
                    "epochs_completed": len(history),
                    "selected_epoch": int(selected_epoch.epoch),
                    "selected_validation_loss": float(selected_epoch.validation),
                    "selected_validation_balanced_accuracy": float(
                        selected_epoch.validation_balanced_accuracy
                    ),
                },
            )
            if name == "fusion":
                calibration_prediction = collect_event_predictions(trained, loaders["calibration"])
                qhat = conformal_correction(
                    calibration_prediction["labels"],
                    calibration_prediction["quantiles"][:, 0],
                    calibration_prediction["quantiles"][:, 2],
                    config.price.conformal_alpha,
                )
                export_attention_tables(
                    predictions[name],
                    aligned.iloc[fold_row.test_indices].reset_index(drop=True),
                    paths.outputs / "attention",
                    f"fold_{fold}",
                )
        fusion_truth = predictions["fusion"]["labels"] * predictions["fusion"]["scales"]
        for control in ("price_only", "rate_only", "shuffled_text"):
            control_truth = predictions[control]["labels"] * predictions[control]["scales"]
            if not np.allclose(control_truth, fusion_truth):
                raise AssertionError(f"{control} changed the OOS target series.")
        fold_frame = _prediction_rows(aligned, fold_row.test_indices, fold, predictions, qhat)
        pooled.append(fold_frame)
        if float(np.mean(np.abs(fold_frame.fusion_bp - fold_frame.price_only_bp))) <= 1e-6:
            raise AssertionError("Fusion and price-only forecasts are mechanically identical.")
        row = {
            "fold": fold,
            "n_test": len(fold_frame),
            "fold_qhat": qhat,
            "interval_coverage": float(
                np.mean(
                    (fold_frame.truth_bp >= fold_frame.fusion_lower_bp)
                    & (fold_frame.truth_bp <= fold_frame.fusion_upper_bp)
                )
            ),
        }
        for name in models:
            row[f"{name}_mae_bp"] = float(
                np.mean(np.abs(fold_frame.truth_bp - fold_frame[f"{name}_bp"]))
            )
            row[f"{name}_balanced_accuracy"] = float(
                balanced_accuracy_score(fold_frame.true_class, fold_frame[f"{name}_class"])
            )
        fold_metrics.append(row)

    pooled_frame = pd.concat(pooled, ignore_index=True).sort_values("meeting_date")
    if pooled_frame.meeting_date.duplicated().any():
        raise AssertionError("Pooled walk-forward predictions contain duplicate meetings.")
    if len(pooled_frame) / development_count <= 0.60:
        raise AssertionError("Realised walk-forward coverage fell below 60%.")
    metrics = pd.DataFrame(fold_metrics)
    for control in ("price_only", "rate_only", "shuffled_text"):
        comparison = diebold_mariano(
            pooled_frame.truth_bp.to_numpy(),
            pooled_frame.fusion_bp.to_numpy(),
            pooled_frame[f"{control}_bp"].to_numpy(),
            hac_lag=0,
        )
        for key, value in comparison.items():
            metrics.loc[:, f"pooled_dm_{control}_{key}"] = value
    pooled_frame.to_csv(paths.outputs / "predictions" / "walk_forward_predictions.csv", index=False)
    metrics.to_csv(paths.outputs / "metrics" / "walk_forward_metrics.csv", index=False)
    return pooled_frame, metrics
