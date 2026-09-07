from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score

from ..config import ProjectPaths
from ..evaluation.statistical_tests import (
    diebold_mariano,
    fold_block_bootstrap_mae_difference,
)
from .plots import (
    save_attention_plot,
    save_error_plot,
    save_interval_plot,
    save_prediction_plot,
)


def build_postprocess_report(paths: ProjectPaths) -> dict[str, object]:
    prediction_path = paths.outputs / "predictions" / "walk_forward_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError("Walk-forward predictions do not exist. Run training first.")
    predictions = pd.read_csv(
        prediction_path, parse_dates=["meeting_date", "target_date"]
    )
    figure_dir = paths.outputs / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_prediction_plot(predictions, figure_dir / "walk_forward_forecasts.png")
    save_error_plot(predictions, figure_dir / "absolute_errors.png")
    save_interval_plot(predictions, figure_dir / "conformal_intervals.png")

    attention_files = sorted(paths.outputs.glob("attention/fold_*_price_to_event.csv"))
    if attention_files:
        attention = pd.concat(
            [pd.read_csv(path, parse_dates=["meeting_date"]) for path in attention_files],
            ignore_index=True,
        )
        save_attention_plot(attention, figure_dir / "event_token_attention.png")
        (
            attention.groupby("event_token", as_index=False)
            .attention.mean()
            .sort_values("attention", ascending=False)
            .to_csv(paths.outputs / "tables" / "mean_event_attention.csv", index=False)
        )

    coverage = float(
        np.mean(
            (predictions.truth_bp >= predictions.fusion_lower_bp)
            & (predictions.truth_bp <= predictions.fusion_upper_bp)
        )
    )
    summary: dict[str, object] = {
        "n_oos": len(predictions),
        "pooled_interval_coverage": coverage,
        "pooled_mean_interval_width_bp": float(
            np.mean(predictions.fusion_upper_bp - predictions.fusion_lower_bp)
        ),
        "models": {},
        "paired_tests": {},
    }
    covered_count = int(
        (
            (predictions.truth_bp >= predictions.fusion_lower_bp)
            & (predictions.truth_bp <= predictions.fusion_upper_bp)
        ).sum()
    )
    coverage_interval = stats.binomtest(
        covered_count, n=len(predictions)
    ).proportion_ci(confidence_level=0.95, method="exact")
    summary["coverage_ci_95"] = [
        float(coverage_interval.low),
        float(coverage_interval.high),
    ]
    for name in ("fusion", "price_only", "rate_only", "shuffled_text"):
        error = predictions.truth_bp - predictions[f"{name}_bp"]
        summary["models"][name] = {
            "mae_bp": float(error.abs().mean()),
            "rmse_bp": float(np.sqrt(np.mean(np.square(error)))),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    predictions.true_class, predictions[f"{name}_class"]
                )
            ),
            "macro_f1": float(
                f1_score(
                    predictions.true_class,
                    predictions[f"{name}_class"],
                    average="macro",
                    zero_division=0,
                )
            ),
        }
    for control in ("price_only", "rate_only", "shuffled_text"):
        summary["paired_tests"][f"fusion_vs_{control}"] = diebold_mariano(
            predictions.truth_bp.to_numpy(),
            predictions.fusion_bp.to_numpy(),
            predictions[f"{control}_bp"].to_numpy(),
            hac_lag=0,
        )
        summary["paired_tests"][f"bootstrap_fusion_vs_{control}"] = (
            fold_block_bootstrap_mae_difference(
                predictions, "fusion_bp", f"{control}_bp"
            )
        )
    report_path = paths.outputs / "reports" / "postprocess_summary.json"
    event_path = paths.data / "processed" / "grounded_fomc_events.csv"
    if event_path.exists():
        events = pd.read_csv(event_path)
        availability_columns = [
            column for column in events.columns if column.endswith("__available")
        ]
        availability = (
            events[availability_columns]
            .mean()
            .rename_axis("feature")
            .reset_index(name="availability_rate")
        )
        availability["feature"] = availability.feature.str.removesuffix("__available")
        availability.to_csv(
            paths.outputs / "tables" / "feature_availability.csv", index=False
        )
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
