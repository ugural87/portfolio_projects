from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_prediction_plot(predictions: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.plot(predictions.meeting_date, predictions.truth_bp, label="Realised", color="black")
    for name, color in (
        ("fusion_bp", "#0072B2"),
        ("price_only_bp", "#D55E00"),
        ("rate_only_bp", "#009E73"),
        ("shuffled_text_bp", "#CC79A7"),
    ):
        axis.plot(
            predictions.meeting_date,
            predictions[name],
            label=name.removesuffix("_bp"),
            alpha=0.8,
            color=color,
        )
    axis.axhline(0.0, color="grey", linewidth=0.8)
    axis.set(
        ylabel="Five-business-day DGS10 change (bp)",
        title="Walk-forward FOMC forecasts",
    )
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_error_plot(predictions: pd.DataFrame, output: Path) -> None:
    columns = ["fusion_bp", "price_only_bp", "rate_only_bp", "shuffled_text_bp"]
    errors = pd.DataFrame(
        {
            column.removesuffix("_bp"): (predictions.truth_bp - predictions[column]).abs()
            for column in columns
        }
    )
    long = errors.melt(var_name="model", value_name="absolute_error_bp")
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=long, x="model", y="absolute_error_bp", ax=axis, showfliers=False)
    axis.set(title="Out-of-sample absolute-error distribution", xlabel="")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_interval_plot(predictions: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.fill_between(
        predictions.meeting_date,
        predictions.fusion_lower_bp,
        predictions.fusion_upper_bp,
        color="#56B4E9",
        alpha=0.25,
        label="Conformal interval",
    )
    axis.plot(
        predictions.meeting_date,
        predictions.fusion_bp,
        color="#0072B2",
        label="Fusion median",
    )
    axis.scatter(
        predictions.meeting_date,
        predictions.truth_bp,
        color="black",
        s=12,
        label="Realised",
    )
    axis.set(ylabel="Basis points", title="Fold-calibrated prediction intervals")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_attention_plot(attention: pd.DataFrame, output: Path) -> None:
    pivot = attention.pivot_table(
        index="event_token", columns="meeting_date", values="attention", aggfunc="mean"
    )
    figure, axis = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, cmap="viridis", ax=axis, cbar_kws={"label": "Mean attention"})
    axis.set(
        title="Price summary attention to FOMC event tokens",
        xlabel="Meeting",
        ylabel="Token",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
