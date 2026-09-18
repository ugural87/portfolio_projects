from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from taxi_duration.config import load_config
from taxi_duration.data.ingest import download_month, latest_available_period
from taxi_duration.data.validation import Period
from taxi_duration.logging import configure_logging
from taxi_duration.models.artifact import load_artifact
from taxi_duration.monitoring.drift import drift_report
from taxi_duration.training.pipeline import evaluate_artifact, train_from_parquet

app = typer.Typer(no_args_is_help=True)


def _period(text: str | None) -> Period | None:
    return Period.parse(text) if text else None


def default_period(today: date, lag_months: int = 2) -> Period:
    """Conservative source month: TLC publishes monthly files with a lag of roughly two months."""
    index = today.year * 12 + (today.month - 1) - lag_months
    return Period(index // 12, index % 12 + 1)


def period_window(end: Period, months: int) -> list[Period]:
    if months < 1:
        raise ValueError("months must be at least 1")
    return [end.shift(offset) for offset in range(-(months - 1), 1)]


@app.command("resolve-period")
def resolve_period(
    year: str = "",
    month: str = "",
    lag_months: int = 2,
    config_path: Path = Path("configs/staging.yaml"),
) -> None:
    """Resolve an explicit period or probe backwards to the newest published TLC month."""
    if bool(year.strip()) != bool(month.strip()):
        raise typer.BadParameter("provide both --year and --month, or neither")
    config = load_config(config_path)
    period = (
        Period(int(year), int(month))
        if year.strip()
        else latest_available_period(
            default_period(date.today(), lag_months),
            config.data.source_url_template,
            config.data.source_lookback_months,
        )
    )
    window = period_window(period, config.data.training_window_months)
    typer.echo(
        f"year={period.year:04d}\nmonth={period.month:02d}\nperiod={period}\n"
        f"window={','.join(map(str, window))}"
    )


@app.command()
def download(
    year: int,
    month: int,
    config_path: Path = Path("configs/development.yaml"),
    overwrite: bool = False,
) -> None:
    """Download one immutable monthly TLC parquet file."""
    config = load_config(config_path)
    path, checksum = download_month(
        year, month, config.data.raw_dir, config.data.source_url_template, overwrite=overwrite
    )
    typer.echo(json.dumps({"path": str(path), "sha256": checksum}))


@app.command()
def train(
    data_paths: Annotated[
        list[Path], typer.Argument(help="One or more chronological monthly Parquet files.")
    ],
    config_path: Path = Path("configs/development.yaml"),
    model_path: Path = Path("artifacts/model.joblib"),
    metadata_path: Path = Path("artifacts/metadata.json"),
    metrics_path: Path = Path("artifacts/metrics.json"),
    reference_path: Path = Path("artifacts/reference_profile.json"),
    model_version: str = "local",
    period: Annotated[
        str | None,
        typer.Option(help="Source month YYYY-MM. Defaults to the month in the file name."),
    ] = None,
    champion_model_path: Annotated[
        Path | None, typer.Option(help="Current production model.joblib.")
    ] = None,
    champion_metadata_path: Annotated[
        Path | None, typer.Option(help="Current production metadata.json.")
    ] = None,
) -> None:
    """Train, evaluate, gate (incl. champion comparison) and package a candidate model."""
    configure_logging()
    config = load_config(config_path)
    metrics = train_from_parquet(
        data_paths,
        config,
        config_path,
        model_path,
        metadata_path,
        metrics_path,
        model_version,
        period=_period(period),
        champion_model_path=champion_model_path,
        champion_metadata_path=champion_metadata_path,
        reference_path=reference_path,
    )
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def evaluate(
    data_path: Path,
    config_path: Path = Path("configs/development.yaml"),
    model_path: Path = Path("artifacts/model.joblib"),
    metadata_path: Path = Path("artifacts/metadata.json"),
    output_path: Path = Path("reports/evaluation.json"),
    period: str | None = None,
) -> None:
    """Score a trained artifact on a later month (out-of-period generalisation)."""
    configure_logging()
    report = evaluate_artifact(
        data_path, load_config(config_path), model_path, metadata_path, _period(period)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


@app.command()
def drift(
    data_path: Path,
    config_path: Path = Path("configs/development.yaml"),
    reference_path: Path = Path("artifacts/reference_profile.json"),
    output_path: Path = Path("reports/monitoring/drift.json"),
    fail_on_critical: bool = False,
    model_path: Path = Path("artifacts/model.joblib"),
    metadata_path: Path = Path("artifacts/metadata.json"),
    prediction_drift: bool = True,
) -> None:
    """Compare a raw batch with feature, route, prediction and data-quality references."""
    config = load_config(config_path)
    model = None
    if prediction_drift:
        model, _ = load_artifact(model_path, metadata_path)
    report = drift_report(
        pd.read_parquet(data_path),
        reference_path,
        config.monitoring,
        model=model,
        serving=config.serving if model is not None else None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))
    if fail_on_critical and report["status"] == "critical":
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
