from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from us10y_fomc.config import ProjectPaths, load_project_config
from us10y_fomc.reporting import build_postprocess_report
from us10y_fomc.security.credentials import load_api_key
from us10y_fomc.workflow import (
    download_official_data,
    run_minutes_extraction,
    run_walk_forward_study,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="append", default=[])
    arguments = parser.parse_args()
    config = load_project_config(PROJECT_ROOT)
    paths = ProjectPaths(PROJECT_ROOT).ensure()
    fred_key = load_api_key("FRED_API_KEY", PROJECT_ROOT)
    if config.runtime.download_fomc_minutes:
        print(download_official_data(paths, config, fred_key))
    openai_key = load_api_key(
        "OPENAI_API_KEY",
        PROJECT_ROOT,
        required=config.runtime.run_openai_preflight or config.runtime.run_paid_extraction,
    )
    extraction = run_minutes_extraction(
        paths,
        config,
        openai_key,
        tuple(Path(value).expanduser() for value in arguments.pilot),
    )
    if not extraction.ready:
        raise RuntimeError(
            f"Extraction is incomplete: {len(extraction.missing_dates)} pair(s) remain."
        )
    predictions, metrics = run_walk_forward_study(paths, config)
    print(f"Walk-forward complete: {len(predictions)} predictions, {len(metrics)} folds.")
    print(build_postprocess_report(paths))
