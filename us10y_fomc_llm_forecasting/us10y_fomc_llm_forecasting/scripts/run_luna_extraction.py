from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from us10y_fomc.config import ProjectPaths, load_project_config
from us10y_fomc.security.credentials import load_api_key
from us10y_fomc.workflow import run_minutes_extraction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot",
        action="append",
        default=[],
        help="Validated pilot JSONL path; repeat for multiple files.",
    )
    arguments = parser.parse_args()
    config = load_project_config(PROJECT_ROOT)
    paths = ProjectPaths(PROJECT_ROOT).ensure()
    key = load_api_key(
        "OPENAI_API_KEY",
        PROJECT_ROOT,
        required=config.runtime.run_openai_preflight or config.runtime.run_paid_extraction,
    )
    result = run_minutes_extraction(
        paths, config, key, tuple(Path(value).expanduser() for value in arguments.pilot)
    )
    print(
        {
            "ready": result.ready,
            "accepted": len(result.cached_records),
            "required": result.required_pairs,
            "missing": len(result.missing_dates),
            "spent_usd": result.spent_usd,
        }
    )
