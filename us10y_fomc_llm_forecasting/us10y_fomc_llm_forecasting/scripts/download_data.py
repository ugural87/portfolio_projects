from __future__ import annotations

from _bootstrap import PROJECT_ROOT

from us10y_fomc.config import ProjectPaths, load_project_config
from us10y_fomc.security.credentials import load_api_key
from us10y_fomc.workflow import download_official_data


if __name__ == "__main__":
    config = load_project_config(PROJECT_ROOT)
    paths = ProjectPaths(PROJECT_ROOT).ensure()
    fred_key = load_api_key("FRED_API_KEY", PROJECT_ROOT)
    print(download_official_data(paths, config, fred_key))
