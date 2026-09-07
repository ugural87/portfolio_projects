from __future__ import annotations

from _bootstrap import PROJECT_ROOT

from us10y_fomc.config import ProjectPaths
from us10y_fomc.reporting import build_postprocess_report


if __name__ == "__main__":
    print(build_postprocess_report(ProjectPaths(PROJECT_ROOT).ensure()))
