from __future__ import annotations

from _bootstrap import PROJECT_ROOT

from us10y_fomc.config import ProjectPaths, load_project_config
from us10y_fomc.workflow import run_walk_forward_study


if __name__ == "__main__":
    config = load_project_config(PROJECT_ROOT)
    paths = ProjectPaths(PROJECT_ROOT).ensure()
    predictions, metrics = run_walk_forward_study(paths, config)
    print(f"Saved {len(predictions)} out-of-sample predictions across {len(metrics)} folds.")
