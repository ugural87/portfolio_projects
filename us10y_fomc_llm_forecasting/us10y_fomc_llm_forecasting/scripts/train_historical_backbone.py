from __future__ import annotations

from _bootstrap import PROJECT_ROOT

from us10y_fomc.config import ProjectPaths, load_project_config
from us10y_fomc.workflow import ensure_historical_backbone, prepare_aligned_events


if __name__ == "__main__":
    config = load_project_config(PROJECT_ROOT)
    paths = ProjectPaths(PROJECT_ROOT).ensure()
    _, bundle, aligned, _, _ = prepare_aligned_events(paths, config)
    _, metadata = ensure_historical_backbone(
        paths, config, bundle, aligned.meeting_date.min()
    )
    print({key: value for key, value in metadata.items() if key != "state_dict"})
