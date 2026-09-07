from __future__ import annotations

from pathlib import Path

from us10y_fomc.config import ProjectPaths
from us10y_fomc.progress import progress_enabled
from us10y_fomc.workflow import load_official_documents

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_text_paths_are_rebased_to_the_current_project() -> None:
    paths = ProjectPaths(ROOT)
    documents = load_official_documents(paths)

    assert len(documents) == 267
    assert documents.text_path.map(lambda value: Path(value).is_file()).all()
    assert documents.text_path.map(
        lambda value: paths.minutes_text.resolve() in Path(value).resolve().parents
    ).all()


def test_progress_can_be_disabled_for_ci(monkeypatch) -> None:
    monkeypatch.setenv("US10Y_PROGRESS", "0")
    assert not progress_enabled()
    monkeypatch.setenv("US10Y_PROGRESS", "1")
    assert progress_enabled()
