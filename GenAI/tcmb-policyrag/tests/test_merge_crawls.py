import json
from pathlib import Path

import pytest

from scripts.merge_crawls import merge


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def make_shard(root: Path, name: str, fetched_at: str, text: str) -> Path:
    shard = root / name
    (shard / "raw_html").mkdir(parents=True)
    url = "https://www.tcmb.gov.tr/example"
    document = {
        "id": "tcmb-example",
        "source_url": url,
        "title": "Zorunlu Karşılıklara İlişkin Basın Duyurusu",
        "text": text,
        "published_date": "2024-01-01",
        "document_type": "press_release",
        "policy_instruments": [],
        "fetched_at_utc": fetched_at,
        "raw_html_path": "raw_html/example.html",
        "date_mismatch": False,
    }
    (shard / "documents.jsonl").write_text(json.dumps(document) + "\n", encoding="utf-8")
    (shard / "raw_html" / "example.html").write_text(text, encoding="utf-8")
    write_json(
        shard / "archive_index.json",
        {
            "entries": [
                {
                    "url": url,
                    "title": document["title"],
                    "date": "2024-01-01",
                    "archive_year": 2024,
                }
            ],
            "excluded_documents": [],
            "unparsed_archive_boxes": [],
        },
    )
    write_json(
        shard / "crawl_report.json",
        {"run_at_utc": fetched_at, "errors": []},
    )
    return shard


@pytest.mark.parametrize(
    ("newer_name", "older_name"),
    [("a_newer", "z_older"), ("z_newer", "a_older")],
)
def test_merge_keeps_newest_document_and_matching_html(
    tmp_path: Path, newer_name: str, older_name: str
) -> None:
    newer = make_shard(tmp_path, newer_name, "2026-09-24T12:00:00+00:00", "new")
    older = make_shard(tmp_path, older_name, "2026-09-24T10:00:00+00:00", "old")
    output = tmp_path / "merged"

    report = merge([newer, older], output)

    document = json.loads((output / "documents.jsonl").read_text(encoding="utf-8"))
    assert document["text"] == "new"
    assert (output / "raw_html" / "example.html").read_text(encoding="utf-8") == "new"
    assert report["merge_conflicts"] == 1
    assert report["failures"] == 0
