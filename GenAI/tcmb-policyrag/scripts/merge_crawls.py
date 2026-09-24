"""Merge deterministic crawler shards into one reproducible corpus snapshot."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

try:
    from .tcmb_crawler import instrument_tags
except ImportError:  # direct script execution
    from tcmb_crawler import instrument_tags


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def upsert_newest(
    values: dict[str, dict],
    versions: dict[str, tuple[str, str]],
    key: str,
    value: dict,
    version: tuple[str, str],
) -> None:
    if key not in values or version > versions[key]:
        values[key] = value
        versions[key] = version


def merge(inputs: list[Path], output: Path) -> dict:
    if not inputs:
        raise ValueError("At least one input crawl directory is required")
    output.mkdir(parents=True, exist_ok=True)
    raw_output = output / "raw_html"
    raw_output.mkdir(parents=True, exist_ok=True)

    documents_by_url: dict[str, dict] = {}
    document_versions: dict[str, tuple[str, str]] = {}
    document_sources: dict[str, Path] = {}
    archive_entries: dict[str, dict] = {}
    archive_versions: dict[str, tuple[str, str]] = {}
    excluded: dict[str, dict] = {}
    excluded_versions: dict[str, tuple[str, str]] = {}
    unparsed: dict[str, dict] = {}
    unparsed_versions: dict[str, tuple[str, str]] = {}
    errors: list[dict] = []
    conflicting_urls: set[str] = set()
    shard_runs: dict[str, str] = {}

    for directory in sorted(inputs):
        report = read_json(directory / "crawl_report.json")
        shard_time = report.get("run_at_utc") or report.get("snapshot_at_utc") or ""
        shard_version = (shard_time, directory.as_posix())
        shard_runs[directory.name] = shard_time
        errors.extend(report.get("errors", []))

        for line in (directory / "documents.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            document = json.loads(line)
            document["policy_instruments"] = instrument_tags(
                document["title"] + "\n" + document["text"]
            )
            url = document["source_url"]
            if url in documents_by_url:
                conflicting_urls.add(url)
            version = (document.get("fetched_at_utc") or shard_time, directory.as_posix())
            if url not in documents_by_url or version > document_versions[url]:
                documents_by_url[url] = document
                document_versions[url] = version
                document_sources[url] = directory

        archive = read_json(directory / "archive_index.json")
        for item in archive.get("entries", []):
            upsert_newest(archive_entries, archive_versions, item["url"], item, shard_version)
        for item in archive.get("excluded_documents", []):
            key = f"{item.get('title', '')}|{item.get('date', '')}"
            upsert_newest(excluded, excluded_versions, key, item, shard_version)
        for item in archive.get("unparsed_archive_boxes", []):
            key = f"{item.get('title', '')}|{item.get('date', '')}"
            upsert_newest(unparsed, unparsed_versions, key, item, shard_version)

    documents = sorted(
        documents_by_url.values(),
        key=lambda item: (item.get("published_date") or "9999", item["source_url"]),
    )
    entries = sorted(
        archive_entries.values(),
        key=lambda item: (item.get("date") or "9999", item["url"]),
    )
    excluded = sorted(
        excluded.values(),
        key=lambda item: (item.get("date") or "9999", item.get("title", "")),
    )
    unparsed = sorted(unparsed.values(), key=lambda item: item.get("title", ""))

    expected_raw_html: set[Path] = set()
    for document in documents:
        relative = Path(document["raw_html_path"])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(
                {
                    "url": document["source_url"],
                    "stage": "merge_raw_html",
                    "error": f"Unsafe raw_html_path: {relative}",
                }
            )
            continue
        source = document_sources[document["source_url"]] / relative
        destination = output / relative
        expected_raw_html.add(destination.resolve())
        if not source.is_file():
            errors.append(
                {
                    "url": document["source_url"],
                    "stage": "merge_raw_html",
                    "error": f"Missing source file: {source}",
                }
            )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for stale in raw_output.glob("*.html"):
        if stale.resolve() not in expected_raw_html:
            stale.unlink()

    with (output / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")

    archive_payload = {
        "start_year": min(item["archive_year"] for item in entries),
        "end_year": max(item["archive_year"] for item in entries),
        "discovered_count": len(entries),
        "excluded_count": len(excluded),
        "excluded_documents": excluded,
        "unparsed_archive_boxes": unparsed,
        "entries": entries,
    }
    (output / "archive_index.json").write_text(
        json.dumps(archive_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "discovered": len(entries),
        "selected": len(entries),
        "exported": len(documents),
        "failures": len(errors),
        "errors": errors,
        "scope": "all",
        "excluded_documents": excluded,
        "unparsed_archive_boxes": unparsed,
        "discovered_by_year": dict(
            sorted(Counter(str(item["archive_year"]) for item in entries).items())
        ),
        "exported_by_year": dict(
            sorted(
                Counter(
                    item["published_date"][:4] if item.get("published_date") else "unknown"
                    for item in documents
                ).items()
            )
        ),
        "instrument_counts": dict(
            sorted(Counter(tag for item in documents for tag in item["policy_instruments"]).items())
        ),
        "date_mismatch_count": sum(bool(item.get("date_mismatch")) for item in documents),
        "unknown_date_count": sum(not item.get("published_date") for item in documents),
        "snapshot_at_utc": max(item["fetched_at_utc"] for item in documents),
        "source_shards": [path.name for path in sorted(inputs)],
        "source_shard_runs": dict(sorted(shard_runs.items())),
        "merge_conflicts": len(conflicting_urls),
    }
    (output / "crawl_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = merge(args.inputs, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
