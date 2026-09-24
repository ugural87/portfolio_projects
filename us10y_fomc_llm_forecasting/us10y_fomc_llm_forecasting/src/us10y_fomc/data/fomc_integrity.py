from __future__ import annotations

import numpy as np
import pandas as pd


def audit_fomc_documents(
    manifest: pd.DataFrame,
    excluded: pd.DataFrame,
    documents: pd.DataFrame,
    start_year: int,
    today: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    years = pd.Index(range(start_year, today.year + 1), name="meeting_year")
    coverage = documents.groupby("meeting_year").size().reindex(years, fill_value=0).rename(
        "scheduled_minutes"
    ).to_frame()
    coverage["complete_year"] = coverage.index < today.year
    coverage["status"] = np.where(
        (~coverage.complete_year) | coverage.scheduled_minutes.between(7, 12), "PASS", "FAIL"
    )
    failures = coverage.loc[coverage.complete_year & coverage.status.eq("FAIL")]
    expected_exclusions = {pd.Timestamp("2003-09-15")}
    actual_exclusions = set(pd.to_datetime(excluded.meeting_date)) if not excluded.empty else set()
    release_dates = pd.to_datetime(documents.minutes_release_date, errors="coerce")
    release_gap = (release_dates - pd.to_datetime(documents.meeting_date)).dt.days
    checks = {
        "non_empty": not documents.empty,
        "chronological": documents.meeting_date.is_monotonic_increasing,
        "unique_meeting_dates": documents.meeting_date.is_unique,
        "unique_document_hashes": documents.text_sha256.is_unique,
        "minimum_text_length": bool((documents.text_chars >= 3_000).all()),
        "maximum_text_length": bool((documents.text_chars <= 200_000).all()),
        "release_date_coverage": float(release_dates.notna().mean()) >= 0.95,
        "release_dates_after_meeting": bool(
            release_gap.dropna().between(1, 90).all()
        ),
        "completed_year_coverage": failures.empty,
        "canonical_url_dates_present": int(manifest.canonical_minutes_date.isna().sum()) == 0,
        "no_unexpected_exclusions": actual_exclusions.issubset(expected_exclusions),
        "unscheduled_2020_03_15_absent": pd.Timestamp("2020-03-15") not in set(documents.meeting_date),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"FOMC document-integrity gate failed: {failed}")
    return coverage, checks
