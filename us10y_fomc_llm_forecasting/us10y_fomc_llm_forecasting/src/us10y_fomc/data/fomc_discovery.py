from __future__ import annotations

import re
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from ..progress import progress
from .fred_client import make_official_fed_session

FED_BASE_URL = "https://www.federalreserve.gov"


def meeting_date_span_from_heading(heading: str, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    normalized = re.sub(
        r"(?P<month_a>[A-Za-z]+)/(?P<month_b>[A-Za-z]+)\s+"
        r"(?P<day_a>\d{1,2})\s*[-–]\s*(?P<day_b>\d{1,2})",
        r"\g<month_a> \g<day_a>-\g<month_b> \g<day_b>",
        heading,
    )
    match = re.search(
        r"(?P<month_a>[A-Za-z]+)\s+(?P<day_a>\d{1,2})"
        r"(?:\s*[-–]\s*(?:(?P<month_b>[A-Za-z]+)\s+)?(?P<day_b>\d{1,2}))?",
        normalized,
    )
    if match is None:
        raise ValueError(f"Cannot parse meeting heading: {heading}")
    start_month = match.group("month_a")
    end_month = match.group("month_b") or start_month
    start = pd.Timestamp(pd.to_datetime(f"{start_month} {int(match.group('day_a'))}, {year}"))
    end = pd.Timestamp(
        pd.to_datetime(f"{end_month} {int(match.group('day_b') or match.group('day_a'))}, {year}")
    )
    if end < start:
        end += pd.DateOffset(years=1)
    return start, end


def canonical_minutes_date_from_url(url: str) -> pd.Timestamp:
    match = re.search(r"(?<!\d)((?:19|20)\d{6})(?!\d)", url)
    return (
        pd.NaT if match is None else pd.Timestamp(pd.to_datetime(match.group(1), format="%Y%m%d"))
    )


def discover_fomc_minutes(start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    historical_years = range(start_year, min(end_year, 2020) + 1)
    with make_official_fed_session() as session:
        for year in progress(
            historical_years,
            desc="Historical FOMC indexes",
            total=len(historical_years),
            unit="year",
        ):
            index_url = f"{FED_BASE_URL}/monetarypolicy/fomchistorical{year}.htm"
            response = session.get(index_url, timeout=(10, 120))
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for heading in soup.find_all("h5"):
                title = " ".join(heading.get_text(" ", strip=True).split())
                lowered = title.lower()
                if "meeting" not in lowered or any(
                    term in lowered for term in ("conference call", "notation", "unscheduled")
                ):
                    continue
                panel = heading.find_parent("div", class_="panel")
                root = panel or heading.parent
                text = " ".join(root.get_text(" ", strip=True).split())
                links = []
                for anchor in root.find_all("a", href=True):
                    label = " ".join(anchor.get_text(" ", strip=True).split()).lower()
                    href = anchor["href"]
                    if (
                        "minute" in label
                        or "fomcminutes" in href.lower()
                        or "min.htm" in href.lower()
                    ):
                        links.append(urljoin(index_url, href))
                if not links:
                    continue
                links = list(dict.fromkeys(links))
                links.sort(key=lambda value: (value.lower().endswith(".pdf"), len(value)))
                release_match = re.search(r"Released\s+([^\)]+)", text, re.I)
                start, end = meeting_date_span_from_heading(title, year)
                records.append(
                    {
                        "meeting_year": year,
                        "meeting_heading": title,
                        "meeting_start_date": start,
                        "meeting_date": end,
                        "minutes_release_date": (
                            pd.to_datetime(release_match.group(1), errors="coerce")
                            if release_match
                            else pd.NaT
                        ),
                        "minutes_url": links[0],
                        "source_index_url": index_url,
                    }
                )

        if end_year >= 2021:
            calendar_url = f"{FED_BASE_URL}/monetarypolicy/fomccalendars.htm"
            response = session.get(calendar_url, timeout=(10, 120))
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            current: dict[pd.Timestamp, dict[str, object]] = {}
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"]
                match = re.search(r"fomcminutes(?P<date>20\d{6})\.(?:htm|pdf)$", href, re.I)
                if match is None:
                    continue
                meeting_date = pd.to_datetime(match.group("date"), format="%Y%m%d")
                if not max(start_year, 2021) <= meeting_date.year <= end_year:
                    continue
                minutes_url = urljoin(calendar_url, href)
                container = anchor.find_parent(["div", "li", "p"]) or anchor.parent
                context = " ".join(container.get_text(" ", strip=True).split())
                release_match = re.search(r"Released\s+([^\)]+)", context, re.I)
                candidate = {
                    "meeting_year": meeting_date.year,
                    "meeting_heading": meeting_date.strftime("%B %d Meeting - %Y"),
                    "meeting_start_date": meeting_date,
                    "meeting_date": meeting_date,
                    "minutes_release_date": (
                        pd.to_datetime(release_match.group(1), errors="coerce")
                        if release_match
                        else pd.NaT
                    ),
                    "minutes_url": minutes_url,
                    "source_index_url": calendar_url,
                }
                existing = current.get(meeting_date)
                if existing is None or (
                    str(existing["minutes_url"]).lower().endswith(".pdf")
                    and minutes_url.lower().endswith(".htm")
                ):
                    current[meeting_date] = candidate
            records.extend(current.values())

    raw = pd.DataFrame(records).sort_values("meeting_date").reset_index(drop=True)
    if raw.empty:
        raise RuntimeError("No official scheduled FOMC minutes were discovered.")
    raw["canonical_minutes_date"] = raw.minutes_url.apply(canonical_minutes_date_from_url)
    outside = raw.canonical_minutes_date.notna() & (
        raw.canonical_minutes_date.lt(raw.meeting_start_date)
        | raw.canonical_minutes_date.gt(raw.meeting_date)
    )
    excluded = raw.loc[outside].copy()
    excluded["exclusion_reason"] = "minutes_url_date_outside_meeting_span"
    manifest = raw.loc[~outside].reset_index(drop=True)
    if manifest.meeting_date.duplicated().any():
        raise AssertionError("Duplicate scheduled FOMC meeting dates were discovered.")
    return manifest, excluded.reset_index(drop=True)
