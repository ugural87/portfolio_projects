from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"


class FredApiError(RuntimeError):
    """Raised when a FRED request or response is invalid."""


def _make_retry_session(accept: str) -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
    session.headers.update({"Accept": accept, "User-Agent": "us10y-fomc-fusion/0.1.2"})
    return session


def make_fred_session() -> requests.Session:
    """Return a retrying session for the FRED JSON API."""
    return _make_retry_session("application/json")


def make_official_fed_session() -> requests.Session:
    """Return a retrying session for Federal Reserve HTML and PDF documents."""
    return _make_retry_session("text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8")


def make_http_session() -> requests.Session:
    """Backward-compatible alias for the FRED JSON session."""
    return make_fred_session()


def _sanitized_error(response: requests.Response, api_key: str) -> str:
    message = "No error details were returned."
    try:
        message = str(response.json().get("error_message", message))
    except ValueError:
        pass
    return message.replace(api_key, "<redacted>")


def fetch_fred_series(
    series_id: str,
    start_date: str,
    api_key: str,
    session: requests.Session | None = None,
) -> pd.Series:
    owns_session = session is None
    active_session = make_fred_session() if session is None else session
    try:
        response = active_session.get(
            f"{FRED_API_BASE_URL}/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start_date,
                "sort_order": "asc",
                "units": "lin",
                "output_type": 1,
                "limit": 100_000,
            },
            timeout=(10, 120),
        )
        if response.status_code != 200:
            raise FredApiError(
                f"FRED returned HTTP {response.status_code} for {series_id}: "
                f"{_sanitized_error(response, api_key)}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise FredApiError(f"FRED returned invalid JSON for {series_id}.") from exc
        observations = payload.get("observations")
        if not isinstance(observations, list) or not observations:
            raise FredApiError(f"FRED returned no observations for {series_id}.")
        if int(payload.get("count", len(observations))) > len(observations):
            raise FredApiError(f"FRED response for {series_id} was truncated.")
        frame = pd.DataFrame.from_records(observations)
        if not {"date", "value"}.issubset(frame.columns):
            raise FredApiError(f"FRED response for {series_id} lacks date or value fields.")
        dates = pd.to_datetime(frame["date"], errors="raise")
        values = pd.to_numeric(frame["value"].replace(".", np.nan), errors="coerce")
        series = pd.Series(values.to_numpy(), index=dates, name=series_id)
        return series.loc[~series.index.duplicated(keep="last")].sort_index()
    except requests.Timeout as exc:
        raise FredApiError(f"FRED timed out while fetching {series_id}.") from exc
    except requests.RequestException as exc:
        raise FredApiError(f"FRED connection failed while fetching {series_id}.") from exc
    finally:
        if owns_session:
            active_session.close()
