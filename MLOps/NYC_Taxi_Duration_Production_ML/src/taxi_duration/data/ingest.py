from __future__ import annotations

import hashlib
import time
from pathlib import Path

import httpx
import structlog

from taxi_duration.data.validation import Period

LOGGER = structlog.get_logger(__name__)


_NOT_PUBLISHED = frozenset({403, 404})  # CloudFront/S3 answer 403 for a missing object


class SourceProbeError(RuntimeError):
    """The source could not be probed; availability is unknown, not negative."""


def _probe_status(url: str, timeout_seconds: float) -> int:
    response = httpx.head(url, follow_redirects=True, timeout=timeout_seconds)
    if response.status_code != 405:
        return response.status_code
    with httpx.stream(
        "GET",
        url,
        headers={"Range": "bytes=0-0"},
        follow_redirects=True,
        timeout=timeout_seconds,
    ) as range_response:
        return range_response.status_code


def source_available(
    period: Period,
    url_template: str,
    timeout_seconds: float = 20.0,
    attempts: int = 3,
    backoff_seconds: float = 2.0,
) -> bool:
    """Return whether the immutable TLC object exists; calendar lag alone is not proof.

    Only 200/206 mean published and only 403/404 mean not published. Network errors and any
    other status are retried and then raised, so a transient outage cannot silently move the
    training window to an older month.
    """
    url = url_template.format(year=period.year, month=period.month)
    last_problem = "no attempt made"
    for attempt in range(1, attempts + 1):
        try:
            status = _probe_status(url, timeout_seconds)
        except httpx.HTTPError as exc:
            last_problem = f"{type(exc).__name__}: {exc}"
        else:
            if status in {200, 206}:
                return True
            if status in _NOT_PUBLISHED:
                return False
            last_problem = f"unexpected HTTP status {status}"
        LOGGER.warning("source_probe_retry", url=url, attempt=attempt, problem=last_problem)
        if attempt < attempts:
            time.sleep(backoff_seconds * attempt)
    raise SourceProbeError(f"could not determine availability of {url}: {last_problem}")


def latest_available_period(
    start: Period,
    url_template: str,
    max_lookback_months: int,
) -> Period:
    """Probe backwards from *start* and select the newest actually published source file."""
    for offset in range(max_lookback_months):
        candidate = start.shift(-offset)
        if source_available(candidate, url_template):
            return candidate
    raise RuntimeError(
        f"no published TLC source found from {start} across {max_lookback_months} months"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_month(
    year: int,
    month: int,
    destination_dir: Path,
    url_template: str,
    *,
    overwrite: bool = False,
) -> tuple[Path, str]:
    if year < 2009 or month not in range(1, 13):
        raise ValueError("year must be >= 2009 and month must be 1..12")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"yellow_tripdata_{year}-{month:02d}.parquet"
    if destination.exists() and not overwrite:
        return destination, sha256_file(destination)

    url = url_template.format(year=year, month=month)
    temporary = destination.with_suffix(".parquet.part")
    LOGGER.info("download_started", url=url, destination=str(destination))
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    output.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    checksum = sha256_file(destination)
    LOGGER.info("download_completed", bytes=destination.stat().st_size, sha256=checksum)
    return destination, checksum
