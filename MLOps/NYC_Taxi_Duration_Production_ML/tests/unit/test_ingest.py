import httpx
import pytest

from taxi_duration.data import ingest
from taxi_duration.data.validation import Period


def test_latest_available_period_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    available = {Period(2026, 5)}

    def fake_available(period: Period, _: str, timeout_seconds: float = 20.0) -> bool:
        del timeout_seconds
        return period in available

    monkeypatch.setattr(ingest, "source_available", fake_available)
    resolved = ingest.latest_available_period(Period(2026, 7), "ignored", 6)
    assert resolved == Period(2026, 5)


def test_latest_available_period_fails_after_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    def never_available(period: Period, template: str, timeout_seconds: float = 20.0) -> bool:
        del period, template, timeout_seconds
        return False

    monkeypatch.setattr(ingest, "source_available", never_available)
    try:
        ingest.latest_available_period(Period(2026, 7), "ignored", 2)
    except RuntimeError as error:
        assert "no published TLC source" in str(error)
    else:
        raise AssertionError("bounded source lookup should fail explicitly")


@pytest.mark.parametrize(
    ("status", "expected"), [(200, True), (206, True), (403, False), (404, False)]
)
def test_source_status_mapping(
    monkeypatch: pytest.MonkeyPatch, status: int, expected: bool
) -> None:
    monkeypatch.setattr(ingest, "_probe_status", lambda url, timeout: status)
    assert ingest.source_available(Period(2026, 5), "{year}-{month}", attempts=1) is expected


def test_network_error_is_not_treated_as_unpublished(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(url: str, timeout: float) -> int:
        raise httpx.ConnectError("connection reset")

    monkeypatch.setattr(ingest, "_probe_status", broken)
    with pytest.raises(ingest.SourceProbeError, match="ConnectError"):
        ingest.source_available(Period(2026, 5), "{year}-{month}", backoff_seconds=0.0)


def test_transient_error_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter([503, 200])
    monkeypatch.setattr(ingest, "_probe_status", lambda url, timeout: next(responses))
    assert ingest.source_available(Period(2026, 5), "{year}-{month}", backoff_seconds=0.0)


def test_probe_error_stops_the_lookback(monkeypatch: pytest.MonkeyPatch) -> None:
    def outage(period: Period, template: str, timeout_seconds: float = 20.0) -> bool:
        raise ingest.SourceProbeError("outage")

    monkeypatch.setattr(ingest, "source_available", outage)
    with pytest.raises(ingest.SourceProbeError):
        ingest.latest_available_period(Period(2026, 7), "ignored", 6)
