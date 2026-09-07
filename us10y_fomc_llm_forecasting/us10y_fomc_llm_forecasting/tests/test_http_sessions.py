from __future__ import annotations

from us10y_fomc.data.fred_client import make_fred_session, make_official_fed_session


def test_fred_and_federal_reserve_sessions_negotiate_separate_content_types() -> None:
    with make_fred_session() as fred, make_official_fed_session() as federal_reserve:
        assert fred.headers["Accept"] == "application/json"
        assert "text/html" in federal_reserve.headers["Accept"]
        assert "application/pdf" in federal_reserve.headers["Accept"]
        assert fred.headers["Accept"] != federal_reserve.headers["Accept"]
