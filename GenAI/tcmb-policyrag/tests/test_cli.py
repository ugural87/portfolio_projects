from datetime import date

import pytest
from typer import BadParameter
from typer.testing import CliRunner

from tcmb_policyrag.cli import app, cli_date


def test_cli_help_loads() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "build-index" in result.stdout
    assert "search" in result.stdout


def test_cli_date_parses_iso_date() -> None:
    assert cli_date("2024-01-31", "--date-from") == date(2024, 1, 31)


def test_cli_date_rejects_invalid_date() -> None:
    with pytest.raises(BadParameter):
        cli_date("2024-13-01", "--date-from")
