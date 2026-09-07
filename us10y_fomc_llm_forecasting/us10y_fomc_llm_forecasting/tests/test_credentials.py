from pathlib import Path

import pytest

from us10y_fomc.security.credentials import CredentialError, load_api_key


def test_rtf_textedit_credentials_are_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_bytes(b"{\\rtf1 OPENAI_API_KEY=not-plain-text}")
    with pytest.raises(CredentialError, match="RTF"):
        load_api_key("OPENAI_API_KEY", tmp_path)
