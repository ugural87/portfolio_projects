from __future__ import annotations

import hashlib
import os
from pathlib import Path


class CredentialError(RuntimeError):
    """Raised when a required credential is missing or malformed."""


def _read_plain_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if raw.startswith(b"{\\rtf"):
        raise CredentialError(
            f"{path} is an RTF document. Save it as plain text before using it as .env."
        )
    text = raw.decode("utf-8")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise CredentialError(f"Malformed .env line {line_number}: expected NAME=value.")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name:
            values[name] = value
    return values


def _validate_key(name: str, value: str) -> None:
    if name == "FRED_API_KEY":
        if len(value) != 32 or not value.isalnum() or value.lower() != value:
            raise CredentialError(
                "FRED_API_KEY must be a 32-character lowercase alphanumeric string."
            )
    elif name == "OPENAI_API_KEY":
        if len(value) < 20 or not value.startswith("sk-"):
            raise CredentialError("OPENAI_API_KEY does not have the expected key format.")


def load_api_key(name: str, project_root: Path | str, required: bool = True) -> str | None:
    """Load a key from the process environment, then from an untracked project .env file."""
    if name not in {"FRED_API_KEY", "OPENAI_API_KEY"}:
        raise ValueError(f"Unsupported credential name: {name}")
    value = os.getenv(name, "").strip()
    if not value:
        value = _read_plain_env(Path(project_root).resolve() / ".env").get(name, "").strip()
    if not value:
        if required:
            raise CredentialError(
                f"{name} is missing. Put it in the process environment or project-root .env."
            )
        return None
    _validate_key(name, value)
    return value


def credential_fingerprint(value: str) -> str:
    """Return a non-reversible short identifier suitable for audit logs."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
