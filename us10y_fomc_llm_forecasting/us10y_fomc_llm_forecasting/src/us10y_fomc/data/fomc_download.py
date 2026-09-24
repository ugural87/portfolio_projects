from __future__ import annotations

import hashlib
import io
import re
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

from ..progress import progress
from .fred_client import make_official_fed_session


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_document_text(content: bytes, url: str, content_type: str) -> str:
    if url.lower().endswith(".pdf") or "application/pdf" in content_type.lower():
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        main = soup.find("main") or soup.find(id="content") or soup.body or soup
        text = main.get_text("\n", strip=True)
    return clean_text(text)


def download_minutes(manifest: pd.DataFrame, text_directory: Path) -> pd.DataFrame:
    text_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    records = manifest.to_dict("records")
    with make_official_fed_session() as session:
        for record in progress(
            records,
            desc="Official FOMC minutes",
            total=len(records),
            unit="document",
        ):
            date_key = pd.Timestamp(record["meeting_date"]).strftime("%Y-%m-%d")
            path = text_directory / f"{date_key}.txt"
            if path.exists():
                text = path.read_text(encoding="utf-8")
            else:
                response = session.get(str(record["minutes_url"]), timeout=(10, 180))
                response.raise_for_status()
                text = extract_document_text(
                    response.content,
                    str(record["minutes_url"]),
                    response.headers.get("content-type", ""),
                )
                path.write_text(text, encoding="utf-8")
                time.sleep(0.05)
            rows.append(
                {
                    **record,
                    "text_path": str(path),
                    "text_chars": len(text),
                    "text_words": len(re.findall(r"\b\w+\b", text)),
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
    return pd.DataFrame(rows).sort_values("meeting_date").reset_index(drop=True)
