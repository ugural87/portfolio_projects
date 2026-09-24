from __future__ import annotations

import json
from pathlib import Path

from .models import Document


def load_documents(path: Path, include_reference: bool = True) -> list[Document]:
    documents: list[Document] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not include_reference and raw.get("document_type") == "reference_page":
                continue
            document = Document.model_validate(raw)
            if document.id in seen:
                raise ValueError(f"Duplicate document id at line {line_number}: {document.id}")
            if not document.text.strip():
                raise ValueError(f"Empty document at line {line_number}: {document.id}")
            seen.add(document.id)
            documents.append(document)
    if not documents:
        raise ValueError(f"No documents found in {path}")
    return documents
