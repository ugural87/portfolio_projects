from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import tiktoken

from .models import Chunk, Document


class TokenChunker:
    def __init__(self, max_tokens: int = 450, overlap_tokens: int = 70) -> None:
        if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("Require max_tokens > overlap_tokens >= 0")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def _decode_window(self, tokens: list[int]) -> tuple[str, int]:
        """Decode without inserting Unicode replacement characters at token boundaries."""
        max_trim = min(4, len(tokens) - 1)
        for total_trim in range(0, max_trim + 1):
            for left_trim in range(0, total_trim + 1):
                right_trim = total_trim - left_trim
                end = len(tokens) - right_trim if right_trim else len(tokens)
                candidate = tokens[left_trim:end]
                if not candidate:
                    continue
                try:
                    text = self.encoding.decode(candidate, errors="strict").strip()
                except UnicodeDecodeError:
                    continue
                if "\ufffd" not in text:
                    return text, len(candidate)
        raise UnicodeError("Could not decode token window without replacement characters")

    def _windows(self, text: str) -> Iterable[tuple[str, int]]:
        paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
        tokens: list[int] = []
        for paragraph in paragraphs:
            paragraph_tokens = self.encoding.encode(paragraph + "\n")
            tokens.extend(paragraph_tokens)
        step = self.max_tokens - self.overlap_tokens
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.max_tokens]
            if not window:
                break
            yield self._decode_window(window)
            if start + self.max_tokens >= len(tokens):
                break

    def chunk(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            for index, (text, token_count) in enumerate(self._windows(document.text)):
                digest = hashlib.sha256(f"{document.id}:{index}:{text}".encode()).hexdigest()[:20]
                chunks.append(
                    Chunk(
                        chunk_id=f"chk-{digest}",
                        document_id=document.id,
                        chunk_index=index,
                        text=text,
                        title=document.title,
                        source_url=document.source_url,
                        published_date=document.published_date,
                        announcement_number=document.announcement_number,
                        document_type=document.document_type,
                        policy_instruments=document.policy_instruments,
                        token_count=token_count,
                    )
                )
        return chunks
