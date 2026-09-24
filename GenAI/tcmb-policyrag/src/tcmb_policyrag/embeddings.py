from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from openai import OpenAI


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    def __init__(
        self,
        model: str,
        cache_dir: Path | None = None,
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> None:
        self.model_name = model
        self.client = OpenAI()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.show_progress = show_progress
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\0{text}".encode()).hexdigest()

    def _report_progress(self, done: int, total: int, cache_hits: int) -> None:
        if not self.show_progress:
            return
        percent = 100.0 if total == 0 else done / total * 100
        print(
            f"Embedding progress: {done:,}/{total:,} ({percent:5.1f}%) "
            f"| cache hits: {cache_hits:,}",
            file=sys.stderr,
            flush=True,
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str, Path | None]] = []
        for index, text in enumerate(texts):
            cache_path = self.cache_dir / f"{self._key(text)}.json" if self.cache_dir else None
            if cache_path and cache_path.exists():
                vectors[index] = json.loads(cache_path.read_text())
            else:
                missing.append((index, text, cache_path))
        cache_hits = len(texts) - len(missing)
        self._report_progress(cache_hits, len(texts), cache_hits)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model_name, input=[item[1] for item in batch]
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            for (index, _text, cache_path), result in zip(batch, ordered, strict=True):
                vector = result.embedding
                vectors[index] = vector
                if cache_path:
                    cache_path.write_text(json.dumps(vector), encoding="utf-8")
            self._report_progress(cache_hits + start + len(batch), len(texts), cache_hits)
        if any(vector is None for vector in vectors):
            raise RuntimeError("Embedding response was incomplete")
        return np.asarray(vectors, dtype="float32")


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding provider returned a zero vector")
    return cast(np.ndarray, (vectors / norms).astype("float32"))
