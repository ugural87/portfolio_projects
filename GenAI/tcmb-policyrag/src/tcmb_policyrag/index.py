from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import faiss

from .embeddings import Embedder, normalize
from .models import Chunk


@dataclass
class VectorStore:
    index: faiss.Index
    chunks: list[Chunk]
    manifest: dict[str, object]

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> VectorStore:
        if not chunks:
            raise ValueError("Cannot index an empty chunk collection")
        vectors = normalize(embedder.embed([chunk.embedding_text() for chunk in chunks]))
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "embedding_model": embedder.model_name,
            "dimension": int(vectors.shape[1]),
            "chunk_count": len(chunks),
        }
        return cls(index=index, chunks=chunks, manifest=manifest)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "vectors.faiss"))
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(chunk.model_dump_json() + "\n")
        (directory / "manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> VectorStore:
        required = [
            directory / "vectors.faiss",
            directory / "chunks.jsonl",
            directory / "manifest.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Index is incomplete. Missing: {', '.join(missing)}")
        index = faiss.read_index(str(required[0]))
        with required[1].open(encoding="utf-8") as handle:
            chunks = [Chunk.model_validate_json(line) for line in handle if line.strip()]
        manifest = json.loads(required[2].read_text(encoding="utf-8"))
        if index.ntotal != len(chunks) or manifest.get("chunk_count") != len(chunks):
            raise ValueError("FAISS index, chunk metadata and manifest counts do not match")
        return cls(index=index, chunks=chunks, manifest=manifest)
