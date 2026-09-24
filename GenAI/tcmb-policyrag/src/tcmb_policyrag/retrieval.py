from __future__ import annotations

from datetime import date

from .embeddings import Embedder, normalize
from .index import VectorStore
from .models import Chunk, SearchFilters, SearchHit


def matches(chunk: Chunk, filters: SearchFilters) -> bool:
    published = date.fromisoformat(chunk.published_date) if chunk.published_date else None
    if filters.date_from and (published is None or published < filters.date_from):
        return False
    if filters.date_to and (published is None or published > filters.date_to):
        return False
    if filters.document_type and chunk.document_type != filters.document_type:
        return False
    return not filters.instruments or bool(set(filters.instruments) & set(chunk.policy_instruments))


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder
        expected = store.manifest.get("embedding_model")
        if expected != embedder.model_name:
            raise ValueError(
                f"Index uses {expected!r}, query embedder uses {embedder.model_name!r}"
            )

    def search(
        self,
        query: str,
        top_k: int = 8,
        filters: SearchFilters | None = None,
        min_score: float = 0.0,
        max_chunks_per_document: int = 2,
    ) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        filters = filters or SearchFilters()
        vector = normalize(self.embedder.embed([query]))
        has_filters = bool(
            filters.date_from or filters.date_to or filters.instruments or filters.document_type
        )
        candidate_k = (
            len(self.store.chunks)
            if has_filters
            else min(len(self.store.chunks), max(top_k * 12, 80))
        )
        scores, indices = self.store.index.search(vector, candidate_k)
        hits: list[SearchHit] = []
        seen_chunks: set[str] = set()
        document_counts: dict[str, int] = {}
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0 or float(score) < min_score:
                continue
            chunk = self.store.chunks[int(index)]
            if chunk.chunk_id in seen_chunks or not matches(chunk, filters):
                continue
            if document_counts.get(chunk.document_id, 0) >= max_chunks_per_document:
                continue
            seen_chunks.add(chunk.chunk_id)
            document_counts[chunk.document_id] = document_counts.get(chunk.document_id, 0) + 1
            hits.append(SearchHit(chunk=chunk, score=float(score), rank=len(hits) + 1))
            if len(hits) == top_k:
                break
        return hits
