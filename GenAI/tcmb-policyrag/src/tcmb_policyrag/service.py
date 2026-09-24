from __future__ import annotations

from pathlib import Path

from .chunking import TokenChunker
from .config import Settings
from .corpus import load_documents
from .embeddings import OpenAIEmbedder
from .generation import AnswerGenerator
from .index import VectorStore
from .models import Answer, SearchFilters, SearchHit
from .retrieval import Retriever


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = OpenAIEmbedder(settings.embedding_model, cache_dir=None)
        self.store = VectorStore.load(settings.index_dir)
        self.retriever = Retriever(self.store, self.embedder)
        self.generator = AnswerGenerator(settings.chat_model)

    @classmethod
    def build_index(
        cls,
        corpus_path: Path,
        settings: Settings,
        max_tokens: int = 450,
        overlap_tokens: int = 70,
    ) -> VectorStore:
        documents = load_documents(corpus_path)
        chunks = TokenChunker(max_tokens, overlap_tokens).chunk(documents)
        embedder = OpenAIEmbedder(
            settings.embedding_model,
            cache_dir=settings.index_dir / "embedding_cache",
            show_progress=True,
        )
        store = VectorStore.build(chunks, embedder)
        store.manifest.update(
            {
                "document_count": len(documents),
                "max_tokens": max_tokens,
                "overlap_tokens": overlap_tokens,
                "corpus_path": str(corpus_path),
            }
        )
        store.save(settings.index_dir)
        return store

    def search(
        self, question: str, filters: SearchFilters | None = None, top_k: int | None = None
    ) -> list[SearchHit]:
        return self.retriever.search(
            question,
            top_k=top_k or self.settings.top_k,
            filters=filters,
            min_score=self.settings.min_score,
        )

    def ask(
        self, question: str, filters: SearchFilters | None = None, top_k: int | None = None
    ) -> Answer:
        hits = self.search(question, filters, top_k)
        return self.generator.answer(question, hits)
