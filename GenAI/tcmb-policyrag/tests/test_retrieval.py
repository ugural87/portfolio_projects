import numpy as np
import pytest
from pydantic import ValidationError

from tcmb_policyrag.index import VectorStore
from tcmb_policyrag.models import Chunk, SearchFilters
from tcmb_policyrag.retrieval import Retriever


class FakeEmbedder:
    model_name = "fake-v1"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            folded = text.casefold()
            vectors.append(
                [
                    float("zorunlu" in folded),
                    float("kkm" in folded),
                    float("likidite" in folded),
                    0.1,
                ]
            )
        return np.asarray(vectors, dtype="float32")


def chunk(cid: str, text: str, date: str, instrument: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id=f"doc-{cid}",
        chunk_index=0,
        text=text,
        title=text,
        source_url=f"https://www.tcmb.gov.tr/{cid}",
        published_date=date,
        announcement_number=cid,
        document_type="press_release",
        policy_instruments=[instrument],
        token_count=5,
    )


def test_dense_retrieval_and_metadata_filter(tmp_path) -> None:
    chunks = [
        chunk("2024-01", "Zorunlu karşılık kararı", "2024-01-01", "reserve_requirements"),
        chunk("2023-02", "KKM dönüşüm kararı", "2023-02-01", "deposit_and_kkm"),
        chunk("2024-03", "Likidite kararı", "2024-03-01", "liquidity"),
    ]
    embedder = FakeEmbedder()
    store = VectorStore.build(chunks, embedder)
    store.save(tmp_path)
    retriever = Retriever(VectorStore.load(tmp_path), embedder)
    hits = retriever.search(
        "zorunlu karşılık nedir",
        top_k=2,
        filters=SearchFilters(date_from="2024-01-01", instruments=["reserve_requirements"]),
    )
    assert [hit.chunk.announcement_number for hit in hits] == ["2024-01"]


def test_search_filters_validate_dates() -> None:
    with pytest.raises(ValidationError):
        SearchFilters(date_from="2024-1-1")
    with pytest.raises(ValidationError):
        SearchFilters(date_from="2025-01-01", date_to="2024-01-01")
