from tcmb_policyrag.chunking import TokenChunker
from tcmb_policyrag.models import Document


def test_chunks_preserve_metadata_and_overlap() -> None:
    document = Document(
        id="doc-1",
        title="Test Duyurusu",
        text=" ".join(["zorunlu karşılık değişikliği"] * 150),
        source_url="https://www.tcmb.gov.tr/test",
        published_date="2024-05-23",
        announcement_number="2024-29",
        policy_instruments=["reserve_requirements"],
    )
    chunks = TokenChunker(max_tokens=80, overlap_tokens=20).chunk([document])
    assert len(chunks) > 1
    assert all(chunk.token_count <= 80 for chunk in chunks)
    assert all(chunk.announcement_number == "2024-29" for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all("\ufffd" not in chunk.text for chunk in chunks)
