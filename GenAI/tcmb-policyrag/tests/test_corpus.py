from pathlib import Path

from tcmb_policyrag.chunking import TokenChunker
from tcmb_policyrag.corpus import load_documents


def test_bundled_corpus_is_readable() -> None:
    path = Path(__file__).parents[1] / "data/raw/documents.jsonl"
    documents = load_documents(path)
    assert len(documents) == 481
    assert sum(document.document_type == "press_release" for document in documents) == 478
    assert sum(document.document_type == "reference_page" for document in documents) == 3
    assert all(document.source_url.startswith("https://www.tcmb.gov.tr/") for document in documents)
    assert all("\ufffd" not in chunk.text for chunk in TokenChunker().chunk(documents))
