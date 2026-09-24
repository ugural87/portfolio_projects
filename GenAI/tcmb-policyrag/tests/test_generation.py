from types import SimpleNamespace

from tcmb_policyrag.generation import AnswerGenerator
from tcmb_policyrag.models import Chunk, SearchHit


class FakeResponses:
    def create(self, **_kwargs):
        return SimpleNamespace(
            output_text=(
                '{"answer":"Geçerli [K2], bilinmeyen [K9] ve sonra [K1].",'
                '"citation_ids":["K1","K2","K9"],"abstained":false}'
            )
        )


def hit(number: int) -> SearchHit:
    chunk = Chunk(
        chunk_id=f"chunk-{number}",
        document_id=f"doc-{number}",
        chunk_index=0,
        text=f"Kaynak {number}",
        title=f"Duyuru {number}",
        source_url=f"https://www.tcmb.gov.tr/{number}",
        published_date="2024-01-01",
        announcement_number=f"2024-0{number}",
        document_type="press_release",
        policy_instruments=[],
        token_count=2,
    )
    return SearchHit(chunk=chunk, score=0.9, rank=number)


def test_citations_are_deterministic_and_unknown_ids_are_removed() -> None:
    generator = AnswerGenerator.__new__(AnswerGenerator)
    generator.model = "fake"
    generator.client = SimpleNamespace(responses=FakeResponses())
    answer = generator.answer("Soru", [hit(1), hit(2)])
    assert [citation.citation_id for citation in answer.citations] == ["K1", "K2"]
    assert "[K9]" not in answer.answer
    assert answer.diagnostics["unknown_citation_ids"] == ["K9"]
