from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Document(BaseModel):
    id: str
    title: str
    text: str
    source_url: str
    published_date: str | None = None
    announcement_number: str | None = None
    document_type: str = "press_release"
    policy_instruments: list[str] = Field(default_factory=list)
    content_sha256: str | None = None


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    title: str
    source_url: str
    published_date: str | None
    announcement_number: str | None
    document_type: str
    policy_instruments: list[str]
    token_count: int

    def embedding_text(self) -> str:
        date = self.published_date or "Tarih belirtilmemiş"
        number = self.announcement_number or "Numara belirtilmemiş"
        return f"Başlık: {self.title}\nTarih: {date}\nDuyuru: {number}\n{self.text}"


class SearchFilters(BaseModel):
    date_from: date | None = None
    date_to: date | None = None
    instruments: list[str] = Field(default_factory=list)
    document_type: str | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> SearchFilters:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from cannot be later than date_to")
        return self


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    rank: int


class Citation(BaseModel):
    citation_id: str
    announcement_number: str | None
    title: str
    published_date: str | None
    source_url: str
    quote: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    retrieved: list[SearchHit]
    model: str
    abstained: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)
