from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from .models import SearchFilters
from .service import RAGService


class EvalCase(BaseModel):
    id: str
    question: str
    relevant_announcements: list[str] = Field(min_length=1)
    filters: SearchFilters = Field(default_factory=SearchFilters)


def load_cases(path: Path) -> list[EvalCase]:
    return [EvalCase.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def evaluate_retrieval(service: RAGService, cases: list[EvalCase], k: int = 5) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for case in cases:
        hits = service.search(case.question, filters=case.filters, top_k=k)
        retrieved = [hit.chunk.announcement_number for hit in hits]
        relevant = set(case.relevant_announcements)
        matched_ranks = [i + 1 for i, number in enumerate(retrieved) if number in relevant]
        recall = len(set(retrieved) & relevant) / len(relevant)
        reciprocal_rank = 1 / min(matched_ranks) if matched_ranks else 0.0
        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        rows.append(
            {
                "id": case.id,
                "retrieved": retrieved,
                "recall_at_k": recall,
                "reciprocal_rank": reciprocal_rank,
            }
        )
    return {
        "k": k,
        "case_count": len(rows),
        "recall_at_k": mean(recalls),
        "mrr": mean(reciprocal_ranks),
        "cases": rows,
    }
