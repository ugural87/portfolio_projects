from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .extraction_cache import append_jsonl, load_jsonl


@dataclass(frozen=True)
class TokenPrices:
    input_usd_per_million: float
    output_usd_per_million: float


def usage_tokens(usage: dict | None) -> tuple[int, int]:
    if not isinstance(usage, dict):
        return 0, 0
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    for attempt in usage.get("incomplete_attempts") or []:
        attempt_usage = attempt.get("usage") if isinstance(attempt, dict) else None
        if isinstance(attempt_usage, dict):
            input_tokens += int(attempt_usage.get("input_tokens") or 0)
            output_tokens += int(attempt_usage.get("output_tokens") or 0)
    return input_tokens, output_tokens


def token_cost(usage: dict | None, prices: TokenPrices) -> float:
    input_tokens, output_tokens = usage_tokens(usage)
    return (
        input_tokens * prices.input_usd_per_million
        + output_tokens * prices.output_usd_per_million
    ) / 1_000_000


class CostLedger:
    def __init__(self, path: Path, budget_scope: str, prices: TokenPrices) -> None:
        self.path = path
        self.budget_scope = budget_scope
        self.prices = prices

    def spent(self) -> float:
        return sum(
            float(row.get("cost_usd") or 0.0)
            for row in load_jsonl(self.path)
            if row.get("budget_scope") == self.budget_scope
        )

    def reserve(
        self,
        meeting_date: str,
        input_tokens: int,
        primary_output_tokens: int,
        retry_output_tokens: int,
        hard_budget_usd: float,
    ) -> float:
        reserve = (
            2 * input_tokens * self.prices.input_usd_per_million
            + (primary_output_tokens + retry_output_tokens)
            * self.prices.output_usd_per_million
        ) / 1_000_000
        if hard_budget_usd <= 0:
            raise ValueError("A positive hard paid-run budget is required.")
        if self.spent() + reserve > hard_budget_usd:
            raise RuntimeError(
                f"Paid-run budget stop before {meeting_date}: spent=${self.spent():.4f}, "
                f"reserve=${reserve:.4f}, cap=${hard_budget_usd:.4f}."
            )
        append_jsonl(
            self.path,
            {
                "budget_scope": self.budget_scope,
                "current_meeting_date": meeting_date,
                "cost_usd": reserve,
                "status": "reserved_before_api_call",
            },
        )
        return reserve

    def reconcile(self, meeting_date: str, reserve: float, usage: dict, status: str) -> float:
        actual = token_cost(usage, self.prices)
        input_tokens, output_tokens = usage_tokens(usage)
        append_jsonl(
            self.path,
            {
                "budget_scope": self.budget_scope,
                "current_meeting_date": meeting_date,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "actual_attempt_cost_usd": actual,
                "reserved_cost_usd": reserve,
                "cost_usd": actual - reserve,
                "status": status,
            },
        )
        return actual

