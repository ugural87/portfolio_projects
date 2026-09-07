from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI
else:
    OpenAI = Any

from ..config import ExtractionConfig
from .feature_contract import FEATURE_RUBRIC, LLM_SCHEMA_VERSION
from .grounding import audit_grounding, hydrate_comparison
from .schemas import FOMC_SELECTION_JSON_SCHEMA, FomcMinutesSelection
from .sentence_catalog import (
    SENTENCE_CATALOG_VERSION,
    build_sentence_catalog,
    format_sentence_catalog,
)


class IncompleteComparison(RuntimeError):
    def __init__(self, reason: str, usage: dict, attempts: list[dict]):
        super().__init__(f"Incomplete Luna response: {reason}")
        self.reason = reason
        self.usage = usage
        self.attempts = attempts


class SchemaValidationFailure(RuntimeError):
    def __init__(self, error: str, raw_output: str, usage: dict):
        super().__init__("Completed response failed local schema validation.")
        self.error = error
        self.raw_output_sha256 = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        self.raw_output_excerpt = raw_output[:4_000]
        self.usage = usage


def pipeline_signature(model_id: str) -> str:
    payload = (
        LLM_SCHEMA_VERSION
        + "\n"
        + SENTENCE_CATALOG_VERSION
        + "\n"
        + model_id
        + "\n"
        + FEATURE_RUBRIC
        + "\n"
        + json.dumps(FOMC_SELECTION_JSON_SCHEMA, sort_keys=True)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def inference_policy_payload(config: ExtractionConfig) -> dict:
    return {
        "version": config.inference_policy_version,
        "model_id": config.model_id,
        "reasoning_effort": config.reasoning_effort,
        "primary_output_tokens": config.primary_output_tokens,
        "retry_output_tokens": config.retry_output_tokens,
        "maximum_attempts": 2,
        "content_filter_retry_cap": config.primary_output_tokens,
        "max_output_tokens_retry_cap": config.retry_output_tokens,
        "store": False,
    }


def inference_policy_signature(config: ExtractionConfig) -> str:
    return hashlib.sha256(
        json.dumps(inference_policy_payload(config), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _refusal(response) -> str | None:
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", None) == "refusal":
                return content.refusal
    return None


def compare_minutes_once(
    client: OpenAI,
    previous,
    current,
    config: ExtractionConfig,
    max_output_tokens: int,
):
    previous_text = Path(previous.text_path).read_text(encoding="utf-8")
    current_text = Path(current.text_path).read_text(encoding="utf-8")
    previous_catalog = build_sentence_catalog(previous_text, "P")
    current_catalog = build_sentence_catalog(current_text, "C")
    response = client.responses.create(
        model=config.model_id,
        reasoning={"effort": config.reasoning_effort},
        max_output_tokens=max_output_tokens,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a monetary-policy research extractor. Compare two official FOMC "
                    "minutes sentence catalogs under the fixed rubric. Select only supplied "
                    "sentence IDs and return the required structured object.\n\n" + FEATURE_RUBRIC
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PREVIOUS MEETING DATE: {pd.Timestamp(previous.meeting_date).date()}\n"
                    f"<previous_sentence_catalog>\n{format_sentence_catalog(previous_catalog)}\n"
                    f"</previous_sentence_catalog>\n\n"
                    f"CURRENT MEETING DATE: {pd.Timestamp(current.meeting_date).date()}\n"
                    f"<current_sentence_catalog>\n{format_sentence_catalog(current_catalog)}\n"
                    f"</current_sentence_catalog>"
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "FomcMinutesSelection",
                "schema": FOMC_SELECTION_JSON_SCHEMA,
                "strict": True,
            }
        },
        store=False,
    )
    usage = response.usage.model_dump() if response.usage is not None else {}
    usage["requested_output_tokens"] = max_output_tokens
    if response.status == "incomplete":
        reason = getattr(response.incomplete_details, "reason", None) or "unknown"
        raise IncompleteComparison(reason, usage, [])
    if response.status != "completed":
        raise RuntimeError(f"Unexpected Responses API status: {response.status}")
    refusal = _refusal(response)
    if refusal:
        raise RuntimeError(f"Luna refused the comparison: {refusal}")
    payload = response.output_text or ""
    try:
        selection = FomcMinutesSelection.model_validate_json(payload)
        comparison = hydrate_comparison(selection, previous_catalog, current_catalog)
    except Exception as exc:
        raise SchemaValidationFailure(str(exc)[:2_000], payload, usage) from exc
    grounding = audit_grounding(comparison, previous_text, current_text)
    return comparison, usage, grounding


def compare_minutes_with_retry(client: OpenAI, previous, current, config: ExtractionConfig):
    attempts: list[dict] = []
    requested = config.primary_output_tokens
    for attempt_number in range(2):
        try:
            comparison, usage, grounding = compare_minutes_once(
                client, previous, current, config, requested
            )
            usage["incomplete_attempts"] = attempts
            return comparison, usage, grounding
        except IncompleteComparison as exc:
            attempts.append(
                {
                    "reason": exc.reason,
                    "usage": exc.usage,
                    "requested_output_tokens": requested,
                }
            )
            if attempt_number == 1:
                raise IncompleteComparison(exc.reason, exc.usage, attempts) from exc
            if exc.reason == "max_output_tokens":
                requested = config.retry_output_tokens
            elif exc.reason == "content_filter":
                requested = config.primary_output_tokens
            else:
                raise IncompleteComparison(exc.reason, exc.usage, attempts) from exc
    raise AssertionError("Retry loop exited unexpectedly.")
