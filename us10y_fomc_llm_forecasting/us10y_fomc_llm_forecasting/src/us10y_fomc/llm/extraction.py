from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import ExtractionConfig, RuntimeConfig
from ..progress import progress
from .cost_control import CostLedger, TokenPrices
from .extraction_cache import append_jsonl, latest_by_current_date, load_jsonl
from .feature_contract import FOMC_FEATURE_NAMES, GROUNDING_AUDIT_VERSION
from .grounding import audit_grounding, derive_model_safe_features, grounding_qualifies
from .luna_client import (
    IncompleteComparison,
    SchemaValidationFailure,
    compare_minutes_with_retry,
    inference_policy_signature,
    pipeline_signature,
)
from .schemas import FomcMinutesComparison
from .sentence_catalog import build_sentence_catalog, format_sentence_catalog


@dataclass(frozen=True)
class ExtractionPaths:
    production_cache: Path
    quarantine: Path
    cost_ledger: Path
    token_budget: Path


@dataclass(frozen=True)
class ExtractionResult:
    cached_records: dict[str, dict]
    required_pairs: int
    missing_dates: tuple[str, ...]
    ready: bool
    spent_usd: float
    token_budget: pd.DataFrame


def build_pair_token_budget(documents: pd.DataFrame, model_id: str) -> pd.DataFrame:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is required for the paid-run token budget. Install project requirements."
        ) from exc
    try:
        tokenizer = tiktoken.encoding_for_model(model_id)
    except KeyError:
        tokenizer = tiktoken.get_encoding("o200k_base")
    rows = []
    pairs = zip(
        documents.iloc[:-1].itertuples(index=False),
        documents.iloc[1:].itertuples(index=False),
    )
    for position, (previous, current) in enumerate(
        progress(
            pairs,
            desc="Luna token budget",
            total=max(len(documents) - 1, 0),
            unit="pair",
            leave=False,
        )
    ):
        previous_text = Path(previous.text_path).read_text(encoding="utf-8")
        current_text = Path(current.text_path).read_text(encoding="utf-8")
        previous_catalog = build_sentence_catalog(previous_text, "P")
        current_catalog = build_sentence_catalog(current_text, "C")
        tokens = len(tokenizer.encode(format_sentence_catalog(previous_catalog))) + len(
            tokenizer.encode(format_sentence_catalog(current_catalog))
        )
        rows.append(
            {
                "pair_position": position,
                "previous_meeting_date": pd.Timestamp(previous.meeting_date),
                "current_meeting_date": pd.Timestamp(current.meeting_date),
                "previous_sentence_count": len(previous_catalog),
                "current_sentence_count": len(current_catalog),
                "catalog_input_tokens": tokens,
                "budgeted_input_tokens": tokens + 2_500,
            }
        )
    return pd.DataFrame(rows)


def _document_lookup(documents: pd.DataFrame) -> dict[str, object]:
    return {
        pd.Timestamp(row.meeting_date).strftime("%Y-%m-%d"): row
        for row in documents.itertuples(index=False)
    }


def _record_matches(
    record: dict,
    documents_by_date: dict[str, object],
    extraction_config: ExtractionConfig,
) -> bool:
    previous = documents_by_date.get(record.get("previous_meeting_date"))
    current = documents_by_date.get(record.get("current_meeting_date"))
    return bool(
        previous is not None
        and current is not None
        and record.get("previous_sha256") == previous.text_sha256
        and record.get("current_sha256") == current.text_sha256
        and record.get("llm_model") == extraction_config.model_id
        and record.get("pipeline_signature") == pipeline_signature(extraction_config.model_id)
        and record.get("inference_policy_signature")
        == inference_policy_signature(extraction_config)
    )


def _reaudit_record(
    record: dict,
    documents_by_date: dict[str, object],
    extraction_config: ExtractionConfig,
) -> dict:
    if not _record_matches(record, documents_by_date, extraction_config):
        raise ValueError("The record does not match current documents and extraction signatures.")
    comparison = FomcMinutesComparison.model_validate(record["comparison"])
    previous = documents_by_date[record["previous_meeting_date"]]
    current = documents_by_date[record["current_meeting_date"]]
    grounding = audit_grounding(
        comparison,
        Path(previous.text_path).read_text(encoding="utf-8"),
        Path(current.text_path).read_text(encoding="utf-8"),
    )
    return {
        **record,
        "schema_validated": True,
        "grounding_audit_version": GROUNDING_AUDIT_VERSION,
        "grounding": grounding,
        "derived_features": derive_model_safe_features(comparison, grounding),
    }


def load_qualified_records(
    paths: tuple[Path, ...],
    documents: pd.DataFrame,
    config: ExtractionConfig,
) -> dict[str, dict]:
    documents_by_date = _document_lookup(documents)
    accepted: dict[str, dict] = {}
    for path in paths:
        records = load_jsonl(path)
        for record in progress(
            records,
            desc=f"Audit cache {path.name}",
            total=len(records),
            unit="record",
            leave=False,
        ):
            if "comparison" not in record or not _record_matches(record, documents_by_date, config):
                continue
            try:
                audited = _reaudit_record(record, documents_by_date, config)
            except (KeyError, ValueError):
                continue
            if grounding_qualifies(
                audited["grounding"],
                config.minimum_strict_grounding_ratio,
                config.minimum_usable_feature_count,
            ):
                accepted[audited["current_meeting_date"]] = audited
    return accepted


def promote_qualified_records(
    source_paths: tuple[Path, ...],
    production_path: Path,
    documents: pd.DataFrame,
    config: ExtractionConfig,
) -> int:
    production = load_qualified_records((production_path,), documents, config)
    candidates = load_qualified_records(source_paths, documents, config)
    promoted = 0
    for key, record in sorted(candidates.items()):
        if key not in production:
            append_jsonl(production_path, {**record, "promoted_from": "validated_pilot"})
            production[key] = record
            promoted += 1
    return promoted


def extract_minutes_pairs(
    documents: pd.DataFrame,
    api_key: str | None,
    extraction_config: ExtractionConfig,
    runtime_config: RuntimeConfig,
    paths: ExtractionPaths,
    pilot_paths: tuple[Path, ...] = (),
) -> ExtractionResult:
    token_budget = build_pair_token_budget(documents, extraction_config.model_id)
    paths.token_budget.parent.mkdir(parents=True, exist_ok=True)
    token_budget.to_csv(paths.token_budget, index=False)
    if pilot_paths:
        promote_qualified_records(pilot_paths, paths.production_cache, documents, extraction_config)
    documents_by_date = _document_lookup(documents)
    cached = load_qualified_records((paths.production_cache,), documents, extraction_config)
    required = {
        pd.Timestamp(row.current_meeting_date).strftime("%Y-%m-%d")
        for row in token_budget.itertuples(index=False)
    }
    quarantined = latest_by_current_date(
        [
            record
            for record in load_jsonl(paths.quarantine)
            if _record_matches(record, documents_by_date, extraction_config)
        ]
    )
    ledger = CostLedger(
        paths.cost_ledger,
        extraction_config.budget_scope,
        TokenPrices(
            extraction_config.input_usd_per_million_tokens,
            extraction_config.output_usd_per_million_tokens,
        ),
    )

    client = None
    if runtime_config.run_openai_preflight or runtime_config.run_paid_extraction:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI SDK is required for model preflight or paid extraction."
            ) from exc
        if not runtime_config.confirm_api_key or not api_key:
            raise RuntimeError("OpenAI access requires confirm_api_key=true and a valid key.")
        client = OpenAI(api_key=api_key, timeout=600.0, max_retries=3)
        metadata = client.models.retrieve(extraction_config.model_id)
        if metadata.id != extraction_config.model_id:
            raise AssertionError("OpenAI model-access preflight returned the wrong model.")

    if runtime_config.run_paid_extraction:
        if not runtime_config.confirm_paid_extraction:
            raise RuntimeError("Paid extraction requires confirm_paid_extraction=true.")
        if extraction_config.hard_run_budget_usd <= 0:
            raise RuntimeError("Set a positive hard_run_budget_usd before paid extraction.")
        if client is None:
            raise RuntimeError("Paid extraction requires a validated OpenAI client.")
        new_calls = 0
        pairs = zip(
            documents.iloc[:-1].itertuples(index=False),
            documents.iloc[1:].itertuples(index=False),
        )
        for position, (previous, current) in enumerate(
            progress(
                pairs,
                total=len(documents) - 1,
                desc="Luna minutes extraction",
                unit="pair",
            )
        ):
            key = pd.Timestamp(current.meeting_date).strftime("%Y-%m-%d")
            if key in cached:
                continue
            if key in quarantined:
                if not runtime_config.retry_quarantined_pairs:
                    continue
                if not runtime_config.confirm_quarantine_retry:
                    raise RuntimeError("Quarantine retry requires explicit confirmation.")
            if runtime_config.max_new_pairs > 0 and new_calls >= runtime_config.max_new_pairs:
                break
            reserve = ledger.reserve(
                key,
                int(token_budget.loc[position, "budgeted_input_tokens"]),
                extraction_config.primary_output_tokens,
                extraction_config.retry_output_tokens,
                extraction_config.hard_run_budget_usd,
            )
            base = {
                "previous_meeting_date": pd.Timestamp(previous.meeting_date).strftime("%Y-%m-%d"),
                "current_meeting_date": key,
                "previous_sha256": previous.text_sha256,
                "current_sha256": current.text_sha256,
                "llm_model": extraction_config.model_id,
                "pipeline_signature": pipeline_signature(extraction_config.model_id),
                "inference_policy_version": extraction_config.inference_policy_version,
                "inference_policy_signature": inference_policy_signature(extraction_config),
            }
            started = time.time()
            try:
                comparison, usage, grounding = compare_minutes_with_retry(
                    client, previous, current, extraction_config
                )
                record = {
                    **base,
                    "duration_seconds": round(time.time() - started, 3),
                    "schema_validated": True,
                    "grounding_audit_version": GROUNDING_AUDIT_VERSION,
                    "usage": usage,
                    "grounding": grounding,
                    "derived_features": derive_model_safe_features(comparison, grounding),
                    "comparison": comparison.model_dump(),
                }
                if grounding_qualifies(
                    grounding,
                    extraction_config.minimum_strict_grounding_ratio,
                    extraction_config.minimum_usable_feature_count,
                ):
                    append_jsonl(paths.production_cache, record)
                    cached[key] = record
                    status = "accepted"
                else:
                    append_jsonl(
                        paths.quarantine,
                        {**record, "failure": "grounding_or_coverage_below_threshold"},
                    )
                    quarantined[key] = record
                    status = "quarantined_grounding"
            except IncompleteComparison as exc:
                usage = {"incomplete_attempts": exc.attempts}
                record = {
                    **base,
                    "failure": "incomplete_response",
                    "failure_reason": exc.reason,
                    "attempts": exc.attempts,
                }
                append_jsonl(paths.quarantine, record)
                quarantined[key] = record
                status = "quarantined_incomplete"
            except SchemaValidationFailure as exc:
                usage = exc.usage
                record = {
                    **base,
                    "failure": "schema_validation_error",
                    "schema_error": exc.error,
                    "raw_output_sha256": exc.raw_output_sha256,
                    "raw_output_excerpt": exc.raw_output_excerpt,
                    "usage": usage,
                }
                append_jsonl(paths.quarantine, record)
                quarantined[key] = record
                status = "quarantined_schema"
            ledger.reconcile(key, reserve, usage, status)
            new_calls += 1

    cached = load_qualified_records((paths.production_cache,), documents, extraction_config)
    missing = tuple(sorted(required - set(cached)))
    return ExtractionResult(
        cached_records=cached,
        required_pairs=len(required),
        missing_dates=missing,
        ready=not missing and len(cached) == len(required),
        spent_usd=ledger.spent(),
        token_budget=token_budget,
    )


def build_grounded_event_table(
    extraction_result: ExtractionResult,
    rates: pd.DataFrame,
) -> pd.DataFrame:
    if not extraction_result.ready:
        raise RuntimeError("The grounded production cache is incomplete.")
    rows = []
    cached_records = sorted(extraction_result.cached_records.items())
    for key, record in progress(
        cached_records,
        desc="Grounded event table",
        total=len(cached_records),
        unit="meeting",
        leave=False,
    ):
        row: dict[str, object] = {
            "meeting_date": pd.Timestamp(key),
            "previous_meeting_date": pd.Timestamp(record["previous_meeting_date"]),
            "llm_model": record["llm_model"],
            "oracle_minutes_backdated_to_meeting": True,
        }
        for feature in FOMC_FEATURE_NAMES:
            safe = record["derived_features"][feature]
            row[feature] = float(safe["score"])
            row[f"{feature}__confidence"] = float(safe["confidence"])
            row[f"{feature}__available"] = float(safe["model_mask"])
            row[f"{feature}__source_available"] = float(safe["is_available"])
            row[f"{feature}__grounded"] = safe["is_grounded"]
        rows.append(row)
    events = pd.DataFrame(rows).merge(
        rates[["meeting_date", "actual_rate_change_bp", "fred_target_midpoint"]],
        on="meeting_date",
        how="left",
        validate="one_to_one",
    )
    if events[["actual_rate_change_bp", "fred_target_midpoint"]].isna().any().any():
        raise RuntimeError("A grounded event lacks authoritative FRED rate facts.")
    return events.sort_values("meeting_date").reset_index(drop=True)
