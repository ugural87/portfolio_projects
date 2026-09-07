from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import ProjectConfig, ProjectPaths
from ..data.fomc_integrity import audit_fomc_documents
from ..data.policy_rates import build_policy_rate_audit
from ..llm.extraction import ExtractionPaths, build_pair_token_budget, extract_minutes_pairs


def run_local_preflight(
    documents: pd.DataFrame,
    manifest: pd.DataFrame,
    excluded: pd.DataFrame,
    fred_key: str,
    openai_key: str | None,
    config: ProjectConfig,
    paths: ProjectPaths,
    pilot_paths: tuple[Path, ...] = (),
) -> tuple[pd.DataFrame, dict[str, object]]:
    coverage, document_checks = audit_fomc_documents(
        manifest, excluded, documents, config.extraction.start_year
    )
    rates, disagreements = build_policy_rate_audit(documents, fred_key)
    token_budget = build_pair_token_budget(documents, config.extraction.model_id)
    extraction = extract_minutes_pairs(
        documents,
        openai_key,
        config.extraction,
        config.runtime,
        ExtractionPaths(
            production_cache=paths.extraction_cache / "luna_production.jsonl",
            quarantine=paths.quarantine / "luna_quarantine.jsonl",
            cost_ledger=paths.audit_reports / "luna_cost_ledger.jsonl",
            token_budget=paths.audit_reports / "luna_token_budget.csv",
        ),
        pilot_paths=pilot_paths,
    )
    checks = {
        "official_documents_local": len(documents) > 0,
        "document_integrity": all(document_checks.values()),
        "fred_rate_audit": not rates.empty,
        "token_budget_computed": len(token_budget) == len(documents) - 1,
        "openai_model_access": (
            not config.runtime.run_openai_preflight or openai_key is not None
        ),
        "production_extraction_complete": extraction.ready,
        "missing_extraction_pairs": len(extraction.missing_dates),
        "minutes_parser_disagreements": len(disagreements),
    }
    return rates, {"coverage": coverage, "checks": checks, "extraction": extraction}
