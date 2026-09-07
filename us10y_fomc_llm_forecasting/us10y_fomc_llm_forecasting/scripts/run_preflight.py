from __future__ import annotations

import json

import pandas as pd

from _bootstrap import PROJECT_ROOT

from us10y_fomc.config import ProjectPaths, load_project_config
from us10y_fomc.data.fomc_integrity import audit_fomc_documents
from us10y_fomc.data.treasury_data import load_treasury_panel
from us10y_fomc.llm.extraction import ExtractionPaths, extract_minutes_pairs
from us10y_fomc.security.credentials import credential_fingerprint, load_api_key
from us10y_fomc.workflow import load_official_documents


if __name__ == "__main__":
    config = load_project_config(PROJECT_ROOT)
    paths = ProjectPaths(PROJECT_ROOT).ensure()
    fred_key = load_api_key("FRED_API_KEY", PROJECT_ROOT)
    openai_key = load_api_key(
        "OPENAI_API_KEY",
        PROJECT_ROOT,
        required=config.runtime.run_openai_preflight or config.runtime.run_paid_extraction,
    )
    panel = load_treasury_panel(paths.data / "raw")
    documents = load_official_documents(paths)
    rates = pd.read_csv(
        paths.data / "processed" / "fomc_policy_rates.csv",
        parse_dates=["meeting_date", "rate_effective_date"],
    )
    manifest = pd.read_csv(
        paths.manifests / "fomc_manifest.csv",
        parse_dates=["meeting_date", "meeting_start_date", "canonical_minutes_date"],
    )
    excluded_path = paths.manifests / "fomc_excluded.csv"
    excluded = pd.read_csv(excluded_path, parse_dates=["meeting_date"]) if excluded_path.stat().st_size else pd.DataFrame()
    _, integrity = audit_fomc_documents(
        manifest, excluded, documents, config.extraction.start_year
    )
    result = extract_minutes_pairs(
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
    )
    rate_lookup = rates.set_index("meeting_date").actual_rate_change_bp
    rate_anchors = {
        pd.Timestamp("2008-12-16"): -87.5,
        pd.Timestamp("2020-04-29"): 0.0,
        pd.Timestamp("2022-06-15"): 75.0,
    }
    fred_rate_audit = all(
        date in rate_lookup.index
        and abs(float(rate_lookup.loc[date]) - expected) <= 1e-6
        for date, expected in rate_anchors.items()
    )
    report = {
        "treasury_panel": bool(len(panel) and panel.index.is_unique),
        "document_integrity": all(integrity.values()),
        "fred_rate_audit": fred_rate_audit,
        "token_budget": len(result.token_budget) == len(documents) - 1,
        "openai_model_access": (
            not config.runtime.run_openai_preflight or openai_key is not None
        ),
        "production_extraction_complete": result.ready,
        "missing_pairs": len(result.missing_dates),
        "fred_key_fingerprint": credential_fingerprint(fred_key),
        "openai_key_fingerprint": (
            credential_fingerprint(openai_key) if openai_key else None
        ),
        "safe_paid_switches": (
            not config.runtime.run_paid_extraction
            or (
                config.runtime.confirm_paid_extraction
                and config.extraction.hard_run_budget_usd > 0
            )
        ),
    }
    report["go_for_training"] = all(
        report[key]
        for key in (
            "treasury_panel",
            "document_integrity",
            "fred_rate_audit",
            "token_budget",
            "production_extraction_complete",
            "safe_paid_switches",
        )
    )
    output = paths.audit_reports / "preflight_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
