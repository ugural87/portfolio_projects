from __future__ import annotations

from .feature_contract import FOMC_FEATURE_NAMES, GROUNDING_AUDIT_VERSION
from .schemas import FomcMinutesComparison, FomcMinutesSelection
from .sentence_catalog import build_sentence_catalog, resolve_evidence_id


def hydrate_comparison(
    selection: FomcMinutesSelection,
    previous_catalog: dict[str, str],
    current_catalog: dict[str, str],
) -> FomcMinutesComparison:
    payload: dict[str, object] = {"pair_summary": selection.pair_summary}
    for feature in FOMC_FEATURE_NAMES:
        assessment = getattr(selection, feature)
        values = assessment.model_dump()
        if assessment.is_available:
            previous_evidence = (
                ""
                if feature == "current_policy_stance"
                else resolve_evidence_id(assessment.previous_evidence_id, previous_catalog, "P")
            )
            current_evidence = resolve_evidence_id(
                assessment.current_evidence_id, current_catalog, "C"
            )
        else:
            previous_evidence = ""
            current_evidence = ""
        payload[feature] = {
            **values,
            "previous_evidence": previous_evidence,
            "current_evidence": current_evidence,
        }
    return FomcMinutesComparison.model_validate(payload)


def audit_grounding(
    comparison: FomcMinutesComparison,
    previous_text: str,
    current_text: str,
) -> dict[str, object]:
    previous_catalog = build_sentence_catalog(previous_text, "P")
    current_catalog = build_sentence_catalog(current_text, "C")
    checks: list[dict[str, object]] = []
    feature_mask: dict[str, bool] = {}
    available = 0
    passed_checks = 0
    total_checks = 0
    for feature in FOMC_FEATURE_NAMES:
        assessment = getattr(comparison, feature)
        if not assessment.is_available:
            feature_mask[feature] = False
            checks.append({"feature": feature, "available": False, "usable": False})
            continue
        available += 1
        previous_required = feature != "current_policy_stance"
        current_grounded = bool(
            assessment.current_evidence_id in current_catalog
            and current_catalog[assessment.current_evidence_id] == assessment.current_evidence
        )
        previous_grounded = bool(
            not previous_required
            or (
                assessment.previous_evidence_id in previous_catalog
                and previous_catalog[assessment.previous_evidence_id] == assessment.previous_evidence
            )
        )
        total_checks += 1 + int(previous_required)
        passed_checks += int(current_grounded) + int(previous_required and previous_grounded)
        usable = current_grounded and previous_grounded
        feature_mask[feature] = usable
        checks.append(
            {
                "feature": feature,
                "available": True,
                "previous_required": previous_required,
                "previous_grounded": previous_grounded,
                "current_grounded": current_grounded,
                "usable": usable,
            }
        )
    usable_count = sum(feature_mask.values())
    ratio = passed_checks / total_checks if total_checks else 0.0
    return {
        "audit_version": GROUNDING_AUDIT_VERSION,
        "strict_grounding_ratio": ratio,
        "validated_grounding_ratio": ratio,
        "grounding_ratio": ratio,
        "available_feature_ratio": available / len(FOMC_FEATURE_NAMES),
        "usable_feature_ratio": usable_count / len(FOMC_FEATURE_NAMES),
        "available_features": available,
        "usable_features": usable_count,
        "total_features": len(FOMC_FEATURE_NAMES),
        "total_checks": total_checks,
        "strict_passed_checks": passed_checks,
        "validated_passed_checks": passed_checks,
        "feature_usable_mask": feature_mask,
        "checks": checks,
    }


def derive_model_safe_features(
    comparison: FomcMinutesComparison, grounding: dict[str, object]
) -> dict[str, dict[str, object]]:
    mask = grounding.get("feature_usable_mask") or {}
    safe: dict[str, dict[str, object]] = {}
    for feature in FOMC_FEATURE_NAMES:
        assessment = getattr(comparison, feature)
        available = bool(assessment.is_available)
        grounded = bool(mask.get(feature, False)) if available else None
        usable = bool(available and grounded)
        safe[feature] = {
            "score": float(assessment.score) if usable else 0.0,
            "confidence": float(assessment.confidence) if usable else 0.0,
            "is_available": available,
            "is_grounded": grounded,
            "model_mask": 1.0 if usable else 0.0,
            "status": "USABLE" if usable else ("MASKED" if available else "NOT_APPLICABLE"),
        }
    return safe


def grounding_qualifies(
    grounding: dict[str, object], minimum_ratio: float, minimum_usable: int
) -> bool:
    return bool(
        float(grounding.get("strict_grounding_ratio", 0.0)) >= minimum_ratio
        and int(grounding.get("usable_features", 0)) >= minimum_usable
    )

