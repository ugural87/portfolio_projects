from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .feature_contract import FOMC_FEATURE_NAMES, PAIR_SUMMARY_MAX_CHARS, RATIONALE_MAX_CHARS
from .sentence_catalog import SENTENCE_ID_PATTERN


class FeatureSelectionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float
    confidence: float = Field(ge=0.0, le=1.0)
    is_available: bool
    previous_evidence_id: str = Field(max_length=6)
    current_evidence_id: str = Field(max_length=6)
    rationale: str = Field(max_length=RATIONALE_MAX_CHARS)


class DirectionalSelectionAssessment(FeatureSelectionAssessment):
    score: float = Field(ge=-1.0, le=1.0)


class BoundedSelectionAssessment(FeatureSelectionAssessment):
    score: float = Field(ge=0.0, le=1.0)


class FomcMinutesSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_policy_stance: DirectionalSelectionAssessment
    policy_stance_shift: DirectionalSelectionAssessment
    expected_rate_path_shift: DirectionalSelectionAssessment
    inflation_assessment_shift: DirectionalSelectionAssessment
    inflation_risk_shift: DirectionalSelectionAssessment
    unemployment_assessment_shift: DirectionalSelectionAssessment
    labor_market_risk_shift: DirectionalSelectionAssessment
    growth_activity_shift: DirectionalSelectionAssessment
    financial_conditions_shift: DirectionalSelectionAssessment
    balance_sheet_policy_shift: DirectionalSelectionAssessment
    risk_balance_shift: DirectionalSelectionAssessment
    committee_dispersion: BoundedSelectionAssessment
    policy_uncertainty: BoundedSelectionAssessment
    forward_guidance_strength: BoundedSelectionAssessment
    pair_summary: str = Field(max_length=PAIR_SUMMARY_MAX_CHARS)

    @model_validator(mode="after")
    def validate_selection_contract(self):
        for name in FOMC_FEATURE_NAMES:
            assessment = getattr(self, name)
            if not assessment.is_available:
                if assessment.score != 0 or assessment.confidence != 0:
                    raise ValueError(f"Unavailable {name} must have zero score and confidence")
                if assessment.previous_evidence_id or assessment.current_evidence_id:
                    raise ValueError(f"Unavailable {name} must have empty evidence IDs")
                continue
            if not SENTENCE_ID_PATTERN.fullmatch(assessment.current_evidence_id):
                raise ValueError(f"{name} has an invalid current evidence ID")
            if not assessment.current_evidence_id.startswith("C"):
                raise ValueError(f"{name} current evidence must reference the C catalog")
            if name == "current_policy_stance":
                if assessment.previous_evidence_id:
                    raise ValueError("current_policy_stance must not use previous evidence")
            else:
                if not SENTENCE_ID_PATTERN.fullmatch(assessment.previous_evidence_id):
                    raise ValueError(f"{name} has an invalid previous evidence ID")
                if not assessment.previous_evidence_id.startswith("P"):
                    raise ValueError(f"{name} previous evidence must reference the P catalog")
        return self


class FeatureAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float
    confidence: float = Field(ge=0.0, le=1.0)
    is_available: bool
    previous_evidence_id: str
    current_evidence_id: str
    previous_evidence: str
    current_evidence: str
    rationale: str = Field(max_length=RATIONALE_MAX_CHARS)


class DirectionalAssessment(FeatureAssessment):
    score: float = Field(ge=-1.0, le=1.0)


class BoundedAssessment(FeatureAssessment):
    score: float = Field(ge=0.0, le=1.0)


class FomcMinutesComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_policy_stance: DirectionalAssessment
    policy_stance_shift: DirectionalAssessment
    expected_rate_path_shift: DirectionalAssessment
    inflation_assessment_shift: DirectionalAssessment
    inflation_risk_shift: DirectionalAssessment
    unemployment_assessment_shift: DirectionalAssessment
    labor_market_risk_shift: DirectionalAssessment
    growth_activity_shift: DirectionalAssessment
    financial_conditions_shift: DirectionalAssessment
    balance_sheet_policy_shift: DirectionalAssessment
    risk_balance_shift: DirectionalAssessment
    committee_dispersion: BoundedAssessment
    policy_uncertainty: BoundedAssessment
    forward_guidance_strength: BoundedAssessment
    pair_summary: str = Field(max_length=PAIR_SUMMARY_MAX_CHARS)

    @model_validator(mode="after")
    def validate_hydrated_contract(self):
        for name in FOMC_FEATURE_NAMES:
            assessment = getattr(self, name)
            if not assessment.is_available:
                if assessment.score != 0 or assessment.confidence != 0:
                    raise ValueError(f"Unavailable {name} must have zero score and confidence")
                if any(
                    (
                        assessment.previous_evidence_id,
                        assessment.current_evidence_id,
                        assessment.previous_evidence,
                        assessment.current_evidence,
                    )
                ):
                    raise ValueError(f"Unavailable {name} must have empty evidence fields")
                continue
            if not assessment.current_evidence_id or not assessment.current_evidence:
                raise ValueError(f"{name} lacks hydrated current evidence")
            if name == "current_policy_stance":
                if assessment.previous_evidence_id or assessment.previous_evidence:
                    raise ValueError("current_policy_stance must not use previous evidence")
            elif not assessment.previous_evidence_id or not assessment.previous_evidence:
                raise ValueError(f"{name} lacks hydrated previous evidence")
        return self


FOMC_SELECTION_JSON_SCHEMA = FomcMinutesSelection.model_json_schema()

