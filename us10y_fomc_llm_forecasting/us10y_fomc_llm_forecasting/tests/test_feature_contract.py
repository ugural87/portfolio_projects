from pathlib import Path

from us10y_fomc.config import load_project_config
from us10y_fomc.llm.feature_contract import FOMC_FEATURE_NAMES
from us10y_fomc.llm.luna_client import inference_policy_signature, pipeline_signature


ROOT = Path(__file__).resolve().parents[1]


def test_feature_contract_and_validated_signatures() -> None:
    config = load_project_config(ROOT)
    assert len(FOMC_FEATURE_NAMES) == 14
    assert len(set(FOMC_FEATURE_NAMES)) == 14
    assert pipeline_signature(config.extraction.model_id) == (
        "89fd90f801f26cc153ef51d662f06ae9c7414301cdf98bfa800735fc76b2b412"
    )
    assert inference_policy_signature(config.extraction) == (
        "8586acf7347ddd07a5b65626300c0ef72685d5e54a8e5b996b005810ea70fecd"
    )
