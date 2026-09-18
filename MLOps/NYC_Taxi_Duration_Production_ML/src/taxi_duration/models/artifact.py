from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

FEATURE_CONTRACT_VERSION = "1.2.0"


class ArtifactCompatibilityError(RuntimeError):
    """The serialized model was produced by a different library runtime than the loader."""


def runtime_versions() -> dict[str, str]:
    """Libraries whose version determines whether a joblib/pickle artifact loads correctly."""
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def check_runtime_compatibility(metadata: dict[str, Any]) -> None:
    """Refuse to load an artifact whose scikit-learn/numpy versions differ from this runtime.

    Unpickling across scikit-learn versions is unsupported and can fail or, worse, load and
    predict differently. Python patch versions are allowed to differ.
    """
    recorded = metadata.get("runtime_versions")
    if not isinstance(recorded, dict):
        raise ArtifactCompatibilityError("artifact metadata has no runtime_versions")
    current = runtime_versions()
    mismatches = [
        f"{name}: artifact={recorded.get(name)} runtime={current[name]}"
        for name in ("scikit-learn", "numpy", "pandas", "joblib")
        if recorded.get(name) != current[name]
    ]
    if recorded.get("python", "").rsplit(".", 1)[0] != current["python"].rsplit(".", 1)[0]:
        mismatches.append(f"python: artifact={recorded.get('python')} runtime={current['python']}")
    if mismatches:
        raise ArtifactCompatibilityError("incompatible runtime: " + "; ".join(mismatches))


@dataclass(frozen=True)
class ModelMetadata:
    model_version: str
    created_at_utc: str
    git_sha: str
    data_sha256: str
    config_sha256: str
    metrics: dict[str, float]
    runtime_versions: dict[str, str]
    feature_contract_version: str = FEATURE_CONTRACT_VERSION


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_artifact(
    model: Pipeline,
    metadata: ModelMetadata,
    model_path: Path,
    metadata_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(".joblib.tmp")
    joblib.dump(model, temporary)
    temporary.replace(model_path)
    metadata_path.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True), encoding="utf-8"
    )


def load_artifact(model_path: Path, metadata_path: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Load a trusted artifact after verifying it matches the current library runtime."""
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    check_runtime_compatibility(metadata)
    model = joblib.load(model_path)
    return model, metadata


def new_metadata(
    model_version: str,
    git_sha: str,
    data_sha256: str,
    config_sha256: str,
    metrics: dict[str, float],
) -> ModelMetadata:
    return ModelMetadata(
        model_version=model_version,
        created_at_utc=datetime.now(UTC).isoformat(),
        git_sha=git_sha,
        data_sha256=data_sha256,
        config_sha256=config_sha256,
        metrics=metrics,
        runtime_versions=runtime_versions(),
    )
