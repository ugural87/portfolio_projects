from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import json

import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import make_features, target
from .metrics import metric_bundle


@dataclass(frozen=True)
class Candidate:
    family: str
    strategy: str
    params: dict

    @property
    def name(self) -> str:
        return f"{self.family}__{self.strategy}"


def candidate_grid(fast_mode: bool = True) -> list[Candidate]:
    candidates: list[Candidate] = []
    for params in ParameterGrid({"C": [0.1, 1.0]}):
        candidates.append(Candidate("logistic", "unweighted", params))
    for params in ParameterGrid({"C": [0.05, 0.2, 1.0]}):
        candidates.append(Candidate("logistic", "class_weight", params))
    for params in ParameterGrid({"C": [0.1, 1.0], "sampling_strategy": [0.01, 0.03]}):
        candidates.append(Candidate("logistic", "smote", params))
    xgb_grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.04],
        "n_estimators": [280 if fast_mode else 500],
    }
    for params in ParameterGrid(xgb_grid):
        candidates.append(Candidate("xgboost", "scale_pos_weight", params))
    lgb_grid = {
        "num_leaves": [15, 31],
        "min_child_samples": [40],
        "n_estimators": [280 if fast_mode else 500],
    }
    for params in ParameterGrid(lgb_grid):
        candidates.append(Candidate("lightgbm", "scale_pos_weight", params))
    for params in ParameterGrid({"max_depth": [10, None], "n_estimators": [220 if fast_mode else 400]}):
        candidates.append(Candidate("balanced_random_forest", "balanced_bootstrap", params))
    candidates.append(Candidate("easy_ensemble", "random_under_sampling", {"n_estimators": 8 if fast_mode else 16}))
    return candidates


def build_estimator(candidate: Candidate, y_train, random_state: int = 42):
    params = candidate.params
    if candidate.family == "logistic":
        model = LogisticRegression(
            C=float(params["C"]),
            solver="lbfgs",
            max_iter=2_000,
            class_weight="balanced" if candidate.strategy == "class_weight" else None,
            random_state=random_state,
        )
        steps = [("scaler", StandardScaler())]
        if candidate.strategy == "smote":
            steps.append(("smote", SMOTE(
                sampling_strategy=float(params["sampling_strategy"]),
                k_neighbors=5,
                random_state=random_state,
            )))
        steps.append(("model", model))
        return ImbPipeline(steps)

    positive_weight = float((np.asarray(y_train) == 0).sum() / (np.asarray(y_train) == 1).sum())
    if candidate.family == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            scale_pos_weight=positive_weight,
            random_state=random_state,
            n_jobs=-1,
        )
    if candidate.family == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            n_estimators=int(params["n_estimators"]),
            learning_rate=0.04,
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            scale_pos_weight=positive_weight,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    if candidate.family == "balanced_random_forest":
        return BalancedRandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            sampling_strategy="all",
            replacement=True,
            bootstrap=False,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    if candidate.family == "easy_ensemble":
        return EasyEnsembleClassifier(
            n_estimators=int(params["n_estimators"]),
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown candidate family: {candidate.family}")


def temporal_cv(candidate: Candidate, X, y, random_state=42, n_splits=3) -> dict:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    started = perf_counter()
    for train_idx, valid_idx in splitter.split(X):
        if np.unique(y[train_idx]).size < 2 or np.unique(y[valid_idx]).size < 2:
            continue
        estimator = build_estimator(candidate, y[train_idx], random_state)
        estimator.fit(X.iloc[train_idx], y[train_idx])
        probability = estimator.predict_proba(X.iloc[valid_idx])[:, 1]
        scores.append(metric_bundle(y[valid_idx], probability))
    if not scores:
        raise RuntimeError(f"No valid temporal folds for {candidate.name}")
    return {
        "candidate": candidate.name,
        "family": candidate.family,
        "strategy": candidate.strategy,
        "params": json.dumps(candidate.params, sort_keys=True),
        "cv_pr_auc_mean": float(np.mean([s["pr_auc"] for s in scores])),
        "cv_pr_auc_std": float(np.std([s["pr_auc"] for s in scores])),
        "cv_roc_auc_mean": float(np.mean([s["roc_auc"] for s in scores])),
        "cv_brier_mean": float(np.mean([s["brier"] for s in scores])),
        "cv_folds": len(scores),
        "fit_seconds": float(perf_counter() - started),
    }


def train_and_select(parts, artifact_dir: str | Path, random_state=42, fast_mode=True):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    train_df = parts["train"]
    valid_df = parts["validation"]
    X_train, y_train = make_features(train_df), target(train_df)
    X_valid, y_valid = make_features(valid_df), target(valid_df)

    cv_rows = []
    best_by_name: dict[str, tuple[Candidate, float]] = {}
    for candidate in candidate_grid(fast_mode):
        row = temporal_cv(candidate, X_train, y_train, random_state=random_state)
        cv_rows.append(row)
        current = best_by_name.get(candidate.name)
        if current is None or row["cv_pr_auc_mean"] > current[1]:
            best_by_name[candidate.name] = (candidate, row["cv_pr_auc_mean"])

    cv_results = pd.DataFrame(cv_rows).sort_values("cv_pr_auc_mean", ascending=False)
    cv_results.to_csv(artifact_dir / "temporal_cv_results.csv", index=False)

    validation_rows = []
    fitted = {}
    for name, (candidate, _) in best_by_name.items():
        estimator = build_estimator(candidate, y_train, random_state)
        estimator.fit(X_train, y_train)
        probability = estimator.predict_proba(X_valid)[:, 1]
        metrics = metric_bundle(y_valid, probability)
        validation_rows.append({
            "candidate": name,
            "family": candidate.family,
            "strategy": candidate.strategy,
            "params": json.dumps(candidate.params, sort_keys=True),
            **{f"validation_{k}": v for k, v in metrics.items()},
        })
        fitted[name] = (candidate, estimator)

    validation = pd.DataFrame(validation_rows).sort_values("validation_pr_auc", ascending=False)
    validation.to_csv(artifact_dir / "model_validation_comparison.csv", index=False)
    selected_name = str(validation.iloc[0]["candidate"])
    selected_candidate = fitted[selected_name][0]

    development = pd.concat([train_df, valid_df], ignore_index=True)
    X_development, y_development = make_features(development), target(development)
    selected_model = build_estimator(selected_candidate, y_development, random_state)
    selected_model.fit(X_development, y_development)
    joblib.dump(selected_model, artifact_dir / "selected_base_model.joblib")
    metadata = {
        "selected_candidate": selected_name,
        "family": selected_candidate.family,
        "strategy": selected_candidate.strategy,
        "params": selected_candidate.params,
        "selection_metric": "validation_pr_auc",
        "feature_names": X_development.columns.tolist(),
    }
    (artifact_dir / "selected_model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return selected_model, metadata, cv_results, validation


class PlattCalibrator:
    def __init__(self, random_state=42):
        self.model = LogisticRegression(C=1e6, solver="lbfgs", random_state=random_state)

    @staticmethod
    def _logit(probability):
        p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, probability, y):
        self.model.fit(self._logit(probability), np.asarray(y, dtype=int))
        return self

    def predict(self, probability):
        return self.model.predict_proba(self._logit(probability))[:, 1]


def calibrate_selected_model(model, parts, artifact_dir: str | Path, random_state=42):
    artifact_dir = Path(artifact_dir)
    calibration_df = parts["calibration"]
    X_cal, y_cal = make_features(calibration_df), target(calibration_df)
    raw = model.predict_proba(X_cal)[:, 1]
    calibrator = PlattCalibrator(random_state).fit(raw, y_cal)
    calibrated = calibrator.predict(raw)
    comparison = pd.DataFrame([
        {"version": "raw", **metric_bundle(y_cal, raw)},
        {"version": "platt", **metric_bundle(y_cal, calibrated)},
    ])
    comparison.to_csv(artifact_dir / "calibration_fit_summary.csv", index=False)
    joblib.dump(calibrator, artifact_dir / "platt_calibrator.joblib")
    return calibrator, comparison


def predict_calibrated(model, calibrator: PlattCalibrator, frame: pd.DataFrame):
    raw = model.predict_proba(make_features(frame))[:, 1]
    return calibrator.predict(raw)

