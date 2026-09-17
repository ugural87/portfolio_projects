from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection.causal import run_causal_demo
from fraud_detection.config import ProjectConfig, RunConfig
from fraud_detection.data import audit_to_dict, load_dataset
from fraud_detection.decision import run_decision_analysis
from fraud_detection.deep import refit_deep_candidate, run_deep_challenger
from fraud_detection.modeling import calibrate_selected_model, train_and_select
from fraud_detection.metrics import bootstrap_metric_interval, metric_bundle
from fraud_detection.reporting import (
    build_results_markdown,
    causal_figures,
    dashboard,
    decision_figures,
    eda_figures,
    model_figures,
)
from fraud_detection.splitting import split_summary, temporal_partitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.json"))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--full", action="store_true", help="Use slower full settings")
    args = parser.parse_args()
    config = ProjectConfig.from_json(args.config)
    run = config.run
    if args.full:
        run = RunConfig(
            random_state=run.random_state,
            fast_mode=False,
            drop_exact_duplicates=run.drop_exact_duplicates,
            bootstrap_repetitions=max(run.bootstrap_repetitions, 1000),
            causal_bootstrap_repetitions=max(run.causal_bootstrap_repetitions, 100),
            deep_epochs=max(run.deep_epochs, 30),
            deep_patience=max(run.deep_patience, 6),
        )

    artifact_dir = ROOT / "artifacts"
    figure_dir = ROOT / "reports" / "figures"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading and validating canonical data")
    frame, audit = load_dataset(ROOT / "data", run.drop_exact_duplicates, args.force_download)
    (artifact_dir / "data_audit.json").write_text(json.dumps(audit_to_dict(audit), indent=2), encoding="utf-8")
    eda_figures(frame, figure_dir)

    print("[2/7] Creating leakage-resistant temporal partitions")
    parts = temporal_partitions(frame, config.split)
    split_summary(parts).to_csv(artifact_dir / "temporal_split_summary.csv", index=False)

    print("[3/7] Training classical and imbalance-aware candidates")
    model, metadata, _, classical_validation = train_and_select(parts, artifact_dir, run.random_state, run.fast_mode)

    print("[4/7] Training deep MLP challengers and resolving the overall validation winner")
    _, deep_details = run_deep_challenger(parts, artifact_dir, run.deep_epochs, run.deep_patience, run.random_state)
    classical_score = float(classical_validation.iloc[0]["validation_pr_auc"])
    if deep_details["validation_pr_auc"] > classical_score:
        model = refit_deep_candidate(
            parts, artifact_dir, deep_details["selected_loss"], deep_details["epochs_run"], run.random_state
        )
        metadata = {
            "selected_candidate": f"deep_mlp__{deep_details['selected_loss']}",
            "family": "deep_mlp",
            "strategy": deep_details["selected_loss"],
            "params": {"epochs": deep_details["epochs_run"]},
            "selection_metric": "validation_pr_auc",
            "validation_pr_auc": deep_details["validation_pr_auc"],
            "feature_names": model.feature_names,
        }
        (artifact_dir / "selected_model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    deep_details["selected_overall"] = metadata["family"] == "deep_mlp"
    (artifact_dir / "deep_challenger.json").write_text(json.dumps(deep_details, indent=2), encoding="utf-8")
    calibrator, _ = calibrate_selected_model(model, parts, artifact_dir, run.random_state)

    print("[5/7] Freezing policy and evaluating final holdout once")
    run_decision_analysis(
        model, calibrator, parts, config.cost, artifact_dir,
        bootstrap_repetitions=run.bootstrap_repetitions,
        seed=run.random_state,
    )
    final_prediction = np.load(artifact_dir / "final_test_predictions.npz")
    final_metrics = metric_bundle(final_prediction["y"], final_prediction["probability"])
    final_metrics["pr_auc_95pct_interval"] = bootstrap_metric_interval(
        final_prediction["y"], final_prediction["probability"], "pr_auc", run.bootstrap_repetitions, run.random_state
    )
    (artifact_dir / "selected_model_test_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    if deep_details["selected_overall"]:
        deep_details["final_test_metrics"] = final_metrics
        (artifact_dir / "deep_challenger.json").write_text(json.dumps(deep_details, indent=2), encoding="utf-8")

    print("[6/7] Running cross-fitted semi-synthetic causal audit")
    run_causal_demo(
        model, calibrator, parts, config.cost, artifact_dir,
        bootstrap_repetitions=run.causal_bootstrap_repetitions,
        seed=run.random_state,
    )

    print("[7/7] Building figures and result summary")
    model_figures(artifact_dir, figure_dir)
    decision_figures(artifact_dir, figure_dir)
    causal_figures(artifact_dir, figure_dir)
    dashboard(artifact_dir, figure_dir)
    build_results_markdown(ROOT)
    joblib.dump({"model": model, "calibrator": calibrator, "metadata": metadata}, artifact_dir / "fraud_decision_system.joblib")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
