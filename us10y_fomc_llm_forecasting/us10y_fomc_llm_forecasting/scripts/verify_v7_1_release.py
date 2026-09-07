"""Verify the v7.1 release against a fully executed v7 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("price_only", "rate_only", "shuffled_text", "fusion")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_hashes(root: Path, relative_directory: str) -> dict[str, str]:
    directory = root / relative_directory
    return {
        str(path.relative_to(directory)): _sha256(path) for path in sorted(directory.rglob("*.py"))
    }


def _finite_frame(frame: pd.DataFrame, columns: list[str]) -> bool:
    values = frame[columns].to_numpy(dtype=float)
    return bool(np.isfinite(values).all())


def verify(baseline: Path) -> dict[str, object]:
    checks: dict[str, object] = {}

    current_manifest = pd.read_csv(ROOT / "outputs/tables/walk_forward_manifest.csv")
    baseline_manifest = pd.read_csv(baseline / "outputs/tables/walk_forward_manifest.csv")
    pd.testing.assert_frame_equal(current_manifest, baseline_manifest)
    checks["walk_forward_manifest_identical"] = True

    current_predictions = pd.read_csv(ROOT / "outputs/predictions/walk_forward_predictions.csv")
    baseline_predictions = pd.read_csv(
        baseline / "outputs/predictions/walk_forward_predictions.csv"
    )
    identity_columns = ["fold", "meeting_date", "target_date", "truth_bp", "true_class"]
    pd.testing.assert_frame_equal(
        current_predictions[identity_columns], baseline_predictions[identity_columns]
    )
    assert len(current_predictions) == 178
    assert current_predictions[identity_columns].duplicated().sum() == 0
    numeric_columns = [
        column
        for column in current_predictions.columns
        if column not in {"meeting_date", "target_date"}
    ]
    assert _finite_frame(current_predictions, numeric_columns)
    assert (current_predictions["fusion_lower_bp"] <= current_predictions["fusion_upper_bp"]).all()
    checks["oos_predictions"] = {
        "rows": len(current_predictions),
        "folds": int(current_predictions["fold"].nunique()),
        "identity_matches_v7": True,
        "finite": True,
        "intervals_ordered": True,
    }

    protected_directories = ["src/us10y_fomc/models", "src/us10y_fomc/features"]
    for relative_directory in protected_directories:
        current_hashes = _relative_hashes(ROOT, relative_directory)
        baseline_hashes = _relative_hashes(baseline, relative_directory)
        assert current_hashes == baseline_hashes
    checks["model_and_feature_source_identical_to_v7"] = True

    history_checks: list[dict[str, object]] = []
    checkpoint_directory = ROOT / "outputs/checkpoints/fusion"
    baseline_checkpoint_directory = baseline / "outputs/checkpoints/fusion"
    checkpoints = sorted(checkpoint_directory.glob("fold_*.pt"))
    assert len(checkpoints) == 32
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        baseline_checkpoint = torch.load(
            baseline_checkpoint_directory / checkpoint_path.name,
            map_location="cpu",
            weights_only=False,
        )
        fold = int(checkpoint["metadata"]["fold"])
        model_name = str(checkpoint["model_name"])
        assert model_name in MODELS
        assert checkpoint["metadata"]["checkpoint_selection_metric"] == "validation_joint_loss"
        assert checkpoint["metadata"]["checkpoint_min_delta"] == 1e-5
        assert checkpoint["metadata"]["early_stopping_patience"] == 35
        assert set(checkpoint["state_dict"]) == set(baseline_checkpoint["state_dict"])
        assert all(
            checkpoint["state_dict"][key].shape == baseline_checkpoint["state_dict"][key].shape
            for key in checkpoint["state_dict"]
        )
        assert all(torch.isfinite(value).all() for value in checkpoint["state_dict"].values())

        history_path = ROOT / "outputs/metrics" / f"fold_{fold}_{model_name}_training.csv"
        history = pd.read_csv(history_path)
        assert len(history) == checkpoint["metadata"]["epochs_completed"]
        assert history["selected_checkpoint"].sum() == 1
        selected = history.loc[history["selected_checkpoint"]].iloc[0]
        min_loss = float(history["validation"].min())
        selected_loss = float(selected["validation"])
        restored_loss = float(history["restored_validation_loss"].iloc[0])
        assert selected_loss <= min_loss + 1e-5
        assert abs(selected_loss - checkpoint["metadata"]["selected_validation_loss"]) < 1e-6
        assert abs(restored_loss - selected_loss) < 1e-6
        assert int(selected["epoch"]) == checkpoint["metadata"]["selected_epoch"]
        assert history["checkpoint_selection_metric"].eq("validation_joint_loss").all()
        history_checks.append(
            {
                "fold": fold,
                "model": model_name,
                "epochs_completed": len(history),
                "selected_epoch": int(selected["epoch"]),
                "selected_validation_loss": selected_loss,
            }
        )
    checks["event_checkpoints"] = {
        "count": len(checkpoints),
        "all_minimum_validation_loss": True,
        "all_restored_and_rechecked": True,
        "state_contract_matches_v7": True,
        "details": history_checks,
    }

    summary_path = ROOT / "outputs/reports/postprocess_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_oos"] == 178
    assert 0.0 <= summary["pooled_interval_coverage"] <= 1.0
    checks["report"] = {
        "n_oos": summary["n_oos"],
        "pooled_interval_coverage": summary["pooled_interval_coverage"],
        "fusion_mae_bp": summary["models"]["fusion"]["mae_bp"],
        "price_only_mae_bp": summary["models"]["price_only"]["mae_bp"],
        "rate_only_mae_bp": summary["models"]["rate_only"]["mae_bp"],
        "shuffled_text_mae_bp": summary["models"]["shuffled_text"]["mae_bp"],
    }

    checks["status"] = "PASS"
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to an executed v7 project directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/v7_1_release_verification.json",
    )
    args = parser.parse_args()
    result = verify(args.baseline.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "event_checkpoints"}, indent=2
        )
    )
    print(f"Full verification written to {args.output}")


if __name__ == "__main__":
    main()
