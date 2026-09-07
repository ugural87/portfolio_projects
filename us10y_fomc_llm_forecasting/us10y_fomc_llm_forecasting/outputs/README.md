# Generated outputs

This release bundles the completed real-data run: the historical backbone, 32 event-model
checkpoints, 178 OOS predictions across 8 folds, training histories, attention tables, conformal
intervals, figures, runtime logs and verification evidence.

Event checkpoints were selected by minimum validation joint loss and re-evaluated after best-weight
restoration. `outputs/reports/v7_1_release_verification.json` contains the machine-readable audit.

The final holdout is sealed by default. Do not repeatedly enable it while changing the model.
