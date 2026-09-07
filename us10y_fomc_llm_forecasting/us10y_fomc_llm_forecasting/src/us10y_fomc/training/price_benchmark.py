from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import PriceModelConfig
from ..features.price_features import PriceFeatureBundle
from ..features.targets import TargetTransform
from ..models.price_cnn_transformer import YieldCurveCNNTransformer
from ..progress import progress
from .backbone_training import _class_weight, _run_epoch
from .common import RuntimeLogger, seed_everything, select_device
from .conformal import conformal_correction
from .price_dataset import PriceWindowDataset, eligible_anchors, sequential_price_splits


@torch.no_grad()
def _collect(model, loader, device) -> dict[str, np.ndarray]:
    model.eval()
    values = {key: [] for key in ("logits", "quantiles", "labels", "classes", "scales", "anchors")}
    for price, classes, labels, scales, anchors in loader:
        outputs = model(price.to(device))
        values["logits"].append(outputs["direction_logits"].cpu().numpy())
        values["quantiles"].append(outputs["quantiles"].cpu().numpy())
        for key, value in zip(
            ("labels", "classes", "scales", "anchors"), (labels, classes, scales, anchors)
        ):
            values[key].append(value.numpy())
    return {key: np.concatenate(parts) for key, parts in values.items()}


def train_price_benchmark(
    bundle: PriceFeatureBundle,
    config: PriceModelConfig,
    checkpoint_path: Path,
    prediction_path: Path,
    runtime_log: Path | None = None,
) -> dict[str, object]:
    """Standalone daily price benchmark; its weights are never used by FOMC fusion."""
    anchors = eligible_anchors(bundle, config)
    splits = sequential_price_splits(anchors, config)
    target = TargetTransform(bundle, config)
    loaders = {
        name: DataLoader(
            PriceWindowDataset(bundle, target, values, config),
            batch_size=config.batch_size,
            shuffle=name == "train",
        )
        for name, values in splits.items()
    }
    device = select_device()
    if runtime_log and Path(runtime_log).exists():
        Path(runtime_log).unlink()
    logger = RuntimeLogger(runtime_log) if runtime_log else None
    class_weight = _class_weight(loaders["train"].dataset, device)
    selected = None
    seed_progress = progress(
        range(config.n_seeds),
        desc="Price benchmark seeds",
        total=config.n_seeds,
        unit="seed",
    )
    for seed in seed_progress:
        seed_progress.set_postfix(seed=f"{seed + 1}/{config.n_seeds}")
        seed_everything(seed)
        model = YieldCurveCNNTransformer(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        total_steps = max(len(loaders["train"]), 1) * config.max_epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: (
                (step + 1) / max(config.warmup_steps, 1)
                if step < config.warmup_steps
                else 0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * min(
                            (step - config.warmup_steps)
                            / max(total_steps - config.warmup_steps, 1),
                            1.0,
                        )
                    )
                )
            ),
        )
        best_score, best_state, stale = -float("inf"), None, 0
        collapse_streak = 0
        seed_started = time.time()
        epoch_progress = progress(
            range(config.max_epochs),
            desc=f"Price benchmark seed {seed + 1}/{config.n_seeds}",
            total=config.max_epochs,
            unit="epoch",
            leave=False,
        )
        for epoch in epoch_progress:
            epoch_started = time.time()
            train_result = _run_epoch(
                model,
                loaders["train"],
                config,
                device,
                class_weight,
                optimizer,
                scheduler,
            )
            with torch.no_grad():
                validation = _run_epoch(model, loaders["validation"], config, device, class_weight)
            if logger:
                logger.write(
                    "price_benchmark_epoch",
                    seed=seed,
                    epoch=epoch,
                    train_loss=train_result["loss"],
                    validation_loss=validation["loss"],
                    validation_balanced_accuracy=validation["balanced_accuracy"],
                    epoch_duration_seconds=time.time() - epoch_started,
                    estimated_seed_remaining_seconds=(
                        (time.time() - seed_started) / (epoch + 1) * (config.max_epochs - epoch - 1)
                    ),
                )
            if validation["balanced_accuracy"] > best_score + 1e-5:
                best_score = validation["balanced_accuracy"]
                best_state = {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            epoch_progress.set_postfix(
                train=f"{train_result['loss']:.4f}",
                validation=f"{validation['loss']:.4f}",
                bacc=f"{validation['balanced_accuracy']:.3f}",
                best=f"{best_score:.3f}",
                stale=stale,
            )
            collapse_streak = collapse_streak + 1 if validation["distinct_classes"] < 2 else 0
            if collapse_streak >= config.patience:
                raise RuntimeError("Standalone price benchmark collapsed to one class.")
            if stale >= config.patience:
                break
        candidate = {
            "seed": seed,
            "validation_balanced_accuracy": best_score,
            "state_dict": best_state,
        }
        if selected is None or best_score > selected["validation_balanced_accuracy"]:
            selected = candidate
    if selected is None or selected["state_dict"] is None:
        raise RuntimeError("Price benchmark produced no checkpoint.")
    model = YieldCurveCNNTransformer(config).to(device)
    model.load_state_dict(selected["state_dict"])
    calibration = _collect(model, loaders["calibration"], device)
    qhat = conformal_correction(
        calibration["labels"],
        calibration["quantiles"][:, 0],
        calibration["quantiles"][:, 2],
        config.conformal_alpha,
    )
    test = _collect(model, loaders["test"], device)
    dates = bundle.panel.index[test["anchors"] + config.horizon]
    predictions = pd.DataFrame(
        {
            "target_date": dates,
            "truth_bp": test["labels"] * test["scales"],
            "forecast_bp": test["quantiles"][:, 1] * test["scales"],
            "lower_bp": (test["quantiles"][:, 0] - qhat) * test["scales"],
            "upper_bp": (test["quantiles"][:, 2] + qhat) * test["scales"],
            "true_class": test["classes"],
            "predicted_class": test["logits"].argmax(axis=1),
        }
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **selected,
            "purpose": "standalone_price_benchmark_not_for_fusion",
            "price_config": asdict(config),
            "conformal_qhat": qhat,
            "last_training_label_date": bundle.panel.index[
                splits["train"][-1] + config.horizon
            ].isoformat(),
        },
        checkpoint_path,
    )
    return {
        "selected_seed": selected["seed"],
        "validation_balanced_accuracy": selected["validation_balanced_accuracy"],
        "conformal_qhat": qhat,
        "test_rows": len(predictions),
    }
