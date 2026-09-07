from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from ..config import PriceModelConfig
from ..features.price_features import PriceFeatureBundle
from ..features.targets import TargetTransform
from ..models.price_cnn_transformer import YieldCurveCNNTransformer
from ..progress import progress
from .common import RuntimeLogger, seed_everything, select_device
from .losses import joint_forecast_loss
from .price_dataset import PriceWindowDataset, eligible_anchors, sequential_price_splits


def _class_weight(dataset, device) -> torch.Tensor:
    classes = np.asarray([dataset[index][1].item() for index in range(len(dataset))])
    counts = np.bincount(classes, minlength=3)
    weights = counts.sum() / (3.0 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _run_epoch(
    model,
    loader,
    config,
    device,
    class_weight,
    optimizer=None,
    scheduler=None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses = []
    truth, predicted = [], []
    for price, target_class, target_value, _, _ in loader:
        price = price.to(device)
        target_class = target_class.to(device)
        target_value = target_value.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(price)
        loss = joint_forecast_loss(outputs, target_class, target_value, config, class_weight)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite historical price-model loss.")
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        losses.append(float(loss.detach().cpu()))
        truth.append(target_class.detach().cpu().numpy())
        predicted.append(outputs["direction_logits"].argmax(-1).detach().cpu().numpy())
    truth_array = np.concatenate(truth)
    predicted_array = np.concatenate(predicted)
    return {
        "loss": float(np.mean(losses)),
        "balanced_accuracy": float(balanced_accuracy_score(truth_array, predicted_array)),
        "distinct_classes": float(len(np.unique(predicted_array))),
    }


def train_historical_backbone(
    bundle: PriceFeatureBundle,
    first_fomc_meeting_date: pd.Timestamp,
    config: PriceModelConfig,
    checkpoint_path: Path,
    runtime_log: Path | None = None,
) -> dict:
    """Train only where every price label predates the first FOMC sample."""
    cutoff = pd.Timestamp(first_fomc_meeting_date)
    anchors = eligible_anchors(bundle, config, label_before=cutoff)
    splits = sequential_price_splits(anchors, config)
    last_label_date = pd.Timestamp(bundle.panel.index[anchors[-1] + config.horizon])
    if not last_label_date < cutoff:
        raise AssertionError("Historical-backbone label cutoff leaks into FOMC history.")
    target = TargetTransform(bundle, config)
    device = select_device()
    if runtime_log and Path(runtime_log).exists():
        Path(runtime_log).unlink()
    logger = RuntimeLogger(runtime_log) if runtime_log else None
    best_seed_payload = None
    seed_progress = progress(
        range(config.n_seeds),
        desc="Historical backbone seeds",
        total=config.n_seeds,
        unit="seed",
    )
    for seed in seed_progress:
        seed_progress.set_postfix(seed=f"{seed + 1}/{config.n_seeds}")
        seed_everything(seed)
        train_loader = DataLoader(
            PriceWindowDataset(bundle, target, splits["train"], config),
            batch_size=config.batch_size,
            shuffle=True,
        )
        validation_loader = DataLoader(
            PriceWindowDataset(bundle, target, splits["validation"], config),
            batch_size=config.batch_size,
            shuffle=False,
        )
        class_weight = _class_weight(train_loader.dataset, device)
        model = YieldCurveCNNTransformer(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        total_steps = max(len(train_loader), 1) * config.max_epochs
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
        best_score = -float("inf")
        best_state = None
        stale = 0
        collapse_streak = 0
        seed_started = time.time()
        epoch_progress = progress(
            range(config.max_epochs),
            desc=f"Backbone seed {seed + 1}/{config.n_seeds}",
            total=config.max_epochs,
            unit="epoch",
            leave=False,
        )
        for epoch in epoch_progress:
            epoch_started = time.time()
            train_result = _run_epoch(
                model,
                train_loader,
                config,
                device,
                class_weight,
                optimizer,
                scheduler,
            )
            with torch.no_grad():
                validation = _run_epoch(model, validation_loader, config, device, class_weight)
            if logger:
                logger.write(
                    "historical_backbone_epoch",
                    seed=seed,
                    epoch=epoch,
                    train_loss=train_result["loss"],
                    validation_loss=validation["loss"],
                    validation_balanced_accuracy=validation["balanced_accuracy"],
                    distinct_classes=int(validation["distinct_classes"]),
                    learning_rate=optimizer.param_groups[0]["lr"],
                    epoch_duration_seconds=time.time() - epoch_started,
                    estimated_seed_remaining_seconds=(
                        (time.time() - seed_started) / (epoch + 1) * (config.max_epochs - epoch - 1)
                    ),
                )
            if validation["balanced_accuracy"] > best_score + 1e-5:
                best_score = validation["balanced_accuracy"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
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
                raise RuntimeError("Historical backbone collapsed to one validation class.")
            if stale >= config.patience:
                break
        payload = {
            "seed": seed,
            "validation_balanced_accuracy": best_score,
            "state_dict": best_state,
        }
        if (
            best_seed_payload is None
            or best_score > best_seed_payload["validation_balanced_accuracy"]
        ):
            best_seed_payload = payload
    if best_seed_payload is None or best_seed_payload["state_dict"] is None:
        raise RuntimeError("Historical-backbone training produced no checkpoint.")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        **best_seed_payload,
        "price_config": asdict(config),
        "purpose": "fomc_historical_backbone",
        "first_fomc_meeting_date": cutoff.isoformat(),
        "last_training_label_date": last_label_date.isoformat(),
        "split_sizes": {name: len(values) for name, values in splits.items()},
    }
    torch.save(checkpoint, checkpoint_path)
    return {key: value for key, value in checkpoint.items() if key != "state_dict"}


def load_historical_backbone(
    checkpoint_path: Path,
    config: PriceModelConfig,
    first_fomc_meeting_date: pd.Timestamp,
    map_location: str | torch.device = "cpu",
) -> tuple[YieldCurveCNNTransformer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if checkpoint.get("purpose") != "fomc_historical_backbone":
        raise ValueError("Checkpoint is not a dedicated FOMC historical backbone.")
    last_label = pd.Timestamp(checkpoint["last_training_label_date"])
    first_meeting = pd.Timestamp(first_fomc_meeting_date)
    if not last_label < first_meeting:
        raise AssertionError("Backbone checkpoint violates the pre-FOMC leakage boundary.")
    if checkpoint.get("price_config") != asdict(config):
        raise ValueError("Backbone checkpoint configuration does not match current config.")
    model = YieldCurveCNNTransformer(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
