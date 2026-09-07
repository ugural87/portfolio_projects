from __future__ import annotations

import copy
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

from ..config import FusionConfig, PriceModelConfig
from ..features.event_dataset import FomcEventDataset
from ..features.event_scalers import EventScalers, fit_event_scalers
from ..features.price_features import PriceFeatureBundle
from ..features.targets import TargetTransform
from ..progress import progress
from .common import RuntimeLogger, seed_everything, select_device
from .losses import joint_forecast_loss
from .selection import ValidationLossCheckpoint


class ShuffledSemanticDataset(Dataset):
    """Destroy text/event alignment while leaving prices, rates and labels unchanged."""

    def __init__(self, base: FomcEventDataset, seed: int) -> None:
        self.base = base
        self.events = base.events
        self.permutation = np.random.default_rng(seed).permutation(len(base))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int):
        original = list(self.base[item])
        donor = self.base[int(self.permutation[item])]
        original[3] = donor[3]
        original[4] = donor[4]
        return tuple(original)


def build_event_loaders(
    events: pd.DataFrame,
    splits: dict[str, np.ndarray],
    bundle: PriceFeatureBundle,
    target: TargetTransform,
    config: FusionConfig,
    scalers: EventScalers | None = None,
    shuffle_semantics: bool = False,
    seed_offset: int = 0,
) -> tuple[dict[str, DataLoader], EventScalers]:
    scalers = scalers or fit_event_scalers(events, splits["train"])
    loaders: dict[str, DataLoader] = {}
    for split_number, (name, indices) in enumerate(splits.items()):
        dataset: Dataset = FomcEventDataset(events, indices, bundle, target, scalers, config)
        if shuffle_semantics:
            dataset = ShuffledSemanticDataset(
                dataset, seed=config.random_seed + seed_offset + split_number
            )
        loaders[name] = DataLoader(
            dataset,
            # Event validation/calibration/test splits are small. A single deterministic
            # batch makes validation joint loss an exact split-level objective.
            batch_size=config.batch_size if name == "train" else max(len(dataset), 1),
            shuffle=name == "train",
            drop_last=False,
        )
    return loaders, scalers


def _move_batch(batch, device):
    return [value.to(device) for value in batch]


def _forward(model, batch):
    price, price_mask, rates, semantic, semantic_mask, classes, values, scales, rows = batch
    outputs = model(price, price_mask, rates, semantic, semantic_mask)
    return outputs, classes, values, scales, rows


def train_event_model(
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    price_config: PriceModelConfig,
    fusion_config: FusionConfig,
    seed: int,
    runtime_log: Path | None = None,
    stage: str = "fusion",
) -> tuple[torch.nn.Module, pd.DataFrame]:
    seed_everything(seed)
    device = select_device()
    model = model.to(device)
    train_classes = np.asarray(
        [
            loaders["train"].dataset[index][5].item()
            for index in range(len(loaders["train"].dataset))
        ]
    )
    class_counts = np.bincount(train_classes, minlength=3)
    class_weights = class_counts.sum() / (3.0 * np.maximum(class_counts, 1))
    class_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=fusion_config.learning_rate,
        weight_decay=fusion_config.weight_decay,
    )

    def learning_rate_multiplier(epoch: int) -> float:
        if epoch < fusion_config.warmup_epochs:
            return float(epoch + 1) / max(float(fusion_config.warmup_epochs), 1.0)
        progress = (epoch - fusion_config.warmup_epochs) / max(
            fusion_config.max_epochs - fusion_config.warmup_epochs, 1
        )
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    logger = RuntimeLogger(runtime_log) if runtime_log else None
    best_state = None
    checkpoint = ValidationLossCheckpoint(patience=fusion_config.patience)
    collapse_streak = 0
    history = []
    training_started = time.time()
    epoch_progress = progress(
        range(fusion_config.max_epochs),
        desc=stage,
        total=fusion_config.max_epochs,
        unit="epoch",
        leave=False,
    )
    for epoch in epoch_progress:
        epoch_started = time.time()
        epoch_losses = {}
        validation_truth = []
        validation_predicted = []
        for split in ("train", "validation"):
            training = split == "train"
            model.train(training)
            losses = []
            context = torch.enable_grad() if training else torch.no_grad()
            with context:
                for raw_batch in loaders[split]:
                    batch = _move_batch(raw_batch, device)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    outputs, classes, values, _, _ = _forward(model, batch)
                    loss = joint_forecast_loss(outputs, classes, values, price_config, class_weight)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite {stage} loss at epoch {epoch}.")
                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            price_config.gradient_clip,
                        )
                        optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                    if not training:
                        validation_truth.append(classes.detach().cpu().numpy())
                        validation_predicted.append(
                            outputs["direction_logits"].argmax(-1).detach().cpu().numpy()
                        )
            epoch_losses[split] = float(np.mean(losses))
        truth = np.concatenate(validation_truth)
        predicted = np.concatenate(validation_predicted)
        validation_balanced_accuracy = float(balanced_accuracy_score(truth, predicted))
        distinct_classes = int(len(np.unique(predicted)))
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_balanced_accuracy": validation_balanced_accuracy,
            "distinct_classes": distinct_classes,
            **epoch_losses,
        }
        checkpoint_improved = checkpoint.observe(epoch, epoch_losses["validation"])
        if checkpoint_improved:
            best_state = copy.deepcopy(model.state_dict())
        row["checkpoint_selection_metric"] = "validation_joint_loss"
        row["checkpoint_improved"] = checkpoint_improved
        row["best_validation_loss_so_far"] = checkpoint.best_loss
        row["stale_epochs"] = checkpoint.stale_epochs
        epoch_duration = time.time() - epoch_started
        mean_epoch_duration = (time.time() - training_started) / (epoch + 1)
        row["epoch_duration_seconds"] = epoch_duration
        row["estimated_stage_remaining_seconds"] = mean_epoch_duration * (
            fusion_config.max_epochs - epoch - 1
        )
        history.append(row)
        if logger:
            logger.write(f"{stage}_epoch", seed=seed, **row)
        epoch_progress.set_postfix(
            train=f"{epoch_losses['train']:.4f}",
            validation=f"{epoch_losses['validation']:.4f}",
            best=f"{checkpoint.best_loss:.4f}",
            bacc=f"{validation_balanced_accuracy:.3f}",
            stale=checkpoint.stale_epochs,
        )
        collapse_streak = collapse_streak + 1 if distinct_classes < 2 else 0
        if collapse_streak >= fusion_config.patience:
            raise RuntimeError(
                f"Persistent prediction collapse in {stage}: one class for "
                f"{collapse_streak} validation epochs."
            )
        scheduler.step()
        if checkpoint.should_stop:
            break
    if best_state is None:
        raise RuntimeError(f"{stage} training produced no valid model state.")
    model.load_state_dict(best_state)
    model.eval()
    restored_losses = []
    with torch.no_grad():
        for raw_batch in loaders["validation"]:
            batch = _move_batch(raw_batch, device)
            outputs, classes, values, _, _ = _forward(model, batch)
            restored_losses.append(
                float(
                    joint_forecast_loss(outputs, classes, values, price_config, class_weight)
                    .detach()
                    .cpu()
                )
            )
    restored_validation_loss = float(np.mean(restored_losses))
    if not np.isclose(
        restored_validation_loss,
        checkpoint.best_loss,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise AssertionError(
            f"{stage} did not restore the validation-loss checkpoint: "
            f"expected {checkpoint.best_loss:.8f}, got {restored_validation_loss:.8f}."
        )
    history_frame = pd.DataFrame(history)
    history_frame["selected_checkpoint"] = history_frame.epoch.eq(checkpoint.best_epoch)
    history_frame["restored_validation_loss"] = restored_validation_loss
    if int(history_frame.selected_checkpoint.sum()) != 1:
        raise AssertionError("Training history must identify exactly one checkpoint epoch.")
    return model, history_frame


@torch.no_grad()
def collect_event_predictions(model, loader: DataLoader) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "direction_logits": [],
        "quantiles": [],
        "classes": [],
        "labels": [],
        "scales": [],
        "rows": [],
    }
    attention_keys = (
        "event_to_price_attention",
        "price_to_event_attention",
        "event_valid_mask",
        "price_valid_mask",
    )
    attention = {key: [] for key in attention_keys}
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        outputs, classes, values, scales, rows = _forward(model, batch)
        for key in ("direction_logits", "quantiles"):
            collected[key].append(outputs[key].detach().cpu().numpy())
        for key, value in zip(
            ("classes", "labels", "scales", "rows"), (classes, values, scales, rows)
        ):
            collected[key].append(value.detach().cpu().numpy())
        for key in attention_keys:
            if key in outputs:
                attention[key].append(outputs[key].detach().cpu().numpy())
    result = {key: np.concatenate(values) for key, values in collected.items()}
    result.update({key: np.concatenate(values) for key, values in attention.items() if values})
    return result


def save_event_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_name: str,
    price_config: PriceModelConfig,
    fusion_config: FusionConfig,
    scalers: EventScalers,
    metadata: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "price_config": asdict(price_config),
            "fusion_config": asdict(fusion_config),
            "feature_mean": scalers.feature_mean.to_dict(),
            "feature_std": scalers.feature_std.to_dict(),
            "rate_mean": scalers.rate_mean.to_dict(),
            "rate_std": scalers.rate_std.to_dict(),
            "metadata": metadata,
        },
        path,
    )
