from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from .features import make_features, target
from .metrics import metric_bundle


def _torch():
    import torch
    from torch import nn
    return torch, nn


class TorchMLP:
    def __init__(self, input_dim: int, random_state=42):
        torch, nn = _torch()
        torch.manual_seed(random_state)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )


class DeepPredictor:
    """Small sklearn-like wrapper used by calibration and decision modules."""

    def __init__(self, net, scaler, feature_names):
        self.net = net
        self.scaler = scaler
        self.feature_names = list(feature_names)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]
        transformed = self.scaler.transform(X).astype("float32")
        probability = _predict(self.net, transformed)
        return np.column_stack([1 - probability, probability])


def focal_loss(logits, targets, alpha=0.95, gamma=2.0):
    torch, _ = _torch()
    probability = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probability * targets + (1 - probability) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - pt).pow(gamma) * ce).mean()


def _predict(net, X, batch_size=16_384):
    torch, _ = _torch()
    net.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32)
            out.append(torch.sigmoid(net(xb).squeeze(1)).cpu().numpy())
    return np.concatenate(out)


def train_one(
    X_train,
    y_train,
    X_valid,
    y_valid,
    loss_name,
    epochs=18,
    patience=4,
    random_state=42,
):
    torch, nn = _torch()
    rng = np.random.default_rng(random_state)
    torch.manual_seed(random_state)
    holder = TorchMLP(X_train.shape[1], random_state)
    net = holder.net
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    batch_size = 4096
    best_ap = -np.inf
    best_state = None
    history = []
    stale = 0
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(len(X_train))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = torch.as_tensor(X_train[idx], dtype=torch.float32)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = net(xb).squeeze(1)
            loss = bce(logits, yb) if loss_name == "weighted_bce" else focal_loss(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probability = _predict(net, X_valid)
        ap = float(average_precision_score(y_valid, probability))
        history.append({"loss": loss_name, "epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_pr_auc": ap})
        if ap > best_ap + 1e-5:
            best_ap = ap
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    net.load_state_dict(best_state)
    return net, pd.DataFrame(history), best_ap


def train_fixed(X_train, y_train, loss_name, epochs, random_state=42):
    torch, nn = _torch()
    rng = np.random.default_rng(random_state)
    torch.manual_seed(random_state)
    net = TorchMLP(X_train.shape[1], random_state).net
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    batch_size = 4096
    for _ in range(epochs):
        net.train()
        order = rng.permutation(len(X_train))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = torch.as_tensor(X_train[idx], dtype=torch.float32)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = net(xb).squeeze(1)
            loss = bce(logits, yb) if loss_name == "weighted_bce" else focal_loss(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
    return net


def run_deep_challenger(parts, artifact_dir: str | Path, epochs=18, patience=4, random_state=42):
    artifact_dir = Path(artifact_dir)
    train_df, valid_df = parts["train"], parts["validation"]
    X_train_df, X_valid_df = map(make_features, [train_df, valid_df])
    y_train, y_valid = map(target, [train_df, valid_df])
    scaler = StandardScaler().fit(X_train_df)
    X_train = scaler.transform(X_train_df).astype("float32")
    X_valid = scaler.transform(X_valid_df).astype("float32")
    rows = []
    trained = {}
    histories = []
    for i, loss_name in enumerate(["weighted_bce", "focal"]):
        net, history, score = train_one(
            X_train, y_train, X_valid, y_valid,
            loss_name=loss_name,
            epochs=epochs,
            patience=patience,
            random_state=random_state + i,
        )
        trained[loss_name] = net
        histories.append(history)
        rows.append({"model": f"mlp_{loss_name}", "validation_pr_auc": score, "epochs_run": len(history)})
    validation = pd.DataFrame(rows).sort_values("validation_pr_auc", ascending=False)
    selected_loss = str(validation.iloc[0]["model"]).replace("mlp_", "")
    selected_row = validation.iloc[0]
    validation.to_csv(artifact_dir / "deep_validation_comparison.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(artifact_dir / "deep_training_history.csv", index=False)
    details = {
        "selected_loss": selected_loss,
        "selection_metric": "validation_pr_auc",
        "validation_pr_auc": float(selected_row["validation_pr_auc"]),
        "epochs_run": int(selected_row["epochs_run"]),
    }
    (artifact_dir / "deep_challenger.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return validation, details


def refit_deep_candidate(parts, artifact_dir: str | Path, selected_loss: str, epochs: int, random_state=42):
    torch, _ = _torch()
    artifact_dir = Path(artifact_dir)
    development = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    X_df = make_features(development)
    y = target(development)
    scaler = StandardScaler().fit(X_df)
    X = scaler.transform(X_df).astype("float32")
    net = train_fixed(X, y, selected_loss, epochs, random_state=random_state)
    bundle = DeepPredictor(net, scaler, X_df.columns.tolist())
    torch.save({
        "state_dict": net.state_dict(),
        "input_dim": X.shape[1],
        "feature_names": X_df.columns.tolist(),
        "selected_loss": selected_loss,
        "epochs": epochs,
    }, artifact_dir / "deep_mlp.pt")
    joblib.dump(scaler, artifact_dir / "deep_scaler.joblib")
    joblib.dump(bundle, artifact_dir / "selected_base_model.joblib")
    return bundle
