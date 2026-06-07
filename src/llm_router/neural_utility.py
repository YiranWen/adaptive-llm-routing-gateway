"""Neural utility prediction router.

The model predicts two deployment utilities for each context:

    [utility_local, utility_cloud]

The route decision is the argmax of the predicted utilities.  This gives the
neural model the same objective and output format as the linear Ridge utility
baseline, while allowing nonlinear interactions between prompt activations,
serving state, and SLA weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NeuralUtilityConfig:
    """Hyperparameters for the neural utility predictor."""

    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 500
    patience: int = 40
    ranking_weight: float = 0.20
    seed: int = 184


@dataclass
class NeuralUtilityTrainingResult:
    """Trained neural utility predictor and preprocessing state."""

    config: NeuralUtilityConfig
    state_dict: dict[str, Any]
    input_mean: np.ndarray
    input_std: np.ndarray
    best_epoch: int
    best_dev_loss: float
    train_history: list[dict[str, float | int]]


def standardize_fit_transform(
    train_features: np.ndarray,
    dev_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize train/dev features using train statistics."""

    mean = train_features.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_features.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    train_scaled = ((train_features - mean) / std).astype(np.float32)
    dev_scaled = ((dev_features - mean) / std).astype(np.float32)
    return train_scaled, dev_scaled, mean, std


def standardize_apply(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply saved standardization statistics."""

    return ((features - mean) / std).astype(np.float32)


def make_model(input_dim: int, config: NeuralUtilityConfig):
    """Build the PyTorch network lazily so importing this module stays light."""

    import torch
    from torch import nn

    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in config.hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 2))

    model = nn.Sequential(*layers)
    torch.manual_seed(config.seed)
    return model


def utility_loss(predicted, target, *, ranking_weight: float):
    """MSE utility loss plus an auxiliary margin-ranking loss."""

    import torch
    from torch.nn import functional as F

    mse = F.mse_loss(predicted, target)
    if ranking_weight <= 0:
        return mse

    predicted_gap = predicted[:, 1] - predicted[:, 0]
    true_gap = target[:, 1] - target[:, 0]
    cloud_is_better = (true_gap > 0).float()
    weights = torch.clamp(torch.abs(true_gap), min=0.01, max=0.50)
    ranking = F.binary_cross_entropy_with_logits(
        predicted_gap,
        cloud_is_better,
        weight=weights,
    )
    return mse + ranking_weight * ranking


def train_neural_utility_router(
    train_features: np.ndarray,
    train_utilities: np.ndarray,
    dev_features: np.ndarray,
    dev_utilities: np.ndarray,
    *,
    config: NeuralUtilityConfig,
) -> NeuralUtilityTrainingResult:
    """Train a neural utility predictor with early stopping on dev loss."""

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    train_x, dev_x, mean, std = standardize_fit_transform(train_features, dev_features)
    train_y = train_utilities.astype(np.float32)
    dev_y = dev_utilities.astype(np.float32)

    model = make_model(train_x.shape[1], config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(5, config.patience // 4),
    )

    generator = torch.Generator().manual_seed(config.seed)
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    dev_x_tensor = torch.from_numpy(dev_x)
    dev_y_tensor = torch.from_numpy(dev_y)

    best_state = None
    best_dev_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        # Keep a deterministic but nontrivial RNG touch so repeated configs are
        # not affected by global NumPy state.
        _ = rng.random()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            predicted = model(batch_x)
            loss = utility_loss(
                predicted,
                batch_y,
                ranking_weight=config.ranking_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_losses.append(float(loss.detach().item()))

        model.eval()
        with torch.inference_mode():
            dev_pred = model(dev_x_tensor)
            dev_loss = float(
                utility_loss(
                    dev_pred,
                    dev_y_tensor,
                    ranking_weight=config.ranking_weight,
                ).item()
            )
        scheduler.step(dev_loss)

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if dev_loss < best_dev_loss - 1e-6:
            best_dev_loss = dev_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    return NeuralUtilityTrainingResult(
        config=config,
        state_dict=best_state,
        input_mean=mean,
        input_std=std,
        best_epoch=best_epoch,
        best_dev_loss=float(best_dev_loss),
        train_history=history,
    )


def predict_neural_utilities(
    result: NeuralUtilityTrainingResult,
    features: np.ndarray,
) -> np.ndarray:
    """Predict [utility_local, utility_cloud] for a feature matrix."""

    import torch

    x = standardize_apply(features.astype(np.float32), result.input_mean, result.input_std)
    model = make_model(x.shape[1], result.config)
    model.load_state_dict(result.state_dict)
    model.eval()
    with torch.inference_mode():
        predictions = model(torch.from_numpy(x)).detach().cpu().numpy()
    return predictions.astype(np.float32)
