"""PyTorch MLP baseline router."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llm_router.utilities import LARGE, SMALL


@dataclass(frozen=True)
class MLPTrainingResult:
    """Trained MLP router and preprocessing state."""

    model: object
    mean: np.ndarray
    std: np.ndarray
    train_loss: float


def standardize_train_validation(
    train_features: np.ndarray,
    validation_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((train_features - mean) / std).astype(np.float32),
        ((validation_features - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def train_mlp_router(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    *,
    seed: int = 184,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    weight_decay: float = 1e-4,
) -> tuple[MLPTrainingResult, np.ndarray]:
    """Train an MLP to predict whether the small model fails."""

    import torch
    from torch import nn

    torch.manual_seed(seed)
    x_train, x_val, mean, std = standardize_train_validation(
        train_features,
        validation_features,
    )
    y_train = train_labels.astype(np.float32).reshape(-1, 1)

    model = nn.Sequential(
        nn.Linear(x_train.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=0.10),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 1),
    )

    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([negatives / positives]) if positives else torch.tensor([1.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    model.train()
    final_loss = 0.0
    for _epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())

    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(torch.from_numpy(x_val))).numpy().reshape(-1)
    actions = np.where(probabilities >= 0.5, LARGE, SMALL).astype(np.int64)

    return MLPTrainingResult(model, mean, std, final_loss), actions
