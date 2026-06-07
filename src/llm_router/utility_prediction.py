"""Utility Prediction router helper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UtilityPredictionResult:
    """Trained Utility Prediction router metadata."""

    pca_dim: int | None
    ridge_alpha: float


def train_utility_prediction_router(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_utility_matrix: np.ndarray,
    *,
    pca_dim: int | None = 64,
    ridge_alpha: float = 10.0,
) -> tuple[UtilityPredictionResult, np.ndarray]:
    """Predict each arm utility and route to the arm with larger prediction."""

    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps: list[object] = [StandardScaler()]
    if pca_dim is not None:
        steps.append(PCA(n_components=pca_dim, random_state=184))
    steps.append(Ridge(alpha=ridge_alpha))

    model = make_pipeline(*steps)
    model.fit(train_features, train_utility_matrix)
    predicted_utilities = model.predict(validation_features)
    actions = np.argmax(predicted_utilities, axis=1).astype(np.int64)
    return UtilityPredictionResult(pca_dim=pca_dim, ridge_alpha=ridge_alpha), actions
