"""Load cached offline routing artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from llm_router.dataset import load_routing_table


@dataclass(frozen=True)
class OfflineSplit:
    """Cached routing examples for one split."""

    features: np.ndarray
    score_small: np.ndarray
    score_large: np.ndarray
    latency_small: np.ndarray
    latency_large: np.ndarray
    small_failure: np.ndarray
    prompts: np.ndarray
    table: pd.DataFrame


@dataclass(frozen=True)
class OfflineArtifacts:
    """Train/validation offline routing artifacts."""

    train: OfflineSplit
    validation: OfflineSplit


def load_offline_split(feature_path: Path, table_path: Path) -> OfflineSplit:
    """Load and validate a cached feature file plus its routing table."""

    cache = np.load(feature_path, allow_pickle=True)
    table = load_routing_table(table_path)
    required_arrays = {
        "features",
        "score_small",
        "score_large",
        "latency_small",
        "latency_large",
        "small_failure",
        "prompt",
    }
    missing = required_arrays.difference(cache.files)
    if missing:
        raise KeyError(f"{feature_path} is missing arrays: {sorted(missing)}")

    features = cache["features"].astype(np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")

    num_rows = features.shape[0]
    if len(table) != num_rows:
        raise ValueError(
            f"row mismatch: {feature_path} has {num_rows}, "
            f"but {table_path} has {len(table)}"
        )

    arrays = {
        key: cache[key]
        for key in [
            "score_small",
            "score_large",
            "latency_small",
            "latency_large",
            "small_failure",
            "prompt",
        ]
    }
    for key, value in arrays.items():
        if value.shape[0] != num_rows:
            raise ValueError(f"{key} has {value.shape[0]} rows, expected {num_rows}")

    return OfflineSplit(
        features=features,
        score_small=arrays["score_small"].astype(np.float32),
        score_large=arrays["score_large"].astype(np.float32),
        latency_small=arrays["latency_small"].astype(np.float32),
        latency_large=arrays["latency_large"].astype(np.float32),
        small_failure=arrays["small_failure"].astype(np.int64),
        prompts=arrays["prompt"],
        table=table,
    )


def load_offline_artifacts(
    *,
    train_features: Path,
    validation_features: Path,
    train_table: Path,
    validation_table: Path,
) -> OfflineArtifacts:
    """Load train and validation offline artifacts."""

    return OfflineArtifacts(
        train=load_offline_split(train_features, train_table),
        validation=load_offline_split(validation_features, validation_table),
    )

