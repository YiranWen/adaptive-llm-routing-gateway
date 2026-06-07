"""Dataset preparation utilities for offline routing experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATASET_NAME = "routellm/gpt4_dataset"
DEFAULT_SMALL_MODEL = "mixtral"
DEFAULT_LARGE_MODEL = "gpt4"


@dataclass(frozen=True)
class PreparedDatasetMetadata:
    """Metadata saved next to a prepared routing table."""

    dataset_name: str
    split: str
    num_rows: int
    small_model: str
    large_model: str
    small_failure_threshold: float
    latency_small: float
    latency_large: float
    seed: int
    shuffled: bool


def prepare_gpt4_dataset_table(
    dataset: Any,
    *,
    small_failure_threshold: float = 0.7,
    latency_small: float = 0.2,
    latency_large: float = 3.0,
) -> pd.DataFrame:
    """Convert a RouteLLM GPT-4 judge dataset split into a routing table."""

    df = dataset.to_pandas()
    required_columns = {"prompt", "mixtral_score"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")

    table = pd.DataFrame()
    table["prompt"] = df["prompt"].astype(str)
    table["source"] = df.get("source", "").astype(str)
    table["score_small"] = pd.to_numeric(df["mixtral_score"], errors="coerce") / 5.0
    table["score_large"] = 1.0
    table["latency_small"] = float(latency_small)
    table["latency_large"] = float(latency_large)
    table["small_model"] = DEFAULT_SMALL_MODEL
    table["large_model"] = DEFAULT_LARGE_MODEL
    table["small_failure"] = (table["score_small"] < small_failure_threshold).astype(int)

    table = table.dropna(subset=["prompt", "score_small"]).reset_index(drop=True)
    return table


def save_metadata(path: Path, metadata: PreparedDatasetMetadata) -> None:
    """Write metadata as pretty JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")


def load_routing_table(path: Path) -> pd.DataFrame:
    """Load a prepared routing table from CSV or Parquet."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported routing table format: {path}")

