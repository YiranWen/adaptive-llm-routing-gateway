"""Real-time aligned routing utilities.

The offline experiment and Streamlit demo share this formulation:

context = concat(Qwen activation features, serving/system state features)
action = local route or cloud route
utility = quality - alpha * cost - beta * latency
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llm_router.offline_data import OfflineSplit
from llm_router.utilities import LARGE, SMALL, SLAMode


SYSTEM_FEATURE_NAMES = [
    "cpu_percent",
    "memory_percent",
    "battery_percent",
    "budget_remaining",
    "estimated_local_latency",
    "estimated_cloud_latency",
]

DEFAULT_REMOTE_COST_PER_REQUEST = 0.05


@dataclass(frozen=True)
class RealtimeAugmentedSplit:
    """Prompt features expanded with simulated serving states."""

    contexts: np.ndarray
    qwen_features: np.ndarray
    system_features: np.ndarray
    score_small: np.ndarray
    score_large: np.ndarray
    small_failure: np.ndarray
    prompt_index: np.ndarray
    prompts: np.ndarray
    remote_cost_per_request: float


def generate_serving_states(
    num_prompts: int,
    *,
    scenarios_per_prompt: int = 5,
    seed: int = 184,
) -> np.ndarray:
    """Generate realistic normalized system/budget/latency states."""

    rng = np.random.default_rng(seed)
    n = num_prompts * scenarios_per_prompt

    cpu = rng.beta(2.0, 3.0, size=n).astype(np.float32)
    memory = rng.beta(2.5, 2.5, size=n).astype(np.float32)
    battery = rng.uniform(0.15, 1.0, size=n).astype(np.float32)
    budget = rng.beta(2.0, 2.0, size=n).astype(np.float32)
    budget = np.clip(budget, 0.02, 1.0)

    local_noise = rng.normal(0.0, 0.08, size=n).astype(np.float32)
    cloud_noise = rng.normal(0.0, 0.18, size=n).astype(np.float32)
    local_latency = 0.18 + 0.95 * cpu + 0.35 * memory + local_noise
    cloud_latency = 1.05 + 0.45 * rng.random(n) + 0.35 * cpu + cloud_noise

    local_latency = np.clip(local_latency, 0.15, 2.5).astype(np.float32)
    cloud_latency = np.clip(cloud_latency, 0.55, 3.5).astype(np.float32)

    return np.column_stack(
        [
            cpu,
            memory,
            battery,
            budget,
            local_latency,
            cloud_latency,
        ]
    ).astype(np.float32)


def make_system_features(
    *,
    cpu_percent: float,
    memory_percent: float,
    battery_percent: float,
    budget_remaining: float,
    estimated_local_latency: float,
    estimated_cloud_latency: float,
) -> np.ndarray:
    """Build a single normalized serving-state vector."""

    return np.array(
        [
            np.clip(cpu_percent, 0.0, 1.0),
            np.clip(memory_percent, 0.0, 1.0),
            np.clip(battery_percent, 0.0, 1.0),
            np.clip(budget_remaining, 0.0, 1.0),
            max(0.0, estimated_local_latency),
            max(0.0, estimated_cloud_latency),
        ],
        dtype=np.float32,
    )


def augment_offline_split(
    split: OfflineSplit,
    *,
    scenarios_per_prompt: int = 5,
    seed: int = 184,
    remote_cost_per_request: float = DEFAULT_REMOTE_COST_PER_REQUEST,
) -> RealtimeAugmentedSplit:
    """Expand each prompt into multiple serving scenarios."""

    system_features = generate_serving_states(
        len(split.features),
        scenarios_per_prompt=scenarios_per_prompt,
        seed=seed,
    )
    prompt_index = np.repeat(np.arange(len(split.features)), scenarios_per_prompt)
    qwen_features = split.features[prompt_index].astype(np.float32)
    contexts = np.concatenate([qwen_features, system_features], axis=1).astype(np.float32)

    return RealtimeAugmentedSplit(
        contexts=contexts,
        qwen_features=qwen_features,
        system_features=system_features,
        score_small=split.score_small[prompt_index].astype(np.float32),
        score_large=split.score_large[prompt_index].astype(np.float32),
        small_failure=split.small_failure[prompt_index].astype(np.int64),
        prompt_index=prompt_index.astype(np.int64),
        prompts=split.prompts[prompt_index],
        remote_cost_per_request=float(remote_cost_per_request),
    )


def realtime_utility_matrix(
    split: RealtimeAugmentedSplit,
    sla: SLAMode,
) -> np.ndarray:
    """Return utility matrix with columns [local, cloud]."""

    local_latency = split.system_features[:, SYSTEM_FEATURE_NAMES.index("estimated_local_latency")]
    cloud_latency = split.system_features[:, SYSTEM_FEATURE_NAMES.index("estimated_cloud_latency")]
    budget_remaining = split.system_features[:, SYSTEM_FEATURE_NAMES.index("budget_remaining")]

    cost_local = np.zeros_like(split.score_small, dtype=np.float32)
    cost_cloud = split.remote_cost_per_request / np.maximum(budget_remaining, 0.05)

    utility_local = (
        split.score_small
        - sla.alpha_cost * cost_local
        - sla.beta_latency * local_latency
    )
    utility_cloud = (
        split.score_large
        - sla.alpha_cost * cost_cloud
        - sla.beta_latency * cloud_latency
    )
    return np.column_stack([utility_local, utility_cloud]).astype(np.float32)


def realtime_oracle_actions(split: RealtimeAugmentedSplit, sla: SLAMode) -> np.ndarray:
    return np.argmax(realtime_utility_matrix(split, sla), axis=1).astype(np.int64)


def realtime_route_metrics(
    *,
    mode: str,
    policy: str,
    split: RealtimeAugmentedSplit,
    actions: np.ndarray,
    sla: SLAMode,
) -> dict[str, float | int | str]:
    """Evaluate real-time aligned route actions."""

    actions = actions.astype(np.int64)
    utilities = realtime_utility_matrix(split, sla)
    selected_utility = utilities[np.arange(len(actions)), actions]
    oracle = np.argmax(utilities, axis=1).astype(np.int64)

    quality = np.where(actions == LARGE, split.score_large, split.score_small)
    budget_remaining = split.system_features[:, SYSTEM_FEATURE_NAMES.index("budget_remaining")]
    effective_cloud_cost = split.remote_cost_per_request / np.maximum(budget_remaining, 0.05)
    cost = np.where(actions == LARGE, effective_cloud_cost, 0.0)
    local_latency = split.system_features[:, SYSTEM_FEATURE_NAMES.index("estimated_local_latency")]
    cloud_latency = split.system_features[:, SYSTEM_FEATURE_NAMES.index("estimated_cloud_latency")]
    latency = np.where(actions == LARGE, cloud_latency, local_latency)

    tn = int(((oracle == SMALL) & (actions == SMALL)).sum())
    fp = int(((oracle == SMALL) & (actions == LARGE)).sum())
    fn = int(((oracle == LARGE) & (actions == SMALL)).sum())
    tp = int(((oracle == LARGE) & (actions == LARGE)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "mode": mode,
        "policy": policy,
        "num_examples": int(len(actions)),
        "avg_utility": float(selected_utility.mean()),
        "avg_quality": float(quality.mean()),
        "avg_cost": float(cost.mean()),
        "avg_latency": float(latency.mean()),
        "cloud_route_rate": float((actions == LARGE).mean()),
        "routing_accuracy_vs_oracle": float((actions == oracle).mean()),
        "budget_penalty": float(cost.mean()),
        "avg_effective_cloud_cost": float(effective_cloud_cost.mean()),
        "precision_cloud_vs_oracle": float(precision),
        "recall_cloud_vs_oracle": float(recall),
        "f1_cloud_vs_oracle": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
