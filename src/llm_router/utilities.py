"""Utility and metric helpers for LLM routing.

Utility is the deployment objective used by the final project:

utility = quality - alpha * cost - beta * latency
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llm_router.offline_data import OfflineSplit


SMALL = 0
LARGE = 1


@dataclass(frozen=True)
class SLAMode:
    """Utility weights for one SLA preference mode."""

    name: str
    alpha_cost: float
    beta_latency: float


SLA_MODES: dict[str, SLAMode] = {
    "quality_first": SLAMode("quality_first", alpha_cost=0.05, beta_latency=0.02),
    "balanced": SLAMode("balanced", alpha_cost=0.20, beta_latency=0.05),
    "cost_sensitive": SLAMode("cost_sensitive", alpha_cost=0.45, beta_latency=0.08),
}


def arm_utilities(split: OfflineSplit, sla: SLAMode) -> np.ndarray:
    """Return utility matrix with columns [small, large]."""

    small_cost = np.zeros_like(split.score_small, dtype=np.float32)
    large_cost = np.ones_like(split.score_large, dtype=np.float32)
    utility_small = (
        split.score_small
        - sla.alpha_cost * small_cost
        - sla.beta_latency * split.latency_small
    )
    utility_large = (
        split.score_large
        - sla.alpha_cost * large_cost
        - sla.beta_latency * split.latency_large
    )
    return np.column_stack([utility_small, utility_large]).astype(np.float32)


def oracle_actions(split: OfflineSplit, sla: SLAMode) -> np.ndarray:
    """Choose the arm with the best utility for each example."""

    return np.argmax(arm_utilities(split, sla), axis=1).astype(np.int64)


def route_quality(split: OfflineSplit, actions: np.ndarray) -> np.ndarray:
    return np.where(actions == LARGE, split.score_large, split.score_small)


def route_cost(actions: np.ndarray) -> np.ndarray:
    return (actions == LARGE).astype(np.float32)


def route_latency(split: OfflineSplit, actions: np.ndarray) -> np.ndarray:
    return np.where(actions == LARGE, split.latency_large, split.latency_small)


def route_utilities(split: OfflineSplit, actions: np.ndarray, sla: SLAMode) -> np.ndarray:
    utilities = arm_utilities(split, sla)
    return utilities[np.arange(len(actions)), actions]


def confusion_counts(y_true: np.ndarray, actions: np.ndarray) -> dict[str, int]:
    """Confusion matrix for the large/escalate action as positive."""

    y_pred = (actions == LARGE).astype(np.int64)
    y_true = y_true.astype(np.int64)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def policy_metrics(
    *,
    policy: str,
    mode: str,
    split: OfflineSplit,
    actions: np.ndarray,
    sla: SLAMode,
) -> dict[str, float | int | str]:
    """Compute report metrics for a policy on one split and SLA mode."""

    actions = actions.astype(np.int64)
    oracle = oracle_actions(split, sla)
    utilities = route_utilities(split, actions, sla)
    counts = confusion_counts(split.small_failure, actions)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "mode": mode,
        "policy": policy,
        "num_examples": int(len(actions)),
        "routing_accuracy": float((actions == oracle).mean()),
        "classification_accuracy": float((actions == split.small_failure).mean()),
        "precision_large": float(precision),
        "recall_large": float(recall),
        "f1_large": float(f1),
        "avg_utility": float(utilities.mean()),
        "total_utility": float(utilities.sum()),
        "avg_quality": float(route_quality(split, actions).mean()),
        "avg_cost": float(route_cost(actions).mean()),
        "avg_latency": float(route_latency(split, actions).mean()),
        "large_route_rate": float((actions == LARGE).mean()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
