"""Runtime helpers for the Neural Utility Router."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm_router.neural_utility import (
    NeuralUtilityTrainingResult,
    predict_neural_utilities,
)
from llm_router.realtime import LARGE, SMALL
from llm_router.utilities import SLA_MODES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEURAL_MODEL_PATH = ROOT / "models" / "neural_utility_router.pkl"


@dataclass(frozen=True)
class NeuralRoutePrediction:
    """Predicted utilities and route decision for one request."""

    predicted_utility_local: float
    predicted_utility_cloud: float
    raw_utility_gap: float
    cloud_margin: float
    decision_reason: str
    action: int
    route: str


def load_neural_router_bundle(
    model_path: Path = DEFAULT_NEURAL_MODEL_PATH,
) -> dict[str, Any]:
    with model_path.open("rb") as file:
        return pickle.load(file)


def neural_context(
    *,
    prompt_feature: np.ndarray,
    system_features: np.ndarray,
    sla_mode: str,
) -> np.ndarray:
    """Build the 904D NN input: 896D prompt + 6D system + 2D SLA."""

    sla = SLA_MODES[sla_mode]
    sla_features = np.array([sla.alpha_cost, sla.beta_latency], dtype=np.float32)
    return np.concatenate(
        [
            prompt_feature.astype(np.float32),
            system_features.astype(np.float32),
            sla_features,
        ],
        axis=0,
    ).reshape(1, -1)


def neural_model_from_bundle(bundle: dict[str, Any]) -> NeuralUtilityTrainingResult:
    model = bundle.get("neural_model")
    if not isinstance(model, NeuralUtilityTrainingResult):
        raise TypeError("model bundle does not contain a NeuralUtilityTrainingResult")
    return model


def default_cloud_margin(bundle: dict[str, Any], sla_mode: str) -> float:
    margins = bundle.get("neural_meta", {}).get("cloud_margins", {})
    return float(margins.get(sla_mode, 0.0))


def adjusted_cloud_margin(
    *,
    base_margin: float,
    flags: dict[str, Any],
    system_features: np.ndarray | None = None,
) -> float:
    """Apply demo-time prompt heuristic to the dev-tuned margin."""

    margin = max(float(base_margin), 0.0)
    if flags.get("is_simple_factual_short"):
        margin += 0.05
        margin = max(margin, 0.15)
    if flags.get("is_complex"):
        margin -= 0.03

    if system_features is not None:
        system = np.asarray(system_features, dtype=np.float32).reshape(-1)
        if len(system) >= 6:
            cpu, memory, battery, _, local_latency, cloud_latency = system[:6]
            local_latency_disadvantage = float(local_latency - cloud_latency)
            local_system_pressure = (
                cpu >= 0.85
                or memory >= 0.90
                or battery <= 0.10
                or local_latency_disadvantage >= 0.25
            )
            if local_system_pressure:
                margin -= 0.08
                if flags.get("is_simple_factual_short"):
                    margin = min(margin, 0.01)
    return float(np.clip(margin, 0.0, 0.25))


def predict_neural_route(
    *,
    bundle: dict[str, Any],
    prompt_feature: np.ndarray,
    system_features: np.ndarray,
    sla_mode: str,
    estimated_cloud_cost: float,
    budget_remaining: float,
    prompt_flags: dict[str, Any],
) -> NeuralRoutePrediction:
    """Predict utilities, adjust cloud cost, and choose local/cloud route."""

    model = neural_model_from_bundle(bundle)
    context = neural_context(
        prompt_feature=prompt_feature,
        system_features=system_features,
        sla_mode=sla_mode,
    )
    predicted = predict_neural_utilities(model, context).reshape(-1).astype(float)

    sla = SLA_MODES[sla_mode]
    default_cost = float(
        bundle.get("scenario", {}).get("remote_cost_per_request", estimated_cloud_cost)
    )
    default_effective_cost = default_cost / max(float(budget_remaining), 0.05)
    actual_effective_cost = estimated_cloud_cost / max(float(budget_remaining), 0.05)
    predicted[LARGE] -= sla.alpha_cost * (actual_effective_cost - default_effective_cost)

    base_margin = default_cloud_margin(bundle, sla_mode)
    margin = adjusted_cloud_margin(
        base_margin=base_margin,
        flags=prompt_flags,
        system_features=system_features,
    )
    raw_gap = float(predicted[LARGE] - predicted[SMALL])
    if raw_gap <= 0.0:
        action = SMALL
        decision_reason = "local_utility_not_lower"
    elif raw_gap >= margin:
        action = LARGE
        decision_reason = "cloud_gain_exceeded_margin"
    else:
        action = SMALL
        decision_reason = "cloud_gain_below_margin"
    return NeuralRoutePrediction(
        predicted_utility_local=float(predicted[SMALL]),
        predicted_utility_cloud=float(predicted[LARGE]),
        raw_utility_gap=raw_gap,
        cloud_margin=margin,
        decision_reason=decision_reason,
        action=action,
        route="cloud" if action == LARGE else "local",
    )
