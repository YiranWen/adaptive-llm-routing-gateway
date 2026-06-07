"""Production-style real-time routing gateway logic."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from llm_router.backends import BackendResult, call_ollama, call_openai
from llm_router.config import ModelConfig
from llm_router.feature_extractor import MechanisticFeatureExtractor
from llm_router.realtime import (
    DEFAULT_REMOTE_COST_PER_REQUEST,
    LARGE,
    SMALL,
    make_system_features,
)
from llm_router.utilities import SLA_MODES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "utility_prediction_realtime.pkl"
DEFAULT_FEATURE_CACHE = ROOT / "data" / "processed" / "activation_features_train_512.npz"
DEFAULT_LOG_PATH = ROOT / "logs" / "routing_requests.jsonl"
DEFAULT_LOCAL_MODEL = "qwen2.5:0.5b"
DEFAULT_CLOUD_MODEL = "gpt-4o-mini"
DEFAULT_LOCAL_LATENCY = 0.8
DEFAULT_CLOUD_LATENCY = 1.4
CLOUD_MARGINS = {
    "quality_first": 0.07,
    "balanced": 0.04,
    "cost_sensitive": 0.02,
}
CODE_KEYWORDS = {
    "code",
    "function",
    "python",
    "javascript",
    "java",
    "c++",
    "sql",
    "bug",
    "error",
    "stacktrace",
}
REASONING_KEYWORDS = {
    "prove",
    "derive",
    "analyze",
    "compare",
    "optimize",
    "debug",
    "implement",
    "explain in detail",
}


@dataclass(frozen=True)
class RouteDecision:
    """Route decision returned by the gateway."""

    route: str
    predicted_utility_local: float
    predicted_utility_cloud: float
    estimated_cost: float
    estimated_latency: float
    feature_source: str
    explanation: str
    prompt_hash: str
    sla_mode: str
    raw_utility_gap: float
    cloud_margin: float

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {
            key: data[key]
            for key in [
                "route",
                "predicted_utility_local",
                "predicted_utility_cloud",
                "estimated_cost",
                "estimated_latency",
                "feature_source",
                "explanation",
            ]
        }


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def cached_feature_index(prompt: str, num_features: int) -> int:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_features


def prompt_complexity_flags(prompt: str) -> dict[str, Any]:
    normalized = prompt.lower().strip()
    words = re.findall(r"\b[\w'+-]+\b", normalized)
    contains_math_symbols = bool(
        re.search(r"[=+\-*/^∑√≤≥<>]|\b(solve|equation|integral|derivative)\b", normalized)
    )
    contains_code_keywords = any(keyword in normalized for keyword in CODE_KEYWORDS)
    contains_reasoning_keywords = any(keyword in normalized for keyword in REASONING_KEYWORDS)
    contains_simple_factual_pattern = bool(
        re.match(
            r"^(is|are|was|were|what is|who is|when is|where is|what are|who was|when was)\b",
            normalized,
        )
    )
    is_short = len(words) <= 12
    is_long = len(words) >= 45
    is_simple_factual_short = (
        contains_simple_factual_pattern
        and is_short
        and not contains_math_symbols
        and not contains_code_keywords
        and not contains_reasoning_keywords
    )
    is_complex = (
        contains_math_symbols
        or contains_code_keywords
        or contains_reasoning_keywords
        or is_long
    )
    return {
        "prompt_length_words": len(words),
        "contains_math_symbols": contains_math_symbols,
        "contains_code_keywords": contains_code_keywords,
        "contains_reasoning_keywords": contains_reasoning_keywords,
        "contains_simple_factual_pattern": contains_simple_factual_pattern,
        "is_simple_factual_short": is_simple_factual_short,
        "is_long": is_long,
        "is_complex": is_complex,
    }


def final_cloud_margin(sla_mode: str, flags: dict[str, Any]) -> float:
    margin = CLOUD_MARGINS[sla_mode]
    if flags["is_simple_factual_short"]:
        margin += 0.05
    if flags["is_complex"]:
        margin -= 0.03
    return max(0.0, margin)


class RoutingGateway:
    """Real-time local/cloud routing gateway."""

    def __init__(
        self,
        *,
        model_path: Path = DEFAULT_MODEL_PATH,
        feature_cache_path: Path = DEFAULT_FEATURE_CACHE,
        log_path: Path = DEFAULT_LOG_PATH,
        feature_mode: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.feature_cache_path = feature_cache_path
        self.log_path = log_path
        self.feature_mode = feature_mode or os.environ.get("LLM_ROUTER_FEATURE_MODE", "cached")
        self._bundle: dict[str, Any] | None = None
        self._features: np.ndarray | None = None
        self._extractor: MechanisticFeatureExtractor | None = None

    @property
    def bundle(self) -> dict[str, Any]:
        if self._bundle is None:
            with self.model_path.open("rb") as file:
                self._bundle = pickle.load(file)
        return self._bundle

    @property
    def cached_features(self) -> np.ndarray:
        if self._features is None:
            cache = np.load(self.feature_cache_path, allow_pickle=True)
            self._features = cache["features"].astype(np.float32)
        return self._features

    @property
    def extractor(self) -> MechanisticFeatureExtractor:
        if self._extractor is None:
            self._extractor = MechanisticFeatureExtractor(
                ModelConfig(max_length=256, device="auto", layer_offset=-2)
            )
        return self._extractor

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": self.model_path.exists(),
            "feature_cache_loaded": self.feature_cache_path.exists(),
            "feature_mode": self.feature_mode,
        }

    def prompt_feature(self, prompt: str) -> tuple[np.ndarray, str]:
        if self.feature_mode == "qwen":
            try:
                result = self.extractor.extract(prompt)
                return result.features.astype(np.float32), "real Qwen extraction"
            except Exception:
                pass
        features = self.cached_features
        index = cached_feature_index(prompt, len(features))
        return features[index], "cached fallback feature"

    def system_features(
        self,
        *,
        session_budget: float,
        spent_so_far: float,
    ) -> tuple[np.ndarray, float, float, float]:
        import psutil

        cpu = psutil.cpu_percent(interval=0.05) / 100.0
        memory = psutil.virtual_memory().percent / 100.0
        battery_info = psutil.sensors_battery()
        battery = 1.0 if battery_info is None else battery_info.percent / 100.0
        remaining_dollars = max(session_budget - spent_so_far, 0.0)
        budget_remaining = remaining_dollars / session_budget if session_budget > 0 else 0.0
        local_latency = max(0.15, 0.18 + 0.95 * cpu + 0.35 * memory)
        cloud_latency = DEFAULT_CLOUD_LATENCY
        return (
            make_system_features(
                cpu_percent=cpu,
                memory_percent=memory,
                battery_percent=battery,
                budget_remaining=budget_remaining,
                estimated_local_latency=local_latency,
                estimated_cloud_latency=cloud_latency,
            ),
            budget_remaining,
            local_latency,
            cloud_latency,
        )

    def route(
        self,
        *,
        prompt: str,
        sla_mode: str,
        session_budget: float,
        spent_so_far: float,
        estimated_cloud_cost: float,
    ) -> RouteDecision:
        if sla_mode not in SLA_MODES:
            raise ValueError(f"unknown SLA mode: {sla_mode}")

        prompt_feature, feature_source = self.prompt_feature(prompt)
        system_features, budget_remaining, local_latency, cloud_latency = self.system_features(
            session_budget=session_budget,
            spent_so_far=spent_so_far,
        )
        context = np.concatenate([prompt_feature, system_features], axis=0).reshape(1, -1)
        model = self.bundle["models"][sla_mode]
        predicted = model.predict(context).reshape(-1).astype(float)

        sla = SLA_MODES[sla_mode]
        default_cost = self.bundle["scenario"].get(
            "remote_cost_per_request",
            DEFAULT_REMOTE_COST_PER_REQUEST,
        )
        default_effective_cost = default_cost / max(budget_remaining, 0.05)
        actual_effective_cost = estimated_cloud_cost / max(budget_remaining, 0.05)
        adjusted = predicted.copy()
        adjusted[LARGE] -= sla.alpha_cost * (actual_effective_cost - default_effective_cost)

        flags = prompt_complexity_flags(prompt)
        margin = final_cloud_margin(sla_mode, flags)
        raw_gap = float(adjusted[LARGE] - adjusted[SMALL])
        action = LARGE if raw_gap >= margin else SMALL
        route = "cloud" if action == LARGE else "local"
        estimated_cost = actual_effective_cost if action == LARGE else 0.0
        estimated_latency = cloud_latency if action == LARGE else local_latency
        explanation = (
            "Cloud selected because predicted utility gain exceeded the SLA margin."
            if action == LARGE
            else "Local selected because cloud utility gain was too small to justify escalation."
        )

        return RouteDecision(
            route=route,
            predicted_utility_local=float(adjusted[SMALL]),
            predicted_utility_cloud=float(adjusted[LARGE]),
            estimated_cost=float(estimated_cost),
            estimated_latency=float(estimated_latency),
            feature_source=feature_source,
            explanation=explanation,
            prompt_hash=prompt_hash(prompt),
            sla_mode=sla_mode,
            raw_utility_gap=raw_gap,
            cloud_margin=float(margin),
        )

    def chat(
        self,
        *,
        prompt: str,
        sla_mode: str,
        session_budget: float,
        spent_so_far: float,
        estimated_cloud_cost: float,
        local_model: str = DEFAULT_LOCAL_MODEL,
        cloud_model: str = DEFAULT_CLOUD_MODEL,
        call_backend: bool = True,
    ) -> tuple[RouteDecision, BackendResult | None]:
        decision = self.route(
            prompt=prompt,
            sla_mode=sla_mode,
            session_budget=session_budget,
            spent_so_far=spent_so_far,
            estimated_cloud_cost=estimated_cloud_cost,
        )
        if not call_backend:
            return decision, None
        if decision.route == "local":
            return decision, call_ollama(prompt, local_model)
        return decision, call_openai(prompt, cloud_model)

    def append_log(
        self,
        *,
        decision: RouteDecision,
        backend_called: bool,
        success: bool,
        error: str | None = None,
    ) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_hash": decision.prompt_hash,
            "sla_mode": decision.sla_mode,
            "route": decision.route,
            "predicted_utility_local": decision.predicted_utility_local,
            "predicted_utility_cloud": decision.predicted_utility_cloud,
            "estimated_cost": decision.estimated_cost,
            "estimated_latency": decision.estimated_latency,
            "backend_called": backend_called,
            "success": success,
            "error": error,
        }
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
