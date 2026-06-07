"""Streamlit demo for the real-time aligned LLM routing gateway."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_router.config import ModelConfig
from llm_router.feature_extractor import MechanisticFeatureExtractor
from llm_router.neural_runtime import (
    DEFAULT_NEURAL_MODEL_PATH,
    default_cloud_margin,
    load_neural_router_bundle,
    predict_neural_route,
)
from llm_router.realtime import (
    DEFAULT_REMOTE_COST_PER_REQUEST,
    LARGE,
    make_system_features,
)
from llm_router.utilities import SLA_MODES


MODEL_PATH = DEFAULT_NEURAL_MODEL_PATH
FALLBACK_FEATURES_PATH = ROOT / "data" / "processed" / "activation_features_train_512.npz"
DEFAULT_LOCAL_LATENCY = 0.8
DEFAULT_CLOUD_LATENCY = 1.4
DEFAULT_QWEN_MAX_LENGTH = 128
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


st.set_page_config(page_title="Adaptive LLM Routing Gateway", layout="wide")


@st.cache_resource
def load_router_bundle() -> dict[str, object]:
    return load_neural_router_bundle(MODEL_PATH)


@st.cache_resource
def load_feature_extractor(max_length: int) -> MechanisticFeatureExtractor:
    return MechanisticFeatureExtractor(
        ModelConfig(max_length=max_length, device="auto", layer_offset=-2)
    )


@st.cache_data
def load_cached_features() -> np.ndarray:
    cache = np.load(FALLBACK_FEATURES_PATH, allow_pickle=True)
    return cache["features"].astype(np.float32)


def get_system_metrics() -> tuple[float, float, float]:
    import psutil

    cpu = psutil.cpu_percent(interval=0.1) / 100.0
    memory = psutil.virtual_memory().percent / 100.0
    battery_info = psutil.sensors_battery()
    battery = 1.0 if battery_info is None else battery_info.percent / 100.0
    return cpu, memory, battery


def cached_prompt_feature(prompt: str) -> np.ndarray:
    features = load_cached_features()
    index = abs(hash(prompt)) % len(features)
    return features[index]


def prompt_complexity_flags(prompt: str) -> dict[str, object]:
    normalized = prompt.lower().strip()
    words = re.findall(r"\b[\w'+-]+\b", normalized)
    contains_math_symbols = bool(
        re.search(r"[=+\-*/^∑√≤≥<>]|\b(solve|equation|integral|derivative)\b", normalized)
    )
    contains_simple_arithmetic_pattern = bool(
        re.match(
            r"^\s*\d+(\.\d+)?\s*[+\-*/]\s*\d+(\.\d+)?\s*(=|\?|$|equal|equals|is)?",
            normalized,
        )
    )
    contains_greeting_pattern = bool(
        re.match(r"^(hi|hello|hey|how are you|good morning|good afternoon|good evening)\b", normalized)
    )
    contains_code_keywords = any(keyword in normalized for keyword in CODE_KEYWORDS)
    contains_reasoning_keywords = any(keyword in normalized for keyword in REASONING_KEYWORDS)
    contains_simple_factual_pattern = bool(
        re.match(r"^(is|are|was|were|what is|who is|when is|where is|what are|who was|when was)\b", normalized)
    )
    is_short = len(words) <= 12
    is_long = len(words) >= 45
    is_simple_factual_short = (
        (
            contains_simple_factual_pattern
            or contains_simple_arithmetic_pattern
            or contains_greeting_pattern
        )
        and is_short
        and not contains_code_keywords
        and not contains_reasoning_keywords
    )
    is_complex = (
        (contains_math_symbols and not contains_simple_arithmetic_pattern)
        or contains_code_keywords
        or contains_reasoning_keywords
        or is_long
    )
    return {
        "prompt_length_words": len(words),
        "contains_math_symbols": contains_math_symbols,
        "contains_simple_arithmetic_pattern": contains_simple_arithmetic_pattern,
        "contains_greeting_pattern": contains_greeting_pattern,
        "contains_code_keywords": contains_code_keywords,
        "contains_reasoning_keywords": contains_reasoning_keywords,
        "contains_simple_factual_pattern": contains_simple_factual_pattern,
        "is_simple_factual_short": is_simple_factual_short,
        "is_long": is_long,
        "is_complex": is_complex,
    }


@st.cache_data(show_spinner=False, max_entries=64)
def extract_real_qwen_feature_cached(prompt: str, max_length: int) -> tuple[np.ndarray, float]:
    result = load_feature_extractor(max_length).extract(prompt)
    return result.features.astype(np.float32), result.elapsed_ms / 1000.0


def extract_prompt_feature(
    prompt: str,
    *,
    use_cached_fallback: bool,
    qwen_max_length: int,
) -> tuple[np.ndarray, str, float]:
    if use_cached_fallback:
        return cached_prompt_feature(prompt), "cached fallback", 0.0

    started = time.perf_counter()
    try:
        feature, model_forward_s = extract_real_qwen_feature_cached(prompt, qwen_max_length)
        elapsed = time.perf_counter() - started
        if elapsed < 0.05:
            return feature, "Qwen activation (prompt cache hit)", elapsed
        return feature, "Qwen activation", model_forward_s
    except Exception as exc:
        st.warning(f"Qwen extraction failed; using cached fallback. Error: {exc}")
        elapsed = time.perf_counter() - started
        return cached_prompt_feature(prompt), "cached fallback after extractor error", elapsed


def update_latency_average(key: str, observed: float) -> None:
    previous = st.session_state.get(key, observed)
    st.session_state[key] = 0.8 * previous + 0.2 * observed


def call_ollama(prompt: str, model_name: str) -> tuple[str, float]:
    started = time.perf_counter()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    return response.json().get("response", ""), elapsed


def call_openai(prompt: str, api_key: str, model_name: str) -> tuple[str, float]:
    started = time.perf_counter()
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=120,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    content = response.json()["choices"][0]["message"]["content"]
    return content, elapsed


def main() -> None:
    st.title("Real-Time LLM Routing Gateway")
    st.caption("Final demo path: Qwen features + live system state -> Neural Utility Router")

    if not MODEL_PATH.exists():
        st.error(
            "Router model artifact not found. Run: "
            "`python scripts/evaluate_neural_utility_router.py`"
        )
        return

    if "budget_remaining_dollars" not in st.session_state:
        st.session_state.budget_remaining_dollars = 1.0
    if "local_latency_avg" not in st.session_state:
        st.session_state.local_latency_avg = DEFAULT_LOCAL_LATENCY
    if "cloud_latency_avg" not in st.session_state:
        st.session_state.cloud_latency_avg = DEFAULT_CLOUD_LATENCY

    bundle = load_router_bundle()
    neural_meta = bundle.get("neural_meta", {})

    with st.sidebar:
        st.header("Routing Controls")
        sla_name = st.selectbox("SLA mode", list(SLA_MODES), index=1)
        session_budget = st.number_input(
            "Session budget ($)",
            min_value=0.01,
            value=1.00,
            step=0.10,
        )
        if st.button("Reset session budget"):
            st.session_state.budget_remaining_dollars = session_budget
        st.session_state.budget_remaining_dollars = min(
            st.session_state.budget_remaining_dollars,
            session_budget,
        )
        cloud_cost = st.number_input(
            "Estimated cloud cost/request ($)",
            min_value=0.0,
            value=DEFAULT_REMOTE_COST_PER_REQUEST,
            step=0.01,
        )
        use_real_qwen = st.checkbox(
            "Use real Qwen feature extraction",
            value=False,
            help="Slower but encodes the exact typed prompt. Default uses cached fallback for speed.",
        )
        qwen_max_length = st.slider(
            "Qwen max tokens",
            min_value=64,
            max_value=256,
            value=DEFAULT_QWEN_MAX_LENGTH,
            step=32,
            help="Lower values make real Qwen extraction faster. Cached fallback ignores this.",
        )
        if st.button("Preload Qwen model"):
            with st.spinner("Loading Qwen feature extractor once..."):
                load_feature_extractor(qwen_max_length).load()
            st.success("Qwen feature extractor is loaded and cached.")
        override_system_metrics = st.checkbox(
            "Override live system metrics for testing",
            value=False,
        )
        if override_system_metrics:
            st.caption("Testing overrides")
            override_cpu = st.slider("CPU percent", 0, 100, 50)
            override_memory = st.slider("Memory percent", 0, 100, 50)
            override_battery = st.slider("Battery percent", 0, 100, 80)
            override_local_latency = st.slider(
                "Estimated local latency",
                0.1,
                10.0,
                float(DEFAULT_LOCAL_LATENCY),
                0.1,
            )
            override_cloud_latency = st.slider(
                "Estimated cloud latency",
                0.1,
                10.0,
                float(DEFAULT_CLOUD_LATENCY),
                0.1,
            )
            override_budget_remaining = st.slider(
                "Budget remaining",
                0.0,
                1.0,
                1.0,
                0.05,
            )
        st.divider()
        st.header("Optional Generation")
        generate_local = st.checkbox("Try local Ollama generation", value=False)
        ollama_model = st.text_input("Ollama model", value="qwen2.5:0.5b")
        generate_cloud = st.checkbox("Actually call cloud model", value=False)
        cloud_model = st.text_input("Cloud model", value="gpt-4o-mini")
        api_key = st.text_input(
            "Cloud API key",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
        )

        st.divider()
        st.header("Router Model")
        st.caption("Neural Utility Router")
        st.write(
            {
                "hidden_dims": neural_meta.get("hidden_dims"),
                "dropout": neural_meta.get("dropout"),
                "selected_epoch": neural_meta.get("best_epoch"),
                "base_cloud_margin": default_cloud_margin(bundle, sla_name),
            }
        )

    prompt = st.text_area(
        "Prompt",
        value="Explain the difference between supervised learning and reinforcement learning in simple terms.",
        height=150,
    )

    if override_system_metrics:
        cpu = override_cpu / 100.0
        memory = override_memory / 100.0
        battery = override_battery / 100.0
        budget_remaining_norm = override_budget_remaining
        estimated_local_latency = override_local_latency
        estimated_cloud_latency = override_cloud_latency
    else:
        cpu, memory, battery = get_system_metrics()
        budget_remaining_norm = (
            st.session_state.budget_remaining_dollars / session_budget
            if session_budget > 0
            else 0.0
        )
        estimated_local_latency = max(
            0.15,
            0.18 + 0.95 * cpu + 0.35 * memory,
        )
        estimated_cloud_latency = st.session_state.cloud_latency_avg
    system_features = make_system_features(
        cpu_percent=cpu,
        memory_percent=memory,
        battery_percent=battery,
        budget_remaining=budget_remaining_norm,
        estimated_local_latency=estimated_local_latency,
        estimated_cloud_latency=estimated_cloud_latency,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CPU", f"{cpu:.0%}")
    col2.metric("Memory", f"{memory:.0%}")
    col3.metric("Battery", f"{battery:.0%}")
    col4.metric(
        "Budget left",
        f"{budget_remaining_norm:.0%}" if override_system_metrics else f"${st.session_state.budget_remaining_dollars:.2f}",
    )

    if st.button("Route Prompt", type="primary"):
        prompt_feature, feature_source, extraction_s = extract_prompt_feature(
            prompt,
            use_cached_fallback=not use_real_qwen,
            qwen_max_length=qwen_max_length,
        )
        flags = prompt_complexity_flags(prompt)
        prediction = predict_neural_route(
            bundle=bundle,
            prompt_feature=prompt_feature,
            system_features=system_features,
            sla_mode=sla_name,
            estimated_cloud_cost=float(cloud_cost),
            budget_remaining=float(budget_remaining_norm),
            prompt_flags=flags,
        )
        route = prediction.route
        actual_effective_cost = cloud_cost / max(budget_remaining_norm, 0.05)
        estimated_cost = actual_effective_cost if prediction.action == LARGE else 0.0
        estimated_latency = (
            estimated_cloud_latency
            if prediction.action == LARGE
            else estimated_local_latency
        )

        st.subheader("Routing Decision")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Selected route", route.upper())
        r2.metric("Predicted local utility", f"{prediction.predicted_utility_local:.3f}")
        r3.metric("Predicted cloud utility", f"{prediction.predicted_utility_cloud:.3f}")
        r4.metric("Feature source", feature_source)

        if "cached fallback" in feature_source:
            st.warning(
                "Cached fallback does not semantically encode this exact prompt, "
                "so routing may be approximate."
            )

        st.write(
            {
                "sla_mode": sla_name,
                "system_metrics_source": "testing override" if override_system_metrics else "live psutil/session state",
                "estimated_cost": round(float(estimated_cost), 4),
                "estimated_latency": round(float(estimated_latency), 4),
                "budget_remaining_normalized": round(float(budget_remaining_norm), 4),
                "qwen_or_fallback_feature_seconds": round(float(extraction_s), 4),
                "router_model": "Neural Utility Router",
                "hidden_dims": neural_meta.get("hidden_dims"),
                "best_epoch": neural_meta.get("best_epoch"),
            }
        )

        if prediction.action == LARGE:
            st.info(
                "Cloud selected because predicted cloud utility exceeded local utility by the required SLA margin."
            )
        elif prediction.raw_utility_gap <= 0:
            st.info(
                "Local selected because predicted local utility is greater than or equal to predicted cloud utility."
            )
        else:
            st.info(
                "Local selected because cloud utility gain was too small to justify escalation."
            )

        with st.expander("Debug routing details", expanded=False):
            st.markdown("Exact 6D system feature vector passed into the router:")
            st.table(
                {
                    "feature": [
                        "cpu_normalized",
                        "memory_normalized",
                        "battery_normalized",
                        "budget_remaining",
                        "estimated_local_latency",
                        "estimated_cloud_latency",
                    ],
                    "value": [round(float(value), 6) for value in system_features.tolist()],
                }
            )
            st.json(
                {
                    "predicted_local_utility": round(prediction.predicted_utility_local, 6),
                    "predicted_cloud_utility": round(prediction.predicted_utility_cloud, 6),
                    "raw_utility_gap": round(prediction.raw_utility_gap, 6),
                    "base_cloud_margin": default_cloud_margin(bundle, sla_name),
                    "final_cloud_margin": round(prediction.cloud_margin, 6),
                    "prompt_complexity_flags": flags,
                    "final_route": route,
                    "feature_source": feature_source,
                }
            )

        if route == "cloud" and st.button("Apply cloud cost to budget"):
            st.session_state.budget_remaining_dollars = max(
                0.0,
                st.session_state.budget_remaining_dollars - cloud_cost,
            )
            st.rerun()

        if route == "local" and generate_local:
            try:
                output, elapsed = call_ollama(prompt, ollama_model)
                update_latency_average("local_latency_avg", elapsed)
                st.subheader("Local response")
                st.write(output)
            except Exception as exc:
                st.warning(f"Ollama generation unavailable: {exc}")

        if route == "cloud" and generate_cloud:
            if not api_key:
                st.warning("Cloud route selected, but no API key was provided.")
            else:
                try:
                    output, elapsed = call_openai(prompt, api_key, cloud_model)
                    update_latency_average("cloud_latency_avg", elapsed)
                    st.session_state.budget_remaining_dollars = max(
                        0.0,
                        st.session_state.budget_remaining_dollars - cloud_cost,
                    )
                    st.subheader("Cloud response")
                    st.write(output)
                except Exception as exc:
                    st.warning(f"Cloud generation failed: {exc}")


if __name__ == "__main__":
    main()
