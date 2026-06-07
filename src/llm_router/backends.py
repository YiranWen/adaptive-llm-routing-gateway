"""Backend adapters for optional local/cloud generation."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


@dataclass(frozen=True)
class BackendResult:
    """Safe backend call result."""

    backend: str
    success: bool
    message: str
    response_text: str | None
    latency_s: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def call_ollama(prompt: str, model: str = "qwen2.5:0.5b") -> BackendResult:
    """Call local Ollama if available, otherwise return a safe failure."""

    started = time.perf_counter()
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        elapsed = time.perf_counter() - started
        text = response.json().get("response", "")
        return BackendResult(
            backend="ollama",
            success=True,
            message="Local Ollama generation succeeded.",
            response_text=text,
            latency_s=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return BackendResult(
            backend="ollama",
            success=False,
            message=(
                "Local generation is unavailable. Start Ollama at "
                "http://localhost:11434 and pull the requested model to enable it."
            ),
            response_text=None,
            latency_s=elapsed,
            error=str(exc),
        )


def call_openai(prompt: str, model: str = "gpt-4o-mini") -> BackendResult:
    """Call OpenAI chat completions only when OPENAI_API_KEY is present."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return BackendResult(
            backend="openai",
            success=False,
            message="Cloud generation is disabled because OPENAI_API_KEY is not set.",
            response_text=None,
            latency_s=None,
            error="missing OPENAI_API_KEY",
        )

    started = time.perf_counter()
    try:
        response = requests.post(
            OPENAI_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=120,
        )
        response.raise_for_status()
        elapsed = time.perf_counter() - started
        text = response.json()["choices"][0]["message"]["content"]
        return BackendResult(
            backend="openai",
            success=True,
            message="Cloud OpenAI generation succeeded.",
            response_text=text,
            latency_s=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return BackendResult(
            backend="openai",
            success=False,
            message="Cloud generation failed during the OpenAI request.",
            response_text=None,
            latency_s=elapsed,
            error=str(exc),
        )

