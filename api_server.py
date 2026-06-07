"""FastAPI entry point for the production-style LLM routing gateway."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_router.gateway import (
    DEFAULT_CLOUD_MODEL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_REMOTE_COST_PER_REQUEST,
    RoutingGateway,
)


app = FastAPI(title="Adaptive LLM Routing Gateway", version="0.1.0")
gateway = RoutingGateway()


class RouteRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    sla_mode: str = "balanced"
    session_budget: float = Field(2.0, gt=0)
    spent_so_far: float = Field(0.0, ge=0)
    estimated_cloud_cost: float = Field(DEFAULT_REMOTE_COST_PER_REQUEST, ge=0)


class ChatRequest(RouteRequest):
    call_backend: bool = True
    local_model: str = DEFAULT_LOCAL_MODEL
    cloud_model: str = DEFAULT_CLOUD_MODEL


def request_data(request: BaseModel) -> dict[str, Any]:
    """Support both Pydantic v2 and v1."""

    if hasattr(request, "model_dump"):
        return request.model_dump()
    return request.dict()


@app.get("/health")
def health() -> dict[str, Any]:
    return gateway.health()


@app.post("/route")
def route(request: RouteRequest) -> dict[str, Any]:
    try:
        decision = gateway.route(**request_data(request))
        gateway.append_log(
            decision=decision,
            backend_called=False,
            success=True,
        )
        return decision.public_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    try:
        decision, backend_result = gateway.chat(**request_data(request))
        backend_called = backend_result is not None
        success = backend_result.success if backend_result is not None else True
        error = backend_result.error if backend_result is not None else None
        gateway.append_log(
            decision=decision,
            backend_called=backend_called,
            success=success,
            error=error,
        )
        response: dict[str, Any] = {
            **decision.public_dict(),
            "backend_called": backend_called,
            "backend": backend_result.to_dict() if backend_result is not None else None,
        }
        if backend_result is not None:
            response["response_text"] = backend_result.response_text
            response["message"] = backend_result.message
        else:
            response["response_text"] = None
            response["message"] = "Backend call skipped; returning route metadata only."
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
