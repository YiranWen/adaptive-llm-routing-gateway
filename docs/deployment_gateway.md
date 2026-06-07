# Production-Style Routing Gateway

This project now has two user-facing layers:

- `app/streamlit_app.py` is the debug/demo dashboard. It is useful for inspecting utilities, margins, feature sources, budget state, and routing explanations interactively.
- `api_server.py` is the production-style FastAPI gateway. It exposes a small HTTP API that another app, agent framework, or chatbot service could call.

The gateway uses the same final routing formulation as the offline experiment:

```text
context = concat(Qwen activation feature, system/budget/latency features)
action = local or cloud
objective = utility = quality - alpha * cost - beta * latency
```

The selected runtime router is the Neural Utility Router saved at:

```text
models/neural_utility_router.pkl
```

## Files

| File | Role |
| --- | --- |
| `api_server.py` | FastAPI app with `/health`, `/route`, and `/chat` endpoints. |
| `src/llm_router/gateway.py` | Loads the Neural Utility Router, builds the context vector, predicts local/cloud utilities, selects the route, and writes request logs. |
| `src/llm_router/neural_runtime.py` | Shared runtime helper for Neural Utility Router prediction and cloud-margin routing. |
| `src/llm_router/backends.py` | Optional generation backends for local Ollama and OpenAI cloud calls with safe fallback behavior. |
| `logs/routing_requests.jsonl` | Append-only JSONL request log. |
| `app/streamlit_app.py` | Debug dashboard, not the production gateway. |

## Running The API

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the gateway:

```bash
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Check health:

```bash
curl http://localhost:8000/health
```

## `/route`

`POST /route` returns only the routing decision and metadata. It does not call a local or cloud model.

Example:

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain overfitting in one sentence.",
    "sla_mode": "balanced",
    "session_budget": 2.0,
    "spent_so_far": 0.0,
    "estimated_cloud_cost": 0.05
  }'
```

Response shape:

```json
{
  "route": "local",
  "predicted_utility_local": 0.84,
  "predicted_utility_cloud": 0.81,
  "estimated_cost": 0.0,
  "estimated_latency": 0.72,
  "feature_source": "cached fallback feature",
  "explanation": "Local selected because cloud utility gain was too small to justify escalation."
}
```

## `/chat`

`POST /chat` makes the same routing decision and can optionally call the selected backend.

By default, `call_backend` is `true`. Set it to `false` when you only want route metadata.

Local route:

- Uses Ollama at `http://localhost:11434/api/generate`.
- Default model is `qwen2.5:0.5b`.
- If Ollama is not running, the API still returns route metadata and a clear message that local generation is unavailable.

Cloud route:

- Uses OpenAI only when `OPENAI_API_KEY` is present.
- Default model is `gpt-4o-mini`.
- If no API key is set, the API still returns route metadata and a clear message that cloud generation is disabled.

Example:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is gravity?",
    "sla_mode": "cost_sensitive",
    "session_budget": 2.0,
    "spent_so_far": 0.0,
    "estimated_cloud_cost": 0.05,
    "call_backend": true
  }'
```

## Feature Extraction Mode

For speed, the API defaults to cached fallback prompt features. This keeps smoke tests and demos responsive.

To use real Qwen feature extraction in the gateway, start the server with:

```bash
LLM_ROUTER_FEATURE_MODE=qwen python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Real extraction uses `Qwen/Qwen2.5-0.5B-Instruct` through the project feature extractor. Cached fallback does not semantically encode the exact incoming prompt, so it is best treated as a fast demo/testing mode. When real Qwen extraction is enabled, the gateway keeps an in-memory prompt feature cache so repeated prompts do not repeat the forward pass.

## Logging

Every successful `/route` and `/chat` request appends one JSON object to:

```text
logs/routing_requests.jsonl
```

Logged fields include:

```text
timestamp, prompt_hash, sla_mode, route,
predicted_utility_local, predicted_utility_cloud,
estimated_cost, estimated_latency,
backend_called, success, error
```

The prompt itself is not logged; only a short SHA-256 hash is stored.

## Current Limitations

- The final router is trained on a 512/128 offline split with synthetic serving-state augmentation.
- The model is a Neural Utility Prediction router, not a fully live-trained reinforcement learning agent.
- The default API feature mode uses cached fallback features for responsiveness.
- Cost and latency are estimated rather than measured from every possible provider/model combination.
- Ollama must be installed and running for real local generation.
- OpenAI cloud generation requires `OPENAI_API_KEY`; tests do not require it.
- Streamlit remains a debug UI. The FastAPI server is the intended integration point for a production-style LLM serving stack.
