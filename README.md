# Adaptive LLM Routing Gateway

A system-aware LLM routing gateway that decides whether each prompt should be handled by a local model or a cloud model using prompt embeddings, runtime system metrics, budget, latency, and utility prediction.

Author: Yiran Wen  
GitHub: https://github.com/YiranWen

## Why This Project Exists

Cloud LLMs are usually high quality, but they can be expensive and slower because each request goes over the network. Local LLMs are cheaper, more private, and can be faster, but they are weaker on difficult prompts. This project builds a lightweight routing gateway that learns when cloud escalation is worth the extra cost and latency.

The core idea is to treat LLM routing as an AI infrastructure problem. The router does not answer the prompt itself. Instead, it decides which backend should answer.

## Architecture Overview

```text
User Prompt
→ Qwen2.5 Feature Extractor
→ 896D prompt embedding
+ CPU / memory / battery / budget / latency features
→ 902D context vector
→ Neural Utility Router / Utility Prediction Router
→ Local route or Cloud route
```

The router uses both semantic prompt information and live serving-state information:

- Prompt semantics: 896-dimensional Qwen2.5 activation feature.
- System state: CPU utilization, memory utilization, battery level, budget remaining, estimated local latency, and estimated cloud latency.
- Final router input: 902-dimensional context vector.

## Main Components

### Qwen Feature Extraction

`src/llm_router/feature_extractor.py` loads `Qwen/Qwen2.5-0.5B-Instruct` locally through HuggingFace Transformers. The model performs a prompt forward pass only; it does not generate text. A forward hook captures hidden states from the second-to-last decoder layer, then mean-pools non-padding token hidden states into one 896-dimensional prompt feature.

### MLP Classifier Baseline

`src/llm_router/mlp_router.py` implements a PyTorch MLP baseline for predicting whether the local model will fail. The classifier target is:

```text
local_failure = score_local < 0.7
```

Traditional ML evaluation artifacts for the classifier include accuracy, precision, recall, F1, confusion matrix, precision-recall curve, ROC-AUC curve, and learning curve outputs.

### Utility Prediction Router

`src/llm_router/utility_prediction.py` implements the Ridge Regression utility baseline. It predicts two continuous utility scores:

```text
Utility(local)
Utility(cloud)
```

The route with the higher predicted utility is selected. Utility is defined as:

```text
Utility = Quality - alpha * Cost - beta * Latency
```

This baseline uses a stable Ridge Regression pipeline:

```text
StandardScaler → optional PCA → Ridge Regression
```

### Neural Utility Router

`src/llm_router/neural_utility.py` implements a nonlinear neural utility predictor. It solves the same task as Ridge Regression:

```text
input  = 902D real-time context + SLA weights alpha/beta
output = [Utility(local), Utility(cloud)]
route  = argmax(predicted utilities), with a dev-tuned cloud margin
```

The neural router is used for the newer report-facing comparison because it can learn nonlinear interactions between Qwen prompt activations, system load, budget, latency, and SLA preference.

### Streamlit Dashboard

`app/streamlit_app.py` provides a real-time debug and demo dashboard. It shows the selected route, predicted utilities, estimated cost, estimated latency, feature source, live system metrics, and routing explanation.

### FastAPI Gateway

`api_server.py` exposes a minimal production-style routing gateway with:

- `GET /health`
- `POST /route`
- `POST /chat`

The gateway can return route metadata only, or optionally call a backend.

### Optional Backends

`src/llm_router/backends.py` supports:

- Local generation through Ollama, default model `qwen2.5:0.5b`.
- Cloud generation through OpenAI only if `OPENAI_API_KEY` is present.

Cloud generation is optional and disabled safely if no API key is found.

## Model Comparison

The final offline experiment compares:

- Always Local
- Always Cloud
- MLP Classifier
- Utility Prediction Router
- Neural Utility Router
- Oracle / Ground-Truth Best Route

The oracle is computed offline as the route with the higher true utility for each validation example. It is an upper bound, not a deployable policy.

## Evaluation Metrics

Traditional ML metrics:

- Accuracy
- Precision
- Recall
- F1
- Confusion matrix
- Precision-recall curve
- ROC-AUC curve
- Learning curve

System-level routing metrics:

- Average Utility
- Average Quality
- Average Cost
- Average Latency
- Cloud Route Rate
- Regret
- Oracle Utility Ratio
- Margin-aware routing accuracy

The traditional metrics show whether a model matches labels. The system-level metrics show whether the routing gateway makes useful deployment decisions.

## Key Results

In the current course-scale evaluation, the Neural Utility Router outperforms the Ridge Utility Router across all configured SLA modes using the same utility-prediction task and the same validation split. It achieves higher average utility, lower regression error, lower mean regret, and higher oracle utility ratio.

The earlier Ridge Utility Router also achieves higher average utility than the MLP classifier baseline, which motivates the project shift from failure classification to utility prediction.

The Qwen activation features also provide useful semantic signal for local-failure prediction. A cleaned classifier evaluation reaches ROC-AUC around `0.70` on the small validation split, but the dataset is imbalanced and contains few positive local-failure examples.

Core result files:

- `results/neural_utility_comparison_metrics.csv`
- `results/mlp_classifier_metrics.csv`
- `results/dataset_raw_routellm_gpt4_sample.csv`
- `results/dataset_augmented_router_training_sample.csv`

The checked-in `results/plots/` folder keeps only the final report-facing figures,
not every intermediate diagnostic plot.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For macOS Apple Silicon, PyTorch will use MPS when available. CPU mode also works for the small offline experiments and API smoke tests.

## Run the Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

Demo notes:

- Cached fallback prompt features are used by default for speed.
- Select "Use real Qwen feature extraction" to encode the exact typed prompt.
- Cached fallback is fast but approximate.
- Optional local generation requires Ollama.
- Optional cloud generation requires `OPENAI_API_KEY` and explicit user opt-in.

## Run the FastAPI Gateway

```bash
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Route-only request:

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain overfitting in machine learning.",
    "sla_mode": "balanced",
    "session_budget": 2.0,
    "spent_so_far": 0.0,
    "estimated_cloud_cost": 0.05
  }'
```

Chat request:

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

If the selected backend is unavailable, the gateway still returns route metadata and a clear fallback message.

## Optional Backend Setup

### Ollama

Install Ollama, then pull the local model:

```bash
ollama pull qwen2.5:0.5b
```

The gateway expects Ollama at:

```text
http://localhost:11434/api/generate
```

### OpenAI

Set an API key only if you want optional cloud generation:

```bash
export OPENAI_API_KEY="your_key_here"
```

The project does not require OpenAI for offline evaluation, smoke tests, or route-only API calls.

## Reproduce the Offline Experiment

Main neural utility routing experiment:

```bash
python scripts/evaluate_neural_utility_router.py
```

Classifier baseline and final comparison figures:

```bash
python scripts/evaluate_mlp_classifier.py
python scripts/plot_classifier_roc_curves.py
python scripts/plot_neural_vs_ridge.py
```

Dataset preparation and feature caching scripts are included, but the repository only keeps a small final 512/128 processed sample for reproducibility:

```bash
python scripts/prepare_dataset.py
python scripts/cache_activation_features.py
```

## Repository Structure

```text
.
├── api_server.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/                  # ignored; download/regenerate locally
│   └── processed/            # small 512/128 final sample artifacts
├── docs/
│   ├── deployment_gateway.md
│   └── final_project_report.txt
├── models/
│   └── neural_utility_router.pkl
├── results/
│   ├── neural_utility_comparison_metrics.csv
│   ├── mlp_classifier_metrics.csv
│   └── plots/
├── scripts/
│   ├── prepare_dataset.py
│   ├── cache_activation_features.py
│   ├── evaluate_mlp_classifier.py
│   ├── evaluate_neural_utility_router.py
│   ├── plot_classifier_roc_curves.py
│   └── plot_neural_vs_ridge.py
├── src/llm_router/
│   ├── feature_extractor.py
│   ├── mlp_router.py
│   ├── realtime.py
│   ├── utility_prediction.py
│   ├── gateway.py
│   └── backends.py
└── tests/
```

## Limitations

- This is a course-scale prototype, not a production deployment.
- The offline split is small and imbalanced.
- Serving-state features are synthetically augmented because RouteLLM-style datasets do not include live CPU, memory, battery, budget, or latency logs.
- Cloud quality is fixed to `1.0` in the sandbox table.
- The Ridge Utility Router is linear; the newer Neural Utility Router reduces this limitation but is still a course-scale prototype.
- Real production use would need live traffic logs, real latency/cost measurements, and more robust model monitoring.

## Future Work

- Improve the Neural Utility Router with more real traffic logs, stronger nonlinear models, or model ensembles.
- Collect real local/cloud latency and cost logs.
- Add a multi-backend model registry with more than two route choices.
- Learn online from user feedback and observed outcomes.
- Add privacy and safety constraints so sensitive prompts can be forced local.
- Evaluate exact deployed local models rather than using proxy quality labels.
