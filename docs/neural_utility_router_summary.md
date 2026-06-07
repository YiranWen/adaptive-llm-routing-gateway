# Neural Utility Router Summary

This document summarizes the new final model comparison:

```text
Ridge Utility Router vs Neural Utility Router
```

Both models solve the same supervised utility prediction task.

## Task Formulation

Input:

```text
904D input
= 896D Qwen activation feature
+ 6D serving/system feature vector
+ 2D SLA feature vector [alpha_cost, beta_latency]
```

Output:

```text
[predicted_utility_local, predicted_utility_cloud]
```

Decision:

```text
route cloud if predicted_utility_cloud - predicted_utility_local >= cloud_margin
otherwise route local
```

The cloud margin is selected on an internal development split, not on the final validation split.

## Why This Comparison Is Fair

The Ridge Utility Router and Neural Utility Router use:

- the same input features
- the same ground-truth utility labels
- the same route decision rule
- the same internal development split for model selection
- the same final validation split
- the same evaluation metrics

Ridge Regression is a regularized linear model. It assumes a mostly linear relationship between context features and utility.

The Neural Utility Router is a multi-layer neural network. It can learn nonlinear interactions between:

- prompt semantics
- CPU load
- memory load
- battery level
- budget remaining
- estimated local/cloud latency
- SLA cost and latency weights

This is the main reason the Neural Utility Router is a stronger final model.

## Training Setup

The experiment uses the existing 512 train / 128 validation prompt split.

For serving-state augmentation:

```text
scenarios_per_prompt = 10
```

Therefore:

```text
train contexts = 512 prompts * 10 scenarios = 5120 contexts
validation contexts = 128 prompts * 10 scenarios = 1280 contexts
```

Because each context is evaluated under three SLA modes, the utility-prediction training matrix contains:

```text
5120 contexts * 3 SLA modes = 15360 utility training rows
```

The script is:

```bash
python scripts/evaluate_neural_utility_router.py
```

## Selected Neural Model

The selected Neural Utility Router configuration is:

```text
hidden_dims = (256, 128, 64)
dropout = 0.10
learning_rate = 0.0007
weight_decay = 0.0005
ranking_weight = 0.15
best_epoch = 8
```

Selection rule:

```text
Choose the highest internal-dev average utility.
If models are within 0.001 average utility, choose the lower dev combined MAE.
```

This avoids choosing a marginally higher-utility but less stable model.

## Final Validation Results

| SLA Mode | Model | Avg Utility ↑ | Combined MAE ↓ | Combined RMSE ↓ | Mean Regret ↓ | Oracle Utility Ratio ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| quality_first | Ridge Utility Router | 0.9519 | 0.1353 | 0.2460 | 0.0276 | 0.9718 |
| quality_first | Neural Utility Router | 0.9645 | 0.0893 | 0.1577 | 0.0150 | 0.9847 |
| balanced | Ridge Utility Router | 0.8924 | 0.1335 | 0.2459 | 0.0533 | 0.9437 |
| balanced | Neural Utility Router | 0.9032 | 0.0890 | 0.1578 | 0.0425 | 0.9550 |
| cost_sensitive | Ridge Utility Router | 0.8399 | 0.1385 | 0.2479 | 0.0689 | 0.9242 |
| cost_sensitive | Neural Utility Router | 0.8507 | 0.0977 | 0.1609 | 0.0580 | 0.9361 |

## Interpretation

The Neural Utility Router outperforms Ridge Regression across all three SLA modes.

The strongest improvements are in utility prediction quality:

- much lower Combined MAE
- much lower Combined RMSE
- lower mean regret
- higher oracle utility ratio

This supports the final project claim:

```text
LLM routing is better formulated as utility prediction than simple failure classification,
and nonlinear neural utility prediction outperforms a regularized linear regression router.
```

## Generated Files

Metrics:

```text
results/neural_utility_comparison_metrics.csv
results/ridge_utility_internal_dev_ablation.csv
results/neural_utility_internal_dev_ablation.csv
```

Model:

```text
models/neural_utility_router.pkl
```

Plots:

```text
results/plots/neural_utility_policy_comparison.png
results/plots/neural_utility_regression_error.png
results/plots/neural_utility_regret_comparison.png
results/plots/neural_utility_predicted_vs_true_balanced.png
```
