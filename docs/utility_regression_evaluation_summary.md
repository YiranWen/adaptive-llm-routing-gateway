# Utility Regression Evaluation Summary

This document explains the additional evaluation metrics for the Utility Prediction Router.

The Utility Prediction Router is a regression-based router. It predicts:

```text
Utility(local)
Utility(cloud)
```

Then it chooses the route with the higher predicted utility.

## 1. Oracle Route

The oracle route is the ground-truth best route computed from true local/cloud utility:

```text
oracle = argmax(true_utility_local, true_utility_cloud)
```

The oracle is not deployable because it uses the true utility values from the offline validation table. It is used as an evaluation reference.

## 2. Why Confusion Matrix Metrics Can Look Low

Confusion matrix, routing accuracy, precision, recall, and F1 treat route selection as a strict binary classification problem:

```text
local = 0
cloud = 1
```

These metrics count every disagreement with the oracle as equally wrong.

However, for a utility regression router, this can be misleading. Some route disagreements have very small utility gaps. For example:

```text
true Utility(local) = 0.850
true Utility(cloud) = 0.852
```

The oracle chooses cloud, but if the router chooses local, the utility loss is only:

```text
0.852 - 0.850 = 0.002
```

Classification accuracy counts this as a full mistake. Regret counts it as a tiny loss.

## 3. Regret Metrics

For each validation example:

```text
regret = oracle_utility - selected_policy_utility
```

Regret measures how much utility the router lost by not selecting the oracle route.

Reported metrics:

- mean_regret
- median_regret
- max_regret
- percent_regret_lt_0.01
- percent_regret_lt_0.02
- percent_regret_lt_0.05

Interpretation:

- Low mean regret means that the router's mistakes are usually low-cost.
- A high percent_regret_lt_0.01 means most decisions are within 0.01 utility of oracle.
- Max regret shows the worst-case utility loss.

## 4. Oracle Utility Ratio

The oracle utility ratio is:

```text
oracle_utility_ratio = avg_policy_utility / avg_oracle_utility
```

This measures how close the router's achieved utility is to the oracle upper bound.

For example:

```text
oracle_utility_ratio = 0.95
```

means the router achieves 95% of the oracle's average utility.

This is often more meaningful than exact route accuracy because it evaluates utility achieved, not only label matching.

## 5. Regression Prediction Error

Because the Utility Router predicts continuous utility scores, it should also be evaluated as a regression model.

Reported metrics:

- MSE_local
- MSE_cloud
- MAE_local
- MAE_cloud
- combined_MSE across both output columns
- combined_MAE across both output columns

MSE stands for mean squared error. It penalizes large prediction errors more heavily.

MAE stands for mean absolute error. It measures average absolute prediction error in utility units.

These are computed using:

```text
sklearn.metrics.mean_squared_error
sklearn.metrics.mean_absolute_error
```

## 6. Margin-Aware Routing Accuracy

The true utility margin is:

```text
true_margin = abs(true_utility_cloud - true_utility_local)
```

If the margin is very small, the local and cloud routes are nearly tied. In those cases, choosing the opposite route from oracle may not matter much.

Therefore, routing accuracy is reported on:

- all examples
- examples with true_margin > 0.01
- examples with true_margin > 0.02
- examples with true_margin > 0.05

This shows whether the router performs better when the oracle route has a clearer utility advantage.

## 7. Utility Gap Analysis for Errors

For examples where:

```text
router action != oracle action
```

the evaluation computes:

- mean_error_regret
- median_error_regret
- percent_error_regret_lt_0.01
- percent_error_regret_lt_0.05

These metrics answer:

```text
When the router is wrong, how costly are its mistakes?
```

This is especially important for routing systems because not all route disagreements are equally harmful.

## 8. Generated Outputs

CSV files:

- `results/utility_router_regret_metrics.csv`
- `results/utility_router_regression_error_metrics.csv`
- `results/utility_router_margin_accuracy.csv`

Plots:

- `results/plots/utility_router_regret_distribution_by_sla.png`
- `results/plots/utility_router_oracle_ratio_by_sla.png`
- `results/plots/utility_router_margin_accuracy.png`
- `results/plots/utility_router_predicted_vs_true_utility.png`

## 9. Relationship to Traditional Metrics

Traditional classification metrics are still useful:

- confusion matrix
- routing accuracy
- precision
- recall
- F1

They answer:

```text
How often does the router exactly match the oracle route?
```

Utility-regression metrics answer:

```text
How much utility does the router actually lose compared with oracle?
```

Both views should be included in the final report.

The key interpretation is:

```text
Low routing accuracy can still correspond to high average utility if many disagreements have small utility gaps.
```

