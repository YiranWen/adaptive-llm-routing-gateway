"""Utility-regression-specific evaluation for the Utility Prediction Router."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_router.offline_data import load_offline_artifacts
from llm_router.realtime import LARGE, SMALL, augment_offline_split, realtime_utility_matrix
from llm_router.utilities import SLA_MODES


MARGIN_THRESHOLDS = [0.01, 0.02, 0.05]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios-per-prompt", type=int, default=5)
    parser.add_argument("--remote-cost-per-request", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=184)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "utility_prediction_realtime.pkl")
    parser.add_argument(
        "--train-features",
        type=Path,
        default=ROOT / "data" / "processed" / "activation_features_train_512.npz",
    )
    parser.add_argument(
        "--validation-features",
        type=Path,
        default=ROOT / "data" / "processed" / "activation_features_validation_128.npz",
    )
    parser.add_argument(
        "--train-table",
        type=Path,
        default=ROOT / "data" / "processed" / "routing_table_train_512.csv",
    )
    parser.add_argument(
        "--validation-table",
        type=Path,
        default=ROOT / "data" / "processed" / "routing_table_validation_128.csv",
    )
    return parser.parse_args()


def percent_less_than(values: np.ndarray, threshold: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float((values < threshold).mean())


def route_accuracy(actions: np.ndarray, oracle: np.ndarray) -> float:
    if len(actions) == 0:
        return float("nan")
    return float((actions == oracle).mean())


def save_regret_distribution(regret_by_mode: dict[str, np.ndarray], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bins = np.linspace(0.0, max(0.08, max(float(r.max()) for r in regret_by_mode.values())), 35)
    for mode_name, regrets in regret_by_mode.items():
        ax.hist(regrets, bins=bins, alpha=0.42, density=True, label=mode_name)
    ax.set_title("Utility Router Regret Distribution by SLA")
    ax.set_xlabel("Oracle utility - selected policy utility")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_oracle_ratio_plot(regret_metrics: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.bar(regret_metrics["sla_mode"], regret_metrics["oracle_utility_ratio"], color="#4477AA")
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="Oracle")
    ax.set_title("Utility Router Oracle Utility Ratio by SLA")
    ax.set_xlabel("SLA mode")
    ax.set_ylabel("Average policy utility / average oracle utility")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_margin_accuracy_plot(margin_accuracy: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    labels = ["all", ">0.01", ">0.02", ">0.05"]
    x = np.arange(len(labels))
    for mode_name in SLA_MODES:
        subset = margin_accuracy[margin_accuracy["sla_mode"] == mode_name]
        values = [
            float(subset[subset["margin_bucket"] == label]["routing_accuracy_vs_oracle"].iloc[0])
            for label in labels
        ]
        ax.plot(x, values, marker="o", linewidth=2, label=mode_name)
    ax.set_title("Margin-Aware Routing Accuracy")
    ax.set_xlabel("True utility margin bucket")
    ax.set_ylabel("Routing accuracy vs oracle")
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_predicted_vs_true_plot(records: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, record in zip(axes, records):
        true_utilities = record["true_utilities"]
        predicted_utilities = record["predicted_utilities"]
        ax.scatter(
            true_utilities[:, SMALL],
            predicted_utilities[:, SMALL],
            s=12,
            alpha=0.35,
            label="local",
        )
        ax.scatter(
            true_utilities[:, LARGE],
            predicted_utilities[:, LARGE],
            s=12,
            alpha=0.35,
            label="cloud",
        )
        lower = float(min(true_utilities.min(), predicted_utilities.min()))
        upper = float(max(true_utilities.max(), predicted_utilities.max()))
        ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
        ax.set_title(record["sla_mode"])
        ax.set_xlabel("True utility")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Predicted utility")
    axes[0].legend()
    fig.suptitle("Utility Router Predicted vs True Utility")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_offline_artifacts(
        train_features=args.train_features,
        validation_features=args.validation_features,
        train_table=args.train_table,
        validation_table=args.validation_table,
    )
    validation = augment_offline_split(
        artifacts.validation,
        scenarios_per_prompt=args.scenarios_per_prompt,
        seed=args.seed + 1,
        remote_cost_per_request=args.remote_cost_per_request,
    )

    with args.model_path.open("rb") as file:
        bundle = pickle.load(file)

    regret_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    regret_by_mode: dict[str, np.ndarray] = {}
    predicted_true_records: list[dict[str, Any]] = []

    for mode_name, sla in SLA_MODES.items():
        model = bundle["models"][mode_name]
        true_utilities = realtime_utility_matrix(validation, sla)
        predicted_utilities = model.predict(validation.contexts).astype(np.float32)
        actions = np.argmax(predicted_utilities, axis=1).astype(np.int64)
        oracle_actions = np.argmax(true_utilities, axis=1).astype(np.int64)

        selected_policy_utility = true_utilities[np.arange(len(actions)), actions]
        oracle_utility = true_utilities[np.arange(len(actions)), oracle_actions]
        regret = np.maximum(oracle_utility - selected_policy_utility, 0.0)
        regret_by_mode[mode_name] = regret
        predicted_true_records.append(
            {
                "sla_mode": mode_name,
                "true_utilities": true_utilities,
                "predicted_utilities": predicted_utilities,
            }
        )

        errors = actions != oracle_actions
        error_regret = regret[errors]
        avg_policy_utility = float(selected_policy_utility.mean())
        avg_oracle_utility = float(oracle_utility.mean())
        regret_rows.append(
            {
                "sla_mode": mode_name,
                "num_examples": int(len(actions)),
                "avg_policy_utility": avg_policy_utility,
                "avg_oracle_utility": avg_oracle_utility,
                "oracle_utility_ratio": avg_policy_utility / avg_oracle_utility,
                "mean_regret": float(regret.mean()),
                "median_regret": float(np.median(regret)),
                "max_regret": float(regret.max()),
                "percent_regret_lt_0.01": percent_less_than(regret, 0.01),
                "percent_regret_lt_0.02": percent_less_than(regret, 0.02),
                "percent_regret_lt_0.05": percent_less_than(regret, 0.05),
                "num_route_errors": int(errors.sum()),
                "mean_error_regret": float(error_regret.mean()) if len(error_regret) else float("nan"),
                "median_error_regret": float(np.median(error_regret)) if len(error_regret) else float("nan"),
                "percent_error_regret_lt_0.01": percent_less_than(error_regret, 0.01),
                "percent_error_regret_lt_0.05": percent_less_than(error_regret, 0.05),
            }
        )

        regression_rows.append(
            {
                "sla_mode": mode_name,
                "num_examples": int(len(actions)),
                "MSE_local": float(mean_squared_error(true_utilities[:, SMALL], predicted_utilities[:, SMALL])),
                "MSE_cloud": float(mean_squared_error(true_utilities[:, LARGE], predicted_utilities[:, LARGE])),
                "MAE_local": float(mean_absolute_error(true_utilities[:, SMALL], predicted_utilities[:, SMALL])),
                "MAE_cloud": float(mean_absolute_error(true_utilities[:, LARGE], predicted_utilities[:, LARGE])),
                "combined_MSE": float(mean_squared_error(true_utilities, predicted_utilities)),
                "combined_MAE": float(mean_absolute_error(true_utilities, predicted_utilities)),
            }
        )

        true_margin = np.abs(true_utilities[:, LARGE] - true_utilities[:, SMALL])
        margin_rows.append(
            {
                "sla_mode": mode_name,
                "margin_bucket": "all",
                "margin_threshold": "all",
                "num_examples": int(len(actions)),
                "routing_accuracy_vs_oracle": route_accuracy(actions, oracle_actions),
            }
        )
        for threshold in MARGIN_THRESHOLDS:
            mask = true_margin > threshold
            margin_rows.append(
                {
                    "sla_mode": mode_name,
                    "margin_bucket": f">{threshold:.2f}",
                    "margin_threshold": threshold,
                    "num_examples": int(mask.sum()),
                    "routing_accuracy_vs_oracle": route_accuracy(actions[mask], oracle_actions[mask]),
                }
            )

    regret_metrics = pd.DataFrame(regret_rows)
    regression_metrics = pd.DataFrame(regression_rows)
    margin_accuracy = pd.DataFrame(margin_rows)

    regret_metrics.to_csv(args.results_dir / "utility_router_regret_metrics.csv", index=False)
    regression_metrics.to_csv(
        args.results_dir / "utility_router_regression_error_metrics.csv",
        index=False,
    )
    margin_accuracy.to_csv(args.results_dir / "utility_router_margin_accuracy.csv", index=False)

    save_regret_distribution(
        regret_by_mode,
        plots_dir / "utility_router_regret_distribution_by_sla.png",
    )
    save_oracle_ratio_plot(
        regret_metrics,
        plots_dir / "utility_router_oracle_ratio_by_sla.png",
    )
    save_margin_accuracy_plot(
        margin_accuracy,
        plots_dir / "utility_router_margin_accuracy.png",
    )
    save_predicted_vs_true_plot(
        predicted_true_records,
        plots_dir / "utility_router_predicted_vs_true_utility.png",
    )

    generated_files = [
        args.results_dir / "utility_router_regret_metrics.csv",
        args.results_dir / "utility_router_regression_error_metrics.csv",
        args.results_dir / "utility_router_margin_accuracy.csv",
        plots_dir / "utility_router_regret_distribution_by_sla.png",
        plots_dir / "utility_router_oracle_ratio_by_sla.png",
        plots_dir / "utility_router_margin_accuracy.png",
        plots_dir / "utility_router_predicted_vs_true_utility.png",
    ]
    print("Utility-regression-specific evaluation complete.")
    print("\nRegret metrics:")
    print(regret_metrics.to_string(index=False))
    print("\nRegression error metrics:")
    print(regression_metrics.to_string(index=False))
    print("\nMargin-aware routing accuracy:")
    print(margin_accuracy.to_string(index=False))
    print("\nGenerated files checklist:")
    for path in generated_files:
        status = "OK" if path.exists() else "MISSING"
        print(f"[{status}] {path}")


if __name__ == "__main__":
    main()
