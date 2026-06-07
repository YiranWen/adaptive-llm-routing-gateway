"""Compare Ridge and Neural Utility Routers on the same prediction task.

Both models solve:

    input  = 902D real-time context + SLA weights alpha/beta
    output = [utility_local, utility_cloud]
    route  = argmax(predicted utilities)

Model selection uses an internal development split from the training prompts.
The official validation split is used only once for final reporting.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_router.neural_utility import (  # noqa: E402
    NeuralUtilityConfig,
    predict_neural_utilities,
    train_neural_utility_router,
)
from llm_router.offline_data import load_offline_artifacts  # noqa: E402
from llm_router.realtime import (  # noqa: E402
    LARGE,
    SMALL,
    RealtimeAugmentedSplit,
    augment_offline_split,
    realtime_route_metrics,
    realtime_utility_matrix,
)
from llm_router.utilities import SLA_MODES  # noqa: E402


RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
NEURAL_CONFIGS = [
    NeuralUtilityConfig(hidden_dims=(128, 64), dropout=0.05, learning_rate=1e-3, weight_decay=1e-4, ranking_weight=0.0),
    NeuralUtilityConfig(hidden_dims=(128, 64), dropout=0.10, learning_rate=1e-3, weight_decay=1e-4, ranking_weight=0.15),
    NeuralUtilityConfig(hidden_dims=(256, 128), dropout=0.10, learning_rate=1e-3, weight_decay=1e-4, ranking_weight=0.15),
    NeuralUtilityConfig(hidden_dims=(256, 128), dropout=0.15, learning_rate=7e-4, weight_decay=1e-4, ranking_weight=0.25),
    NeuralUtilityConfig(hidden_dims=(256, 128, 64), dropout=0.10, learning_rate=7e-4, weight_decay=5e-4, ranking_weight=0.15),
    NeuralUtilityConfig(hidden_dims=(384, 192), dropout=0.15, learning_rate=5e-4, weight_decay=5e-4, ranking_weight=0.20),
]
MARGIN_THRESHOLDS = [0.01, 0.02, 0.05]
CLOUD_MARGIN_GRID = np.round(np.arange(-0.10, 0.205, 0.005), 3)
DEV_UTILITY_TIE_TOLERANCE = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios-per-prompt", type=int, default=10)
    parser.add_argument("--remote-cost-per-request", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=184)
    parser.add_argument("--dev-size", type=float, default=0.25)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
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


def sla_feature_matrix(split: RealtimeAugmentedSplit, mode_name: str) -> np.ndarray:
    sla = SLA_MODES[mode_name]
    sla_features = np.column_stack(
        [
            np.full(len(split.contexts), sla.alpha_cost, dtype=np.float32),
            np.full(len(split.contexts), sla.beta_latency, dtype=np.float32),
        ]
    )
    return np.concatenate([split.contexts, sla_features], axis=1).astype(np.float32)


def expand_split_by_sla(split: RealtimeAugmentedSplit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    utilities: list[np.ndarray] = []
    modes: list[np.ndarray] = []
    for mode_name, sla in SLA_MODES.items():
        features.append(sla_feature_matrix(split, mode_name))
        utilities.append(realtime_utility_matrix(split, sla))
        modes.append(np.full(len(split.contexts), mode_name, dtype=object))
    return (
        np.concatenate(features, axis=0).astype(np.float32),
        np.concatenate(utilities, axis=0).astype(np.float32),
        np.concatenate(modes, axis=0),
    )


def split_train_dev_by_prompt(
    train: RealtimeAugmentedSplit,
    *,
    dev_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split prompt ids, not augmented rows, to avoid prompt leakage."""

    prompt_ids = np.unique(train.prompt_index)
    prompt_failure = np.array(
        [
            int(train.small_failure[train.prompt_index == prompt_id][0])
            for prompt_id in prompt_ids
        ],
        dtype=np.int64,
    )
    fit_ids, dev_ids = train_test_split(
        prompt_ids,
        test_size=dev_size,
        random_state=seed,
        stratify=prompt_failure,
    )
    fit_mask = np.isin(train.prompt_index, fit_ids)
    dev_mask = np.isin(train.prompt_index, dev_ids)
    return fit_mask, dev_mask


def select_rows_by_mask(
    expanded_x: np.ndarray,
    expanded_y: np.ndarray,
    base_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a base augmented-row mask to all SLA-expanded blocks."""

    repeated_mask = np.tile(base_mask, len(SLA_MODES))
    return expanded_x[repeated_mask], expanded_y[repeated_mask]


def evaluate_predictions(
    *,
    model_name: str,
    split_name: str,
    split: RealtimeAugmentedSplit,
    predictions_by_mode: dict[str, np.ndarray],
    cloud_margins: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cloud_margins = cloud_margins or {mode_name: 0.0 for mode_name in SLA_MODES}
    for mode_name, sla in SLA_MODES.items():
        true_utilities = realtime_utility_matrix(split, sla)
        predicted = predictions_by_mode[mode_name]
        cloud_margin = float(cloud_margins.get(mode_name, 0.0))
        actions = actions_from_predictions(predicted, cloud_margin=cloud_margin)
        oracle_actions = np.argmax(true_utilities, axis=1).astype(np.int64)
        selected_true_utility = true_utilities[np.arange(len(actions)), actions]
        oracle_utility = true_utilities[np.arange(len(actions)), oracle_actions]
        regret = oracle_utility - selected_true_utility

        route_metrics = realtime_route_metrics(
            mode=mode_name,
            policy=model_name,
            split=split,
            actions=actions,
            sla=sla,
        )
        combined_mse = mean_squared_error(true_utilities, predicted)
        combined_mae = mean_absolute_error(true_utilities, predicted)
        rmse = float(np.sqrt(combined_mse))
        row: dict[str, Any] = {
            "model": model_name,
            "split": split_name,
            "cloud_margin": cloud_margin,
            **route_metrics,
            "mse_local": float(mean_squared_error(true_utilities[:, 0], predicted[:, 0])),
            "mse_cloud": float(mean_squared_error(true_utilities[:, 1], predicted[:, 1])),
            "mae_local": float(mean_absolute_error(true_utilities[:, 0], predicted[:, 0])),
            "mae_cloud": float(mean_absolute_error(true_utilities[:, 1], predicted[:, 1])),
            "combined_mse": float(combined_mse),
            "combined_mae": float(combined_mae),
            "combined_rmse": rmse,
            "r2_local": float(r2_score(true_utilities[:, 0], predicted[:, 0])),
            "r2_cloud": float(r2_score(true_utilities[:, 1], predicted[:, 1])),
            "mean_regret": float(regret.mean()),
            "median_regret": float(np.median(regret)),
            "max_regret": float(regret.max()),
            "oracle_utility_ratio": float(selected_true_utility.mean() / oracle_utility.mean()),
            "percent_regret_lt_0.01": float((regret < 0.01).mean()),
            "percent_regret_lt_0.02": float((regret < 0.02).mean()),
            "percent_regret_lt_0.05": float((regret < 0.05).mean()),
        }
        true_margin = np.abs(true_utilities[:, 1] - true_utilities[:, 0])
        for threshold in MARGIN_THRESHOLDS:
            mask = true_margin > threshold
            key = str(threshold).replace(".", "_")
            row[f"routing_acc_margin_gt_{key}"] = (
                float((actions[mask] == oracle_actions[mask]).mean())
                if mask.any()
                else np.nan
            )
            row[f"num_margin_gt_{key}"] = int(mask.sum())
        rows.append(row)
    return rows


def actions_from_predictions(predicted: np.ndarray, *, cloud_margin: float) -> np.ndarray:
    """Route cloud only when predicted utility gain clears the margin."""

    gap = predicted[:, 1] - predicted[:, 0]
    return np.where(gap >= cloud_margin, LARGE, SMALL).astype(np.int64)


def tune_cloud_margins(
    *,
    split: RealtimeAugmentedSplit,
    predictions_by_mode: dict[str, np.ndarray],
) -> dict[str, float]:
    """Tune one cloud escalation margin per SLA mode on internal dev only."""

    margins: dict[str, float] = {}
    for mode_name, sla in SLA_MODES.items():
        true_utilities = realtime_utility_matrix(split, sla)
        best_margin = 0.0
        best_utility = -np.inf
        for margin in CLOUD_MARGIN_GRID:
            actions = actions_from_predictions(
                predictions_by_mode[mode_name],
                cloud_margin=float(margin),
            )
            selected = true_utilities[np.arange(len(actions)), actions]
            avg_utility = float(selected.mean())
            if avg_utility > best_utility:
                best_utility = avg_utility
                best_margin = float(margin)
        margins[mode_name] = best_margin
    return margins


def predictions_by_mode_from_unified_model(model, split: RealtimeAugmentedSplit) -> dict[str, np.ndarray]:
    return {
        mode_name: model.predict(sla_feature_matrix(split, mode_name)).astype(np.float32)
        for mode_name in SLA_MODES
    }


def neural_predictions_by_mode(result, split: RealtimeAugmentedSplit) -> dict[str, np.ndarray]:
    return {
        mode_name: predict_neural_utilities(result, sla_feature_matrix(split, mode_name))
        for mode_name in SLA_MODES
    }


def dev_objective(rows: list[dict[str, Any]]) -> float:
    """Primary model-selection objective: mean dev avg_utility across SLAs."""

    return float(np.mean([row["avg_utility"] for row in rows]))


def train_best_ridge(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    dev_split: RealtimeAugmentedSplit,
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    ablation_rows: list[dict[str, Any]] = []
    best_model = None
    best_score = -np.inf
    best_meta: dict[str, Any] = {}
    for alpha in RIDGE_ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(fit_x, fit_y)
        predictions_by_mode = predictions_by_mode_from_unified_model(model, dev_split)
        margins = tune_cloud_margins(
            split=dev_split,
            predictions_by_mode=predictions_by_mode,
        )
        rows = evaluate_predictions(
            model_name="Ridge Utility Router",
            split_name="internal_dev",
            split=dev_split,
            predictions_by_mode=predictions_by_mode,
            cloud_margins=margins,
        )
        score = dev_objective(rows)
        for row in rows:
            ablation_rows.append({"ridge_alpha": alpha, **row})
        if score > best_score:
            best_score = score
            best_model = model
            best_meta = {
                "ridge_alpha": alpha,
                "dev_avg_utility_mean": score,
                "cloud_margins": margins,
            }

    assert best_model is not None
    return best_model, best_meta, pd.DataFrame(ablation_rows)


def train_best_neural(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    dev_split: RealtimeAugmentedSplit,
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    ablation_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for index, base_config in enumerate(NEURAL_CONFIGS):
        config = NeuralUtilityConfig(
            **{
                **asdict(base_config),
                "seed": 184 + index,
            }
        )
        result = train_neural_utility_router(
            fit_x,
            fit_y,
            dev_x,
            dev_y,
            config=config,
        )
        predictions_by_mode = neural_predictions_by_mode(result, dev_split)
        margins = tune_cloud_margins(
            split=dev_split,
            predictions_by_mode=predictions_by_mode,
        )
        rows = evaluate_predictions(
            model_name="Neural Utility Router",
            split_name="internal_dev",
            split=dev_split,
            predictions_by_mode=predictions_by_mode,
            cloud_margins=margins,
        )
        score = dev_objective(rows)
        mean_combined_mae = float(np.mean([row["combined_mae"] for row in rows]))
        mean_cloud_rate = float(np.mean([row["cloud_route_rate"] for row in rows]))
        for row in rows:
            ablation_rows.append(
                {
                    "config_index": index,
                    "hidden_dims": str(config.hidden_dims),
                    "dropout": config.dropout,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "ranking_weight": config.ranking_weight,
                    "best_epoch": result.best_epoch,
                    "best_dev_loss": result.best_dev_loss,
                    "cloud_margins": str(margins),
                    **row,
                }
            )
        candidates.append(
            {
                "result": result,
                "score": score,
                "mean_combined_mae": mean_combined_mae,
                "mean_cloud_rate": mean_cloud_rate,
                "meta": {
                    "config_index": index,
                    "hidden_dims": str(config.hidden_dims),
                    "dropout": config.dropout,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "ranking_weight": config.ranking_weight,
                    "best_epoch": result.best_epoch,
                    "best_dev_loss": result.best_dev_loss,
                    "dev_avg_utility_mean": score,
                    "dev_mean_combined_mae": mean_combined_mae,
                    "dev_mean_cloud_rate": mean_cloud_rate,
                    "selection_rule": (
                        "highest internal-dev avg utility; if within "
                        f"{DEV_UTILITY_TIE_TOLERANCE}, choose lower dev combined MAE"
                    ),
                    "cloud_margins": margins,
                },
            }
        )

    best_dev_score = max(float(candidate["score"]) for candidate in candidates)
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate["score"]) >= best_dev_score - DEV_UTILITY_TIE_TOLERANCE
    ]
    best_candidate = min(
        eligible,
        key=lambda candidate: (
            float(candidate["mean_combined_mae"]),
            float(candidate["mean_cloud_rate"]),
        ),
    )
    return best_candidate["result"], best_candidate["meta"], pd.DataFrame(ablation_rows)


def subset_split(split: RealtimeAugmentedSplit, mask: np.ndarray) -> RealtimeAugmentedSplit:
    return RealtimeAugmentedSplit(
        contexts=split.contexts[mask],
        qwen_features=split.qwen_features[mask],
        system_features=split.system_features[mask],
        score_small=split.score_small[mask],
        score_large=split.score_large[mask],
        small_failure=split.small_failure[mask],
        prompt_index=split.prompt_index[mask],
        prompts=split.prompts[mask],
        remote_cost_per_request=split.remote_cost_per_request,
    )


def save_policy_comparison_plot(metrics: pd.DataFrame, output: Path) -> None:
    validation = metrics[metrics["split"] == "validation"].copy()
    modes = list(SLA_MODES)
    models = ["Ridge Utility Router", "Neural Utility Router", "Oracle"]
    x = np.arange(len(modes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for index, model in enumerate(models):
        values = [
            validation[(validation["mode"] == mode) & (validation["model"] == model)][
                "avg_utility"
            ].iloc[0]
            for mode in modes
        ]
        ax.bar(x + (index - 1) * width, values, width=width, label=model)
    ax.set_title("Neural Utility Router vs Ridge Regression")
    ax.set_xlabel("SLA mode")
    ax.set_ylabel("Average selected true utility")
    ax.set_xticks(x, modes)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_regression_error_plot(metrics: pd.DataFrame, output: Path) -> None:
    validation = metrics[
        (metrics["split"] == "validation")
        & (metrics["model"].isin(["Ridge Utility Router", "Neural Utility Router"]))
    ].copy()
    modes = list(SLA_MODES)
    x = np.arange(len(modes))
    width = 0.32
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for index, model in enumerate(["Ridge Utility Router", "Neural Utility Router"]):
        values = [
            validation[(validation["mode"] == mode) & (validation["model"] == model)][
                "combined_mae"
            ].iloc[0]
            for mode in modes
        ]
        ax.bar(x + (index - 0.5) * width, values, width=width, label=model)
    ax.set_title("Utility Prediction Error")
    ax.set_xlabel("SLA mode")
    ax.set_ylabel("Combined MAE")
    ax.set_xticks(x, modes)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_regret_plot(metrics: pd.DataFrame, output: Path) -> None:
    validation = metrics[
        (metrics["split"] == "validation")
        & (metrics["model"].isin(["Ridge Utility Router", "Neural Utility Router"]))
    ].copy()
    modes = list(SLA_MODES)
    x = np.arange(len(modes))
    width = 0.32
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for index, model in enumerate(["Ridge Utility Router", "Neural Utility Router"]):
        values = [
            validation[(validation["mode"] == mode) & (validation["model"] == model)][
                "mean_regret"
            ].iloc[0]
            for mode in modes
        ]
        ax.bar(x + (index - 0.5) * width, values, width=width, label=model)
    ax.set_title("Mean Regret vs Oracle")
    ax.set_xlabel("SLA mode")
    ax.set_ylabel("Mean regret")
    ax.set_xticks(x, modes)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_predicted_vs_true_plot(
    *,
    true_utilities: np.ndarray,
    ridge_predictions: np.ndarray,
    neural_predictions: np.ndarray,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, name, predictions in [
        (axes[0], "Ridge", ridge_predictions),
        (axes[1], "Neural", neural_predictions),
    ]:
        ax.scatter(true_utilities.reshape(-1), predictions.reshape(-1), s=8, alpha=0.35)
        low = min(float(true_utilities.min()), float(predictions.min()))
        high = max(float(true_utilities.max()), float(predictions.max()))
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{name}: Predicted vs True Utility")
        ax.set_xlabel("True utility")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Predicted utility")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def oracle_rows(split_name: str, split: RealtimeAugmentedSplit) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode_name, sla in SLA_MODES.items():
        utilities = realtime_utility_matrix(split, sla)
        actions = np.argmax(utilities, axis=1).astype(np.int64)
        rows.append(
            {
                "model": "Oracle",
                "split": split_name,
                **realtime_route_metrics(
                    mode=mode_name,
                    policy="Oracle",
                    split=split,
                    actions=actions,
                    sla=sla,
                ),
                "mse_local": 0.0,
                "mse_cloud": 0.0,
                "mae_local": 0.0,
                "mae_cloud": 0.0,
                "combined_mse": 0.0,
                "combined_mae": 0.0,
                "combined_rmse": 0.0,
                "r2_local": 1.0,
                "r2_cloud": 1.0,
                "mean_regret": 0.0,
                "median_regret": 0.0,
                "max_regret": 0.0,
                "oracle_utility_ratio": 1.0,
                "percent_regret_lt_0.01": 1.0,
                "percent_regret_lt_0.02": 1.0,
                "percent_regret_lt_0.05": 1.0,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_offline_artifacts(
        train_features=args.train_features,
        validation_features=args.validation_features,
        train_table=args.train_table,
        validation_table=args.validation_table,
    )
    train = augment_offline_split(
        artifacts.train,
        scenarios_per_prompt=args.scenarios_per_prompt,
        seed=args.seed,
        remote_cost_per_request=args.remote_cost_per_request,
    )
    validation = augment_offline_split(
        artifacts.validation,
        scenarios_per_prompt=args.scenarios_per_prompt,
        seed=args.seed + 1,
        remote_cost_per_request=args.remote_cost_per_request,
    )

    fit_mask, dev_mask = split_train_dev_by_prompt(
        train,
        dev_size=args.dev_size,
        seed=args.seed,
    )
    fit_split = subset_split(train, fit_mask)
    dev_split = subset_split(train, dev_mask)

    expanded_train_x, expanded_train_y, _modes = expand_split_by_sla(train)
    fit_x, fit_y = select_rows_by_mask(expanded_train_x, expanded_train_y, fit_mask)
    dev_x, dev_y = select_rows_by_mask(expanded_train_x, expanded_train_y, dev_mask)

    ridge_model, ridge_meta, ridge_ablation = train_best_ridge(fit_x, fit_y, dev_split)
    neural_model, neural_meta, neural_ablation = train_best_neural(
        fit_x,
        fit_y,
        dev_x,
        dev_y,
        dev_split,
    )

    rows: list[dict[str, Any]] = []
    rows.extend(
        evaluate_predictions(
            model_name="Ridge Utility Router",
            split_name="validation",
            split=validation,
            predictions_by_mode=predictions_by_mode_from_unified_model(ridge_model, validation),
            cloud_margins=ridge_meta["cloud_margins"],
        )
    )
    rows.extend(
        evaluate_predictions(
            model_name="Neural Utility Router",
            split_name="validation",
            split=validation,
            predictions_by_mode=neural_predictions_by_mode(neural_model, validation),
            cloud_margins=neural_meta["cloud_margins"],
        )
    )
    rows.extend(oracle_rows("validation", validation))

    metrics = pd.DataFrame(rows)
    metrics["model"] = pd.Categorical(
        metrics["model"],
        categories=["Ridge Utility Router", "Neural Utility Router", "Oracle"],
        ordered=True,
    )
    metrics["mode"] = pd.Categorical(metrics["mode"], categories=list(SLA_MODES), ordered=True)
    metrics = metrics.sort_values(["mode", "model"]).reset_index(drop=True)
    metrics_path = args.results_dir / "neural_utility_comparison_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    ridge_ablation.to_csv(args.results_dir / "ridge_utility_internal_dev_ablation.csv", index=False)
    neural_ablation.to_csv(args.results_dir / "neural_utility_internal_dev_ablation.csv", index=False)

    model_path = args.model_dir / "neural_utility_router.pkl"
    with model_path.open("wb") as file:
        pickle.dump(
            {
                "model_type": "neural_utility_router",
                "scenario": {
                    "context_dim": int(train.contexts.shape[1]),
                    "qwen_dim": int(artifacts.train.features.shape[1]),
                    "system_feature_dim": int(train.system_features.shape[1]),
                    "sla_feature_dim": 2,
                    "model_input_dim": int(fit_x.shape[1]),
                    "remote_cost_per_request": args.remote_cost_per_request,
                    "scenarios_per_prompt": args.scenarios_per_prompt,
                    "selected_by": "internal_dev_mean_avg_utility",
                },
                "ridge_baseline_meta": ridge_meta,
                "neural_meta": neural_meta,
                "neural_model": neural_model,
            },
            file,
        )

    save_policy_comparison_plot(
        metrics,
        plots_dir / "neural_utility_policy_comparison.png",
    )
    save_regression_error_plot(
        metrics,
        plots_dir / "neural_utility_regression_error.png",
    )
    save_regret_plot(
        metrics,
        plots_dir / "neural_utility_regret_comparison.png",
    )
    balanced_true = realtime_utility_matrix(validation, SLA_MODES["balanced"])
    save_predicted_vs_true_plot(
        true_utilities=balanced_true,
        ridge_predictions=predictions_by_mode_from_unified_model(ridge_model, validation)["balanced"],
        neural_predictions=neural_predictions_by_mode(neural_model, validation)["balanced"],
        output=plots_dir / "neural_utility_predicted_vs_true_balanced.png",
    )

    print("Experiment setup:")
    print(f"  fit examples: {fit_x.shape}")
    print(f"  dev examples: {dev_x.shape}")
    print(f"  validation contexts: {validation.contexts.shape}")
    print(f"  model input dim: {fit_x.shape[1]}")
    print("\nSelected configs:")
    print(f"  Ridge: {ridge_meta}")
    print(f"  Neural: {neural_meta}")
    print("\nGenerated:")
    for path in [
        metrics_path,
        args.results_dir / "ridge_utility_internal_dev_ablation.csv",
        args.results_dir / "neural_utility_internal_dev_ablation.csv",
        model_path,
        plots_dir / "neural_utility_policy_comparison.png",
        plots_dir / "neural_utility_regression_error.png",
        plots_dir / "neural_utility_regret_comparison.png",
        plots_dir / "neural_utility_predicted_vs_true_balanced.png",
    ]:
        print(f"  [{'OK' if path.exists() else 'MISSING'}] {path}")
    print("\nValidation metrics:")
    show_cols = [
        "mode",
        "model",
        "avg_utility",
        "combined_mae",
        "combined_rmse",
        "r2_local",
        "r2_cloud",
        "mean_regret",
        "oracle_utility_ratio",
        "routing_accuracy_vs_oracle",
        "cloud_route_rate",
        "avg_cost",
        "avg_latency",
    ]
    print(metrics[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
