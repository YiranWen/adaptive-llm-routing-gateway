"""Generate traditional ML evaluation artifacts for the final project."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_router.mlp_router import MLPTrainingResult, train_mlp_router
from llm_router.offline_data import load_offline_artifacts
from llm_router.realtime import (
    LARGE,
    SMALL,
    augment_offline_split,
    realtime_oracle_actions,
    realtime_utility_matrix,
)
from llm_router.utilities import SLA_MODES


FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios-per-prompt", type=int, default=5)
    parser.add_argument("--remote-cost-per-request", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=184)
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


def pca_dim_from_value(value: Any) -> int | None:
    if value in (None, "", "no_pca"):
        return None
    return int(value)


def stratified_fraction_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in np.unique(labels):
        group = np.flatnonzero(labels == label)
        rng.shuffle(group)
        count = max(1, int(round(len(group) * fraction)))
        selected.append(group[:count])
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return indices.astype(np.int64)


def route_fraction_indices(
    train_oracle_by_mode: dict[str, np.ndarray],
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Sample a shared training subset while preserving route labels when possible."""

    labels = np.logical_or.reduce(
        [actions == LARGE for actions in train_oracle_by_mode.values()]
    ).astype(np.int64)
    return stratified_fraction_indices(labels, fraction, seed)


def predict_mlp_probabilities(result: MLPTrainingResult, features: np.ndarray) -> np.ndarray:
    import torch

    model = result.model
    model.eval()
    x = ((features - result.mean) / result.std).astype(np.float32)
    with torch.inference_mode():
        logits = model(torch.from_numpy(x))
        probabilities = torch.sigmoid(logits).numpy().reshape(-1)
    return probabilities.astype(np.float32)


def binary_metric_row(
    *,
    model: str,
    split: str,
    task: str,
    positive_class: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    notes: str = "",
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    accuracy = float(accuracy_score(y_true, y_pred))
    row: dict[str, Any] = {
        "model": model,
        "split": split,
        "task": task,
        "positive_class": positive_class,
        "accuracy": accuracy,
        "error": 1.0 - accuracy,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "pr_auc": np.nan,
        "roc_auc": np.nan,
        "notes": notes,
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        row["pr_auc"] = float(average_precision_score(y_true, y_score))
        row["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return row


def draw_confusion_matrix(
    cm: np.ndarray,
    *,
    title: str,
    class_names: list[str],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_mlp_precision_recall(y_true: np.ndarray, y_score: np.ndarray, output: Path) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, _thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    label = f"MLP validation PR curve (AP={ap:.3f})" if not np.isnan(ap) else "MLP validation PR curve"
    ax.plot(recall, precision, linewidth=2, label=label)
    ax.set_title("MLP Classifier Precision-Recall Curve")
    ax.set_xlabel("Recall for local_failure")
    ax.set_ylabel("Precision for local_failure")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_mlp_learning_curve(curve: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(curve["training_examples"], curve["train_accuracy"], marker="o", label="Train accuracy")
    ax.plot(
        curve["training_examples"],
        curve["validation_accuracy"],
        marker="o",
        label="Validation accuracy",
    )
    ax.set_title("MLP Classifier Learning Curve")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_mlp_error_curve(curve: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(curve["training_examples"], curve["train_error"], marker="o", label="Train error")
    ax.plot(
        curve["training_examples"],
        curve["validation_error"],
        marker="o",
        label="Validation error",
    )
    ax.set_title("MLP Classifier Train vs Validation Error")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Error")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def train_utility_model(
    train_contexts: np.ndarray,
    utility_matrix: np.ndarray,
    *,
    pca_dim: int | None,
    ridge_alpha: float,
):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps: list[object] = [StandardScaler()]
    if pca_dim is not None:
        steps.append(PCA(n_components=pca_dim, random_state=184))
    steps.append(Ridge(alpha=ridge_alpha))
    model = make_pipeline(*steps)
    model.fit(train_contexts, utility_matrix)
    return model


def utility_predictions_to_actions(model: Any, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted_utilities = model.predict(contexts)
    actions = np.argmax(predicted_utilities, axis=1).astype(np.int64)
    return predicted_utilities.astype(np.float32), actions


def plot_utility_pr_by_sla(rows: list[dict[str, Any]], output: Path) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for row in rows:
        y_true = row["y_true"]
        y_score = row["score"]
        precision, recall, _thresholds = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan
        label = f"{row['mode']} (AP={ap:.3f})" if not np.isnan(ap) else row["mode"]
        ax.plot(recall, precision, linewidth=2, label=label)
    ax.set_title("Utility Router Precision-Recall by SLA")
    ax.set_xlabel("Recall for cloud oracle action")
    ax.set_ylabel("Precision for cloud oracle action")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_utility_accuracy_by_sla(metrics: pd.DataFrame, output: Path) -> None:
    validation = metrics[metrics["split"] == "validation"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.bar(validation["sla_mode"], validation["routing_accuracy_vs_oracle"], color="#4477AA")
    ax.set_title("Utility Router Routing Accuracy by SLA")
    ax.set_xlabel("SLA mode")
    ax.set_ylabel("Routing accuracy vs oracle")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_utility_learning_curve(curve: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for mode in SLA_MODES:
        subset = curve[curve["sla_mode"] == mode]
        ax.plot(
            subset["training_examples"],
            subset["validation_routing_accuracy_vs_oracle"],
            marker="o",
            label=f"{mode} validation",
        )
    ax.set_title("Utility Router Learning Curve")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Validation routing accuracy vs oracle")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_utility_error_curve(curve: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for mode in SLA_MODES:
        subset = curve[curve["sla_mode"] == mode]
        ax.plot(
            subset["training_examples"],
            subset["train_route_error"],
            linestyle="--",
            marker="o",
            label=f"{mode} train",
        )
        ax.plot(
            subset["training_examples"],
            subset["validation_route_error"],
            marker="o",
            label=f"{mode} validation",
        )
    ax.set_title("Utility Router Train vs Validation Route Error")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Route error")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_utility_mse_curve(curve: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for mode in SLA_MODES:
        subset = curve[curve["sla_mode"] == mode]
        ax.plot(
            subset["training_examples"],
            subset["train_mse"],
            linestyle="--",
            marker="o",
            label=f"{mode} train",
        )
        ax.plot(
            subset["training_examples"],
            subset["validation_mse"],
            marker="o",
            label=f"{mode} validation",
        )
    ax.set_title("Utility Router Utility MSE Learning Curve")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Mean squared error")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    from sklearn.metrics import confusion_matrix, mean_squared_error

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

    # A. MLP classifier traditional metrics.
    mlp_result, _mlp_validation_actions = train_mlp_router(
        train.contexts,
        train.small_failure,
        validation.contexts,
        seed=args.seed,
    )
    mlp_train_scores = predict_mlp_probabilities(mlp_result, train.contexts)
    mlp_validation_scores = predict_mlp_probabilities(mlp_result, validation.contexts)
    mlp_train_pred = (mlp_train_scores >= 0.5).astype(np.int64)
    mlp_validation_pred = (mlp_validation_scores >= 0.5).astype(np.int64)

    mlp_metric_rows = [
        binary_metric_row(
            model="MLP Classifier",
            split="train",
            task="local_failure prediction",
            positive_class="local_failure = 1",
            y_true=train.small_failure,
            y_pred=mlp_train_pred,
            y_score=mlp_train_scores,
        ),
        binary_metric_row(
            model="MLP Classifier",
            split="validation",
            task="local_failure prediction",
            positive_class="local_failure = 1",
            y_true=validation.small_failure,
            y_pred=mlp_validation_pred,
            y_score=mlp_validation_scores,
        ),
    ]
    mlp_metrics = pd.DataFrame(mlp_metric_rows)
    mlp_metrics.to_csv(args.results_dir / "mlp_classifier_metrics.csv", index=False)

    draw_confusion_matrix(
        confusion_matrix(validation.small_failure, mlp_validation_pred, labels=[0, 1]),
        title="MLP Classifier Confusion Matrix",
        class_names=["no local failure", "local failure"],
        output=plots_dir / "mlp_classifier_confusion_matrix.png",
    )
    plot_mlp_precision_recall(
        validation.small_failure,
        mlp_validation_scores,
        plots_dir / "mlp_classifier_precision_recall_curve.png",
    )

    mlp_curve_rows: list[dict[str, Any]] = []
    for fraction in FRACTIONS:
        indices = stratified_fraction_indices(train.small_failure, fraction, args.seed)
        result, _actions = train_mlp_router(
            train.contexts[indices],
            train.small_failure[indices],
            validation.contexts,
            seed=args.seed,
        )
        train_scores = predict_mlp_probabilities(result, train.contexts[indices])
        validation_scores = predict_mlp_probabilities(result, validation.contexts)
        train_pred = (train_scores >= 0.5).astype(np.int64)
        validation_pred = (validation_scores >= 0.5).astype(np.int64)
        train_row = binary_metric_row(
            model="MLP Classifier",
            split="train",
            task="local_failure prediction",
            positive_class="local_failure = 1",
            y_true=train.small_failure[indices],
            y_pred=train_pred,
            y_score=train_scores,
        )
        validation_row = binary_metric_row(
            model="MLP Classifier",
            split="validation",
            task="local_failure prediction",
            positive_class="local_failure = 1",
            y_true=validation.small_failure,
            y_pred=validation_pred,
            y_score=validation_scores,
        )
        mlp_curve_rows.append(
            {
                "training_fraction": fraction,
                "training_examples": int(len(indices)),
                "train_accuracy": train_row["accuracy"],
                "validation_accuracy": validation_row["accuracy"],
                "train_error": train_row["error"],
                "validation_error": validation_row["error"],
                "train_f1": train_row["f1"],
                "validation_f1": validation_row["f1"],
            }
        )
    mlp_curve = pd.DataFrame(mlp_curve_rows)
    mlp_curve.to_csv(args.results_dir / "mlp_learning_curve.csv", index=False)
    plot_mlp_learning_curve(mlp_curve, plots_dir / "mlp_learning_curve.png")
    plot_mlp_error_curve(mlp_curve, plots_dir / "mlp_train_val_error.png")

    # B/C. Utility Prediction Router traditional routing metrics and learning curves.
    with (args.model_dir / "utility_prediction_realtime.pkl").open("rb") as file:
        bundle = pickle.load(file)

    utility_metric_rows: list[dict[str, Any]] = []
    utility_pr_rows: list[dict[str, Any]] = []
    train_oracle_by_mode = {
        mode_name: realtime_oracle_actions(train, sla)
        for mode_name, sla in SLA_MODES.items()
    }
    utility_learning_rows: list[dict[str, Any]] = []

    for mode_name, sla in SLA_MODES.items():
        model = bundle["models"][mode_name]
        train_utilities = realtime_utility_matrix(train, sla)
        validation_utilities = realtime_utility_matrix(validation, sla)
        train_oracle = train_oracle_by_mode[mode_name]
        validation_oracle = realtime_oracle_actions(validation, sla)
        train_pred_utilities, train_actions = utility_predictions_to_actions(
            model,
            train.contexts,
        )
        validation_pred_utilities, validation_actions = utility_predictions_to_actions(
            model,
            validation.contexts,
        )
        validation_gap = validation_pred_utilities[:, LARGE] - validation_pred_utilities[:, SMALL]
        utility_pr_rows.append(
            {
                "mode": mode_name,
                "y_true": validation_oracle,
                "score": validation_gap,
            }
        )

        for split_name, y_true, y_pred, y_score in [
            (
                "train",
                train_oracle,
                train_actions,
                train_pred_utilities[:, LARGE] - train_pred_utilities[:, SMALL],
            ),
            ("validation", validation_oracle, validation_actions, validation_gap),
        ]:
            row = binary_metric_row(
                model="Utility Prediction Router",
                split=split_name,
                task=f"{mode_name} route classification vs oracle",
                positive_class="cloud route = 1",
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_score,
                notes="Utility score is predicted_utility_cloud - predicted_utility_local.",
            )
            row["sla_mode"] = mode_name
            row["routing_accuracy_vs_oracle"] = row["accuracy"]
            row["route_error"] = row["error"]
            row["precision_cloud"] = row["precision"]
            row["recall_cloud"] = row["recall"]
            row["f1_cloud"] = row["f1"]
            utility_metric_rows.append(row)

        draw_confusion_matrix(
            confusion_matrix(validation_oracle, validation_actions, labels=[0, 1]),
            title=f"Utility Router Confusion Matrix: {mode_name}",
            class_names=["oracle local", "oracle cloud"],
            output=plots_dir / f"utility_router_confusion_matrix_{mode_name}.png",
        )

        pca_dim = pca_dim_from_value(bundle["scenario"].get(f"{mode_name}_pca_dim"))
        ridge_alpha = float(bundle["scenario"].get(f"{mode_name}_ridge_alpha"))
        for fraction in FRACTIONS:
            indices = route_fraction_indices(train_oracle_by_mode, fraction, args.seed)
            candidate = train_utility_model(
                train.contexts[indices],
                train_utilities[indices],
                pca_dim=pca_dim,
                ridge_alpha=ridge_alpha,
            )
            subset_pred_utilities, subset_actions = utility_predictions_to_actions(
                candidate,
                train.contexts[indices],
            )
            val_pred_utilities, val_actions = utility_predictions_to_actions(
                candidate,
                validation.contexts,
            )
            subset_oracle = train_oracle[indices]
            train_row = binary_metric_row(
                model="Utility Prediction Router",
                split="train",
                task=f"{mode_name} route classification vs oracle",
                positive_class="cloud route = 1",
                y_true=subset_oracle,
                y_pred=subset_actions,
                y_score=subset_pred_utilities[:, LARGE] - subset_pred_utilities[:, SMALL],
            )
            validation_row = binary_metric_row(
                model="Utility Prediction Router",
                split="validation",
                task=f"{mode_name} route classification vs oracle",
                positive_class="cloud route = 1",
                y_true=validation_oracle,
                y_pred=val_actions,
                y_score=val_pred_utilities[:, LARGE] - val_pred_utilities[:, SMALL],
            )
            utility_learning_rows.append(
                {
                    "sla_mode": mode_name,
                    "training_fraction": fraction,
                    "training_examples": int(len(indices)),
                    "train_routing_accuracy_vs_oracle": train_row["accuracy"],
                    "validation_routing_accuracy_vs_oracle": validation_row["accuracy"],
                    "train_route_error": train_row["error"],
                    "validation_route_error": validation_row["error"],
                    "train_f1_cloud": train_row["f1"],
                    "validation_f1_cloud": validation_row["f1"],
                    "train_mse": float(mean_squared_error(train_utilities[indices], subset_pred_utilities)),
                    "validation_mse": float(mean_squared_error(validation_utilities, val_pred_utilities)),
                    "pca_dim": "no_pca" if pca_dim is None else pca_dim,
                    "ridge_alpha": ridge_alpha,
                }
            )

    utility_metrics = pd.DataFrame(utility_metric_rows)
    utility_metrics.to_csv(args.results_dir / "utility_router_routing_metrics.csv", index=False)
    utility_learning_curve = pd.DataFrame(utility_learning_rows)
    utility_learning_curve.to_csv(
        args.results_dir / "utility_router_learning_curve.csv",
        index=False,
    )
    plot_utility_pr_by_sla(
        utility_pr_rows,
        plots_dir / "utility_router_precision_recall_by_sla.png",
    )
    plot_utility_accuracy_by_sla(
        utility_metrics,
        plots_dir / "utility_router_routing_accuracy_by_sla.png",
    )
    plot_utility_learning_curve(
        utility_learning_curve,
        plots_dir / "utility_router_learning_curve.png",
    )
    plot_utility_error_curve(
        utility_learning_curve,
        plots_dir / "utility_router_train_val_error.png",
    )
    plot_utility_mse_curve(
        utility_learning_curve,
        plots_dir / "utility_router_mse_learning_curve.png",
    )

    # D. Combined final-report summary.
    validation_mlp = mlp_metrics[mlp_metrics["split"] == "validation"].iloc[0]
    summary_rows = [
        {
            "model": "MLP Classifier",
            "task": "Predict local_failure on validation set",
            "positive_class": "local_failure = 1",
            "accuracy_or_routing_accuracy": validation_mlp["accuracy"],
            "precision": validation_mlp["precision"],
            "recall": validation_mlp["recall"],
            "f1": validation_mlp["f1"],
            "error": validation_mlp["error"],
            "notes": "Traditional supervised classification metric; zero_division=0 where needed.",
        }
    ]
    for mode_name in SLA_MODES:
        row = utility_metrics[
            (utility_metrics["split"] == "validation")
            & (utility_metrics["sla_mode"] == mode_name)
        ].iloc[0]
        summary_rows.append(
            {
                "model": "Utility Prediction Router",
                "task": f"{mode_name} route classification vs oracle",
                "positive_class": "cloud route = 1",
                "accuracy_or_routing_accuracy": row["routing_accuracy_vs_oracle"],
                "precision": row["precision_cloud"],
                "recall": row["recall_cloud"],
                "f1": row["f1_cloud"],
                "error": row["route_error"],
                "notes": "Route classifier compared against oracle utility action; zero_division=0 where needed.",
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.results_dir / "traditional_ml_evaluation_summary.csv", index=False)

    generated_files = [
        args.results_dir / "mlp_classifier_metrics.csv",
        args.results_dir / "mlp_learning_curve.csv",
        args.results_dir / "utility_router_routing_metrics.csv",
        args.results_dir / "utility_router_learning_curve.csv",
        args.results_dir / "traditional_ml_evaluation_summary.csv",
        plots_dir / "mlp_classifier_confusion_matrix.png",
        plots_dir / "mlp_classifier_precision_recall_curve.png",
        plots_dir / "mlp_learning_curve.png",
        plots_dir / "mlp_train_val_error.png",
        plots_dir / "utility_router_confusion_matrix_quality_first.png",
        plots_dir / "utility_router_confusion_matrix_balanced.png",
        plots_dir / "utility_router_confusion_matrix_cost_sensitive.png",
        plots_dir / "utility_router_precision_recall_by_sla.png",
        plots_dir / "utility_router_routing_accuracy_by_sla.png",
        plots_dir / "utility_router_learning_curve.png",
        plots_dir / "utility_router_train_val_error.png",
        plots_dir / "utility_router_mse_learning_curve.png",
    ]

    print(f"train_contexts: {train.contexts.shape}")
    print(f"validation_contexts: {validation.contexts.shape}")
    print("\nTraditional ML evaluation summary:")
    print(summary.to_string(index=False))
    print("\nGenerated files checklist:")
    for path in generated_files:
        status = "OK" if path.exists() else "MISSING"
        print(f"[{status}] {path}")


if __name__ == "__main__":
    main()
