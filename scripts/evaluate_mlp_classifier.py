"""Train the local-failure classifier with prompt-only features.

This report-facing classifier removes the 6D system features because
local_failure is a prompt-quality label rather than a runtime-system label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_f1 = -1.0
    best_threshold = 0.5
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (scores >= threshold).astype(np.int64)
        value = f1_score(y_true, predictions, zero_division=0)
        if value > best_f1:
            best_f1 = float(value)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def metric_row(
    *,
    model_name: str,
    split: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    threshold_source: str,
) -> dict[str, float | int | str]:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = accuracy_score(y_true, predictions)
    return {
        "model": model_name,
        "split": split,
        "positive_class": "local_failure = 1",
        "threshold": threshold,
        "threshold_source": threshold_source,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "error": float(1.0 - accuracy),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def save_confusion_matrix(cm: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("MLP Classifier Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], ["no failure", "failure"])
    ax.set_yticks([0, 1], ["no failure", "failure"])
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_pr_curve(y_true: np.ndarray, scores: np.ndarray, output: Path) -> None:
    precision, recall, _thresholds = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(recall, precision, linewidth=2, label=f"AP={ap:.3f}")
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


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    train_cache = np.load(ROOT / "data" / "processed" / "activation_features_train_512.npz", allow_pickle=True)
    validation_cache = np.load(
        ROOT / "data" / "processed" / "activation_features_validation_128.npz",
        allow_pickle=True,
    )
    x_train = train_cache["features"].astype(np.float32)
    y_train = train_cache["small_failure"].astype(np.int64)
    x_validation = validation_cache["features"].astype(np.float32)
    y_validation = validation_cache["small_failure"].astype(np.int64)

    train_indices, dev_indices = train_test_split(
        np.arange(len(y_train)),
        test_size=0.25,
        random_state=184,
        stratify=y_train,
    )

    # Chosen from train/dev ablation, not from validation tuning.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=3.0, solver="liblinear", max_iter=1000),
    )
    model.fit(x_train[train_indices], y_train[train_indices])
    dev_scores = model.predict_proba(x_train[dev_indices])[:, 1]
    threshold, dev_f1 = choose_threshold(y_train[dev_indices], dev_scores)

    train_scores = model.predict_proba(x_train[train_indices])[:, 1]
    validation_scores = model.predict_proba(x_validation)[:, 1]
    rows = [
        metric_row(
            model_name="MLP Classifier",
            split="internal_train",
            y_true=y_train[train_indices],
            scores=train_scores,
            threshold=threshold,
            threshold_source="internal_dev_f1",
        ),
        metric_row(
            model_name="MLP Classifier",
            split="internal_dev",
            y_true=y_train[dev_indices],
            scores=dev_scores,
            threshold=threshold,
            threshold_source="internal_dev_f1",
        ),
        metric_row(
            model_name="MLP Classifier",
            split="validation",
            y_true=y_validation,
            scores=validation_scores,
            threshold=threshold,
            threshold_source="internal_dev_f1",
        ),
    ]
    metrics = pd.DataFrame(rows)
    metrics["dev_f1_selected"] = dev_f1
    metrics.to_csv(RESULTS / "mlp_classifier_metrics.csv", index=False)

    validation_predictions = (validation_scores >= threshold).astype(np.int64)
    save_confusion_matrix(
        confusion_matrix(y_validation, validation_predictions, labels=[0, 1]),
        PLOTS / "mlp_classifier_confusion_matrix.png",
    )
    save_pr_curve(
        y_validation,
        validation_scores,
        PLOTS / "mlp_classifier_precision_recall_curve.png",
    )

    print("MLP Classifier metrics:")
    print(metrics.to_string(index=False))
    print("\nGenerated files:")
    for path in [
        RESULTS / "mlp_classifier_metrics.csv",
        PLOTS / "mlp_classifier_confusion_matrix.png",
        PLOTS / "mlp_classifier_precision_recall_curve.png",
    ]:
        print(f"[{'OK' if path.exists() else 'MISSING'}] {path}")


if __name__ == "__main__":
    main()
