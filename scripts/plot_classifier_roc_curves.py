"""Generate ROC-AUC curve for the local_failure classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PLOTS = ROOT / "results" / "plots"


def main() -> None:
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

    train_indices, _dev_indices = train_test_split(
        np.arange(len(y_train)),
        test_size=0.25,
        random_state=184,
        stratify=y_train,
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=3.0, solver="liblinear", max_iter=1000),
    )
    classifier.fit(x_train[train_indices], y_train[train_indices])
    scores = classifier.predict_proba(x_validation)[:, 1]
    fpr, tpr, _thresholds = roc_curve(y_validation, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.plot(
        fpr,
        tpr,
        linewidth=2.4,
        color="#4477AA",
        label=f"MLP Classifier, AUC={roc_auc:.3f}",
    )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_title("ROC Curve for Local-Failure Classification")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate / Recall")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output = PLOTS / "classifier_roc_auc_curve.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"MLP Classifier ROC-AUC: {roc_auc:.4f}")
    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
