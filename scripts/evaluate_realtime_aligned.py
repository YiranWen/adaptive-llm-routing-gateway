"""Evaluate real-time aligned MLP and utility-prediction routers."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_router.mlp_router import train_mlp_router
from llm_router.offline_data import load_offline_artifacts
from llm_router.realtime import (
    LARGE,
    SMALL,
    augment_offline_split,
    realtime_oracle_actions,
    realtime_utility_matrix,
    realtime_route_metrics,
)
from llm_router.utilities import SLA_MODES


POLICIES = ["always_local", "always_cloud", "MLP", "utility_prediction", "oracle"]
RR_PCA_DIMS: list[int | None] = [None, 16, 32, 64, 128]
RR_RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


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


def save_plot_policy_comparison(metrics: pd.DataFrame, output: Path) -> None:
    modes = list(SLA_MODES)
    x = np.arange(len(modes))
    width = 0.14
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, policy in enumerate(POLICIES):
        values = [
            metrics[(metrics["mode"] == mode) & (metrics["policy"] == policy)][
                "avg_utility"
            ].iloc[0]
            for mode in modes
        ]
        ax.bar(x + (idx - 2) * width, values, width=width, label=policy)
    ax.set_title("Real-Time Aligned Average Utility")
    ax.set_xlabel("SLA mode")
    ax.set_ylabel("Average utility")
    ax.set_xticks(x, labels=modes)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_plot_cost_latency(metrics: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for policy in POLICIES:
        subset = metrics[metrics["policy"] == policy]
        ax.plot(subset["avg_cost"], subset["avg_latency"], marker="o", label=policy)
    ax.set_title("Real-Time Aligned Cost/Latency Trade-off")
    ax.set_xlabel("Average effective cloud cost")
    ax.set_ylabel("Average latency")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_plot_cloud_rate(metrics: pd.DataFrame, output: Path) -> None:
    modes = list(SLA_MODES)
    x = np.arange(len(modes))
    width = 0.14
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, policy in enumerate(POLICIES):
        values = [
            metrics[(metrics["mode"] == mode) & (metrics["policy"] == policy)][
                "cloud_route_rate"
            ].iloc[0]
            for mode in modes
        ]
        ax.bar(x + (idx - 2) * width, values, width=width, label=policy)
    ax.set_title("Real-Time Aligned Cloud Route Rate")
    ax.set_xlabel("SLA mode")
    ax.set_ylabel("Cloud route rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, labels=modes)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def train_utility_prediction_model(train_contexts: np.ndarray, utility_matrix: np.ndarray):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=64, random_state=184),
        Ridge(alpha=10.0),
    )
    model.fit(train_contexts, utility_matrix)
    return model


def pca_name(pca_dim: int | None) -> str:
    return "no_pca" if pca_dim is None else str(pca_dim)


def train_utility_prediction_model_config(
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

    mlp_result, mlp_actions = train_mlp_router(
        train.contexts,
        train.small_failure,
        validation.contexts,
        seed=args.seed,
    )

    rows: list[dict[str, object]] = []
    rr_ablation_rows: list[dict[str, object]] = []
    model_bundle: dict[str, object] = {
        "scenario": {
            "context_dim": int(train.contexts.shape[1]),
            "qwen_dim": int(artifacts.train.features.shape[1]),
            "system_feature_dim": int(train.system_features.shape[1]),
            "remote_cost_per_request": args.remote_cost_per_request,
        },
        "models": {},
    }

    for mode_name, sla in SLA_MODES.items():
        train_utilities = realtime_utility_matrix(train, sla)
        best_rr = None
        for pca_dim in RR_PCA_DIMS:
            for ridge_alpha in RR_RIDGE_ALPHAS:
                candidate_model = train_utility_prediction_model_config(
                    train.contexts,
                    train_utilities,
                    pca_dim=pca_dim,
                    ridge_alpha=ridge_alpha,
                )
                predictions = candidate_model.predict(validation.contexts)
                actions = np.argmax(predictions, axis=1).astype(np.int64)
                metric = realtime_route_metrics(
                    mode=mode_name,
                    policy="utility_prediction_ablation",
                    split=validation,
                    actions=actions,
                    sla=sla,
                )
                metric["pca_dim"] = pca_name(pca_dim)
                metric["ridge_alpha"] = ridge_alpha
                metric["validation_tuned"] = True
                rr_ablation_rows.append(metric)
                if best_rr is None or metric["avg_utility"] > best_rr["metric"]["avg_utility"]:
                    best_rr = {
                        "model": candidate_model,
                        "actions": actions,
                        "pca_dim": pca_dim,
                        "ridge_alpha": ridge_alpha,
                        "metric": metric,
                    }

        assert best_rr is not None
        utility_model = best_rr["model"]
        rr_actions = best_rr["actions"]
        model_bundle["models"][mode_name] = utility_model
        model_bundle["scenario"][f"{mode_name}_pca_dim"] = pca_name(best_rr["pca_dim"])
        model_bundle["scenario"][f"{mode_name}_ridge_alpha"] = best_rr["ridge_alpha"]

        policies = {
            "always_local": np.full(len(validation.contexts), SMALL, dtype=np.int64),
            "always_cloud": np.full(len(validation.contexts), LARGE, dtype=np.int64),
            "MLP": mlp_actions,
            "utility_prediction": rr_actions,
            "oracle": realtime_oracle_actions(validation, sla),
        }
        for policy, actions in policies.items():
            rows.append(
                {
                    **realtime_route_metrics(
                    mode=mode_name,
                    policy=policy,
                    split=validation,
                    actions=actions,
                    sla=sla,
                    ),
                    "validation_tuned": policy == "utility_prediction",
                    "pca_dim": pca_name(best_rr["pca_dim"]) if policy == "utility_prediction" else "",
                    "ridge_alpha": best_rr["ridge_alpha"] if policy == "utility_prediction" else "",
                }
            )

    metrics = pd.DataFrame(rows)
    metrics["mode"] = pd.Categorical(metrics["mode"], categories=list(SLA_MODES), ordered=True)
    metrics["policy"] = pd.Categorical(metrics["policy"], categories=POLICIES, ordered=True)
    metrics = metrics.sort_values(["mode", "policy"]).reset_index(drop=True)
    metrics_path = args.results_dir / "realtime_aligned_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    rr_ablation_path = args.results_dir / "realtime_utility_prediction_ablation.csv"
    pd.DataFrame(rr_ablation_rows).to_csv(rr_ablation_path, index=False)

    with (args.model_dir / "utility_prediction_realtime.pkl").open("wb") as file:
        pickle.dump(model_bundle, file)

    save_plot_policy_comparison(
        metrics,
        plots_dir / "realtime_aligned_policy_comparison.png",
    )
    save_plot_cost_latency(
        metrics,
        plots_dir / "realtime_aligned_cost_latency_tradeoff.png",
    )
    save_plot_cloud_rate(
        metrics,
        plots_dir / "realtime_aligned_cloud_rate.png",
    )

    print(f"train_contexts: {train.contexts.shape}")
    print(f"validation_contexts: {validation.contexts.shape}")
    print(f"mlp_train_loss: {mlp_result.train_loss:.4f}")
    print(f"metrics: {metrics_path}")
    print(f"utility_prediction_ablation: {rr_ablation_path}")
    print(f"model: {args.model_dir / 'utility_prediction_realtime.pkl'}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
