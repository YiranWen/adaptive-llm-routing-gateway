"""Create report-ready comparison plots for Ridge vs Neural utility routers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
METRICS_PATH = RESULTS / "neural_utility_comparison_metrics.csv"

SLA_ORDER = ["quality_first", "balanced", "cost_sensitive"]
MODEL_ORDER = ["Ridge Utility Router", "Neural Utility Router"]
MODEL_LABELS = {
    "Ridge Utility Router": "Ridge Regression",
    "Neural Utility Router": "Neural Network",
}
COLORS = {
    "Ridge Utility Router": "#8E8E93",
    "Neural Utility Router": "#2F6FDB",
    "Oracle": "#2A9D8F",
}


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def value_for(df: pd.DataFrame, mode: str, model: str, metric: str) -> float:
    return float(df[(df["mode"] == mode) & (df["model"] == model)][metric].iloc[0])


def grouped_bar(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    output: Path,
    include_oracle: bool = False,
    annotate_delta: bool = False,
) -> None:
    models = MODEL_ORDER + (["Oracle"] if include_oracle else [])
    x = np.arange(len(SLA_ORDER))
    width = 0.24 if include_oracle else 0.34
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for idx, model in enumerate(models):
        values = [value_for(df, mode, model, metric) for mode in SLA_ORDER]
        offset = (idx - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=MODEL_LABELS.get(model, model),
            color=COLORS[model],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    if annotate_delta:
        for i, mode in enumerate(SLA_ORDER):
            ridge = value_for(df, mode, "Ridge Utility Router", metric)
            neural = value_for(df, mode, "Neural Utility Router", metric)
            delta = neural - ridge
            y = max(ridge, neural) + 0.018
            ax.text(
                i,
                y,
                f"NN {delta:+.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLORS["Neural Utility Router"],
                fontweight="bold",
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, ["Quality-first", "Balanced", "Cost-sensitive"])
    ax.legend(frameon=False, loc="best")
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def error_reduction_plot(df: pd.DataFrame, output: Path) -> None:
    modes = SLA_ORDER
    mae_reduction = []
    regret_reduction = []
    for mode in modes:
        ridge_mae = value_for(df, mode, "Ridge Utility Router", "combined_mae")
        neural_mae = value_for(df, mode, "Neural Utility Router", "combined_mae")
        ridge_regret = value_for(df, mode, "Ridge Utility Router", "mean_regret")
        neural_regret = value_for(df, mode, "Neural Utility Router", "mean_regret")
        mae_reduction.append((ridge_mae - neural_mae) / ridge_mae * 100)
        regret_reduction.append((ridge_regret - neural_regret) / ridge_regret * 100)

    x = np.arange(len(modes))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars1 = ax.bar(
        x - width / 2,
        mae_reduction,
        width,
        label="MAE reduction",
        color="#2F6FDB",
    )
    bars2 = ax.bar(
        x + width / 2,
        regret_reduction,
        width,
        label="Regret reduction",
        color="#6B8E23",
    )
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("Neural Router Improvement Over Ridge", fontsize=14, fontweight="bold")
    ax.set_ylabel("Relative improvement")
    ax.set_xticks(x, ["Quality-first", "Balanced", "Cost-sensitive"])
    ax.set_ylim(0, max(mae_reduction + regret_reduction) * 1.25)
    ax.legend(frameon=False)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def summary_dashboard(df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    panels = [
        ("avg_utility", "Average Utility", "Higher is better", True),
        ("combined_mae", "Utility Prediction MAE", "Lower is better", False),
        ("mean_regret", "Mean Regret", "Lower is better", False),
        ("oracle_utility_ratio", "Oracle Utility Ratio", "Higher is better", True),
    ]
    x = np.arange(len(SLA_ORDER))
    width = 0.34
    for ax, (metric, title, ylabel, higher) in zip(axes.ravel(), panels):
        for idx, model in enumerate(MODEL_ORDER):
            values = [value_for(df, mode, model, metric) for mode in SLA_ORDER]
            ax.bar(
                x + (idx - 0.5) * width,
                values,
                width=width,
                label=MODEL_LABELS[model],
                color=COLORS[model],
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, ["Quality-first", "Balanced", "Cost-sensitive"])
        clean_axes(ax)
        if higher:
            low = min(value_for(df, mode, model, metric) for mode in SLA_ORDER for model in MODEL_ORDER)
            ax.set_ylim(max(0, low - 0.03), 1.005 if metric == "oracle_utility_ratio" else None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        "Ridge Regression vs Neural Network Utility Router",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS_PATH)
    df = df[df["model"].isin(MODEL_ORDER + ["Oracle"])].copy()

    grouped_bar(
        df,
        metric="avg_utility",
        title="Average Utility: Neural Router Wins Across SLA Modes",
        ylabel="Average selected true utility",
        output=PLOTS / "ridge_vs_nn_average_utility.png",
        include_oracle=True,
        annotate_delta=True,
    )
    grouped_bar(
        df,
        metric="combined_mae",
        title="Utility Prediction Error: Neural Router Has Lower MAE",
        ylabel="Combined MAE",
        output=PLOTS / "ridge_vs_nn_prediction_error_mae.png",
    )
    grouped_bar(
        df,
        metric="mean_regret",
        title="Regret vs Oracle: Neural Router Loses Less Utility",
        ylabel="Mean regret",
        output=PLOTS / "ridge_vs_nn_mean_regret.png",
    )
    grouped_bar(
        df,
        metric="oracle_utility_ratio",
        title="Oracle Utility Ratio: Neural Router Closer to Oracle",
        ylabel="Average utility / oracle average utility",
        output=PLOTS / "ridge_vs_nn_oracle_ratio.png",
    )
    error_reduction_plot(
        df,
        PLOTS / "ridge_vs_nn_relative_improvement.png",
    )
    summary_dashboard(
        df,
        PLOTS / "ridge_vs_nn_summary_dashboard.png",
    )

    generated = [
        PLOTS / "ridge_vs_nn_average_utility.png",
        PLOTS / "ridge_vs_nn_prediction_error_mae.png",
        PLOTS / "ridge_vs_nn_mean_regret.png",
        PLOTS / "ridge_vs_nn_oracle_ratio.png",
        PLOTS / "ridge_vs_nn_relative_improvement.png",
        PLOTS / "ridge_vs_nn_summary_dashboard.png",
    ]
    print("Generated Ridge vs Neural visualization files:")
    for path in generated:
        print(f"[{'OK' if path.exists() else 'MISSING'}] {path}")


if __name__ == "__main__":
    main()
