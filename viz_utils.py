import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Ensure a non-GUI backend is used for plotting
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Union


def plot_shap_importance(shap_importance: pd.DataFrame, out_path: Path):
    """Generates and saves the customized SHAP bar plot for feature importance."""
    topk = shap_importance.head(10)
    shap_values_max = shap_importance["importance"].max()
    
    plt.figure(figsize=(8, 6))
    color = "#ff006e"  # bright pink-red color

    # Plot bars (reversed order for horizontal bar chart)
    bars = plt.barh(topk["feature"][::-1], topk["importance"][::-1],
                    color=color, edgecolor="none")

    plt.xlabel("Mean |SHAP value|", fontsize=11)
    plt.title("Global Feature Importance", fontsize=13, fontweight="bold", loc="right")
    plt.grid(axis="x", linestyle=":", alpha=0.4)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)

    # Add numeric labels at end of bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.02 * shap_values_max,
                 bar.get_y() + bar.get_height()/2,
                 f"+{width:.2f}",
                 va='center', ha='left', fontsize=10, color=color, fontweight="bold")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_metrics_line_chart(df_results: pd.DataFrame, metrics: List[str], out_path: Path):
    """Generates and saves a line chart comparing performance metrics across settings."""
    plt.figure(figsize=(10, 6))
    for metric in metrics:
        plt.plot(df_results["Setting"], df_results[metric], marker="o", label=metric)

    plt.xlabel("Setting")
    plt.ylabel("Score")
    plt.title("Performance Comparison across Settings")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_metrics_bar_chart(df_results: pd.DataFrame, metrics: List[str], out_path: Path):
    """Generates and saves a grouped bar chart comparing performance metrics across settings."""
    settings = df_results["Setting"].tolist()
    x = np.arange(len(metrics))  # positions for metrics
    width = 0.15                  # bar width 
    
    plt.figure(figsize=(12, 6))

    # Plot each setting as a bar cluster
    for i, setting in enumerate(settings):
        plt.bar(
            x + i * width,
            df_results.loc[df_results["Setting"] == setting, metrics].values.flatten(),
            width,
            label=setting
        )

    plt.xticks(x + width * (len(settings)-1) / 2, metrics)
    plt.ylabel("Score")
    plt.title("Performance Comparison across Settings")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()