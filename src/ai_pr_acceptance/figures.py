from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import shap


def save_metrics_bar(metrics: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(metrics["model"], metrics["gb_auc_mean"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("ROC-AUC")
    plt.title("Model Performance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_shap_outputs(model, X: pd.DataFrame, output_prefix: Path) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, max_display=20, show=False, plot_type="dot")
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(output_prefix.name + "_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
