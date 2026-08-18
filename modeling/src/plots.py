"""Gráficos comparativos e explicabilidade do modelo selecionado."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Patch
from sklearn.inspection import permutation_importance


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_model_comparison(metrics: pd.DataFrame, plots_dir: Path) -> None:
    ordered = metrics.sort_values("mae_variation")
    colors = [
        "#F28E2B" if bool(selected) else "#4C78A8"
        for selected in ordered["selected_by_validation"]
    ]
    plt.figure(figsize=(8, 4.5))
    plt.bar(ordered["model"], ordered["mae_variation"], color=colors)
    plt.ylabel("MAE da variação do déficit (mm)")
    plt.title("Comparação no teste temporal")
    plt.xticks(rotation=20)
    plt.legend(
        handles=[
            Patch(color="#F28E2B", label="Selecionado pela validação"),
            Patch(color="#4C78A8", label="Concorrente/baseline"),
        ]
    )
    _save(plots_dir / "comparacao_mae_modelos.png")


def plot_observed_predicted(
    actual: pd.Series,
    predicted: np.ndarray,
    plots_dir: Path,
) -> None:
    actual_values = actual.to_numpy(float)
    predicted_values = np.asarray(predicted, dtype=float)
    lower = min(actual_values.min(), predicted_values.min())
    upper = max(actual_values.max(), predicted_values.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(actual_values, predicted_values, s=12, alpha=0.45)
    plt.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
    plt.xlabel("Déficit observado em D+1 (mm)")
    plt.ylabel("Déficit previsto em D+1 (mm)")
    plt.title("Observado versus previsto")
    _save(plots_dir / "observado_vs_previsto.png")


def plot_residuals(
    predicted: np.ndarray,
    actual: pd.Series,
    plots_dir: Path,
) -> None:
    predicted_values = np.asarray(predicted, dtype=float)
    residuals = predicted_values - actual.to_numpy(float)
    plt.figure(figsize=(8, 4.5))
    plt.scatter(predicted_values, residuals, s=12, alpha=0.45)
    plt.axhline(0.0, linestyle="--", color="black", linewidth=1)
    plt.xlabel("Déficit previsto em D+1 (mm)")
    plt.ylabel("Resíduo: previsto - observado (mm)")
    plt.title("Resíduos do modelo selecionado")
    _save(plots_dir / "residuos.png")


def permutation_feature_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results_dir: Path,
    plots_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=seed,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(results_dir / "permutation_importance.csv", index=False)

    top = importance.head(20).sort_values("importance_mean")
    plt.figure(figsize=(8, 7))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.xlabel("Aumento médio do MAE após permutação")
    plt.title("Permutation importance — top 20")
    _save(plots_dir / "permutation_importance.png")
    return importance


def shap_feature_importance(
    model,
    X_test: pd.DataFrame,
    results_dir: Path,
    plots_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    sample = X_test.sample(min(1000, len(X_test)), random_state=seed)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    values_array = np.asarray(values)
    importance = pd.DataFrame(
        {
            "feature": sample.columns,
            "mean_abs_shap": np.abs(values_array).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(results_dir / "shap_importance.csv", index=False)

    shap.summary_plot(values_array, sample, show=False, max_display=20)
    plt.title("SHAP — modelo selecionado")
    _save(plots_dir / "shap_summary.png")
    return importance
