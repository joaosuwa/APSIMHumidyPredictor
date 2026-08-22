"""Gráficos determinísticos do tuning e da avaliação final."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def _save(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_metric_comparison(
    metrics: pd.DataFrame, title: str, path: Path, champion: str
) -> Path:
    """Compara MAE entre modelos e baseline em um corte."""
    ordered = metrics.sort_values("mae", ascending=True)
    colors = ["#F28E2B" if name == champion else "#4C78A8" for name in ordered["model"]]
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(ordered["model"], ordered["mae"], color=colors)
    plt.ylabel("MAE (mm)")
    plt.title(title)
    plt.xticks(rotation=15)
    for bar, value in zip(bars, ordered["mae"], strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return _save(path)


def plot_observed_predicted(
    actual: Sequence[float], predicted: Sequence[float], model_name: str, path: Path
) -> Path:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    lower = float(min(actual_values.min(), predicted_values.min()))
    upper = float(max(actual_values.max(), predicted_values.max()))
    plt.figure(figsize=(6, 6))
    plt.scatter(actual_values, predicted_values, s=14, alpha=0.45, color="#4C78A8")
    plt.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
    plt.xlabel("Déficit observado em D+1 (mm)")
    plt.ylabel("Déficit previsto em D+1 (mm)")
    plt.title(f"Observado versus previsto — {model_name}")
    return _save(path)


def plot_residuals(
    actual: Sequence[float], predicted: Sequence[float], model_name: str, path: Path
) -> Path:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    residuals = predicted_values - actual_values
    plt.figure(figsize=(8, 4.8))
    plt.scatter(predicted_values, residuals, s=14, alpha=0.45, color="#4C78A8")
    plt.axhline(0.0, linestyle="--", color="black", linewidth=1)
    plt.xlabel("Déficit previsto em D+1 (mm)")
    plt.ylabel("Resíduo: previsto - observado (mm)")
    plt.title(f"Resíduos no teste — {model_name}")
    return _save(path)


def plot_optimization_history(studies: Mapping[str, Any], path: Path) -> Path:
    """Mostra apenas trials completos e a melhor MAE acumulada."""
    plt.figure(figsize=(9, 5.2))
    for model_name, study in studies.items():
        completed = [
            trial
            for trial in study.trials
            if trial.value is not None and trial.state.name == "COMPLETE"
        ]
        if not completed:
            continue
        numbers = [trial.number for trial in completed]
        values = np.asarray([float(trial.value) for trial in completed])
        plt.scatter(numbers, values, s=16, alpha=0.28)
        plt.plot(numbers, np.minimum.accumulate(values), linewidth=2, label=model_name)
    plt.xlabel("Número do trial")
    plt.ylabel("MAE OOF (mm)")
    plt.title("Histórico de otimização Optuna")
    plt.legend()
    plt.grid(alpha=0.2)
    return _save(path)


def plot_fold_mae(fold_metrics: pd.DataFrame, path: Path) -> Path:
    pivot = fold_metrics.pivot(
        index="validation_cycle_id", columns="model", values="mae"
    ).sort_index()
    ax = pivot.plot(kind="bar", figsize=(10, 5.2), width=0.78)
    ax.set_xlabel("cycle_id de validação")
    ax.set_ylabel("MAE (mm)")
    ax.set_title("MAE por fold de desenvolvimento")
    ax.legend(title="Modelo")
    plt.xticks(rotation=0)
    return _save(path)


def permutation_feature_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    seed: int,
    results_path: Path,
    plot_path: Path,
) -> pd.DataFrame:
    """Calcula importância comparável pelo aumento de MAE no holdout."""
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
    results_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(results_path, index=False)

    top = importance.head(20).sort_values("importance_mean")
    plt.figure(figsize=(9, 7))
    plt.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color="#4C78A8",
    )
    plt.xlabel("Aumento do MAE após permutação (mm)")
    plt.title("Permutation importance do campeão — top 20")
    _save(plot_path)
    return importance
