"""Gráficos determinísticos do tuning e da avaliação residual."""

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
    metrics: pd.DataFrame,
    title: str,
    path: Path,
    champion: str,
    *,
    metric_column: str = "rmse_variation",
    metric_label: str = "RMSE da variação (mm)",
) -> Path:
    ordered = metrics.sort_values(metric_column, ascending=True)
    colors = ["#F28E2B" if name == champion else "#4C78A8" for name in ordered["model"]]
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(ordered["model"], ordered[metric_column], color=colors)
    plt.ylabel(metric_label)
    plt.title(title)
    plt.xticks(rotation=15)
    for bar, value in zip(bars, ordered[metric_column], strict=True):
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
    actual: Sequence[float],
    predicted: Sequence[float],
    model_name: str,
    path: Path,
    *,
    value_label: str,
) -> Path:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    lower = float(min(actual_values.min(), predicted_values.min()))
    upper = float(max(actual_values.max(), predicted_values.max()))
    plt.figure(figsize=(6, 6))
    plt.scatter(actual_values, predicted_values, s=14, alpha=0.45, color="#4C78A8")
    plt.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
    plt.xlabel(f"{value_label} observado (mm)")
    plt.ylabel(f"{value_label} previsto (mm)")
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
    plt.xlabel("Variação prevista (mm)")
    plt.ylabel("Resíduo: previsto - observado (mm)")
    plt.title(f"Resíduos da variação no teste — {model_name}")
    return _save(path)


def plot_optimization_history(
    studies: Mapping[str, Any], path: Path, *, metric: str
) -> Path:
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
    plt.ylabel(f"{metric.upper()} OOF da variação (mm)")
    plt.title("Histórico de otimização Optuna")
    plt.legend()
    plt.grid(alpha=0.2)
    return _save(path)


def plot_fold_metric(
    fold_metrics: pd.DataFrame, path: Path, *, metric: str
) -> Path:
    pivot = fold_metrics.pivot(
        index="validation_cycle_id", columns="model", values=metric
    ).sort_index()
    ax = pivot.plot(kind="bar", figsize=(10, 5.2), width=0.78)
    ax.set_xlabel("cycle_id de validação")
    ax.set_ylabel(f"{metric.upper()} da variação (mm)")
    ax.set_title(f"{metric.upper()} por fold de desenvolvimento")
    ax.legend(title="Modelo")
    plt.xticks(rotation=0)
    return _save(path)


def permutation_feature_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    seed: int,
    model_name: str,
    metric: str,
    results_path: Path,
    plot_path: Path,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=(
            "neg_mean_absolute_error"
            if metric == "mae"
            else "neg_root_mean_squared_error"
        ),
        n_repeats=10,
        random_state=seed,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "model": model_name,
            "metric": metric,
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
    plt.xlabel(f"Aumento do {metric.upper()} após permutação (mm)")
    plt.title(f"Permutation importance — {model_name} — top 20")
    _save(plot_path)
    return importance


def plot_feature_importance_comparison(
    importances: Mapping[str, pd.DataFrame], path: Path, *, top_n: int = 15
) -> Path:
    """Compara, na mesma escala, as features mais importantes dos modelos."""
    combined = pd.concat(
        [
            table.loc[:, ["feature", "importance_mean"]].assign(model=model_name)
            for model_name, table in importances.items()
        ],
        ignore_index=True,
    )
    selected = (
        combined.groupby("feature")["importance_mean"]
        .max()
        .nlargest(top_n)
        .index
    )
    pivot = (
        combined.loc[combined["feature"].isin(selected)]
        .pivot(index="feature", columns="model", values="importance_mean")
        .fillna(0.0)
    )
    pivot = pivot.loc[pivot.max(axis=1).sort_values().index]
    ax = pivot.plot(kind="barh", figsize=(11, 7.5), width=0.8)
    ax.set_xlabel("Aumento do RMSE após permutação (mm)")
    ax.set_ylabel("Feature")
    ax.set_title("Permutation importance comparativa — melhores modelos")
    ax.legend(title="Modelo")
    return _save(path)


def plot_rain_rmse(rain_metrics: pd.DataFrame, path: Path) -> Path:
    """Plota o RMSE dos modelos em cada faixa de chuva observada em D+1."""
    order = ["sem_chuva", "fraca", "moderada", "forte"]
    pivot = rain_metrics.pivot(
        index="rain_intensity", columns="model", values="rmse"
    ).reindex(order)
    ax = pivot.plot(kind="bar", figsize=(10, 5.8), width=0.8)
    ax.set_xlabel("Intensidade da chuva observada em D+1")
    ax.set_ylabel("RMSE da variação (mm)")
    ax.set_title("Desvio dos modelos por intensidade de chuva")
    ax.legend(title="Modelo")
    plt.xticks(rotation=0)
    return _save(path)


def plot_residuals_by_observed_rain(
    observed_next_day_rain: Sequence[float],
    actual_variation: Sequence[float],
    predictions: Mapping[str, Sequence[float]],
    path: Path,
) -> Path:
    """Relaciona chuva real e resíduo; positivo subestima a queda do déficit."""
    rain = np.asarray(observed_next_day_rain, dtype=float)
    actual = np.asarray(actual_variation, dtype=float)
    model_names = list(predictions)
    figure, axes = plt.subplots(
        len(model_names), 1, figsize=(9, 3.2 * len(model_names)), sharex=True, sharey=True
    )
    axes_array = np.atleast_1d(axes)
    for axis, model_name in zip(axes_array, model_names, strict=True):
        residual = np.asarray(predictions[model_name], dtype=float) - actual
        axis.scatter(rain, residual, s=13, alpha=0.4, color="#4C78A8")
        axis.axhline(0.0, linestyle="--", color="black", linewidth=1)
        axis.set_title(model_name)
        axis.set_ylabel("Previsto - observado (mm)")
        axis.grid(alpha=0.15)
    axes_array[-1].set_xlabel("Precipitação observada em D+1 (mm)")
    figure.suptitle(
        "Resíduo da variação versus chuva real\n"
        "Resíduo positivo = redução do déficit subestimada",
        y=1.01,
    )
    return _save(path)
