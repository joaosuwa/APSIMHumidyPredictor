"""Métricas e tabelas de previsão para os modelos de déficit hídrico."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: np.ndarray, predicted: np.ndarray, prefix: str) -> dict[str, float]:
    """Calcula métricas de regressão com nomes prefixados."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        f"mae_{prefix}": float(mean_absolute_error(actual, predicted)),
        f"rmse_{prefix}": float(np.sqrt(mean_squared_error(actual, predicted))),
        f"r2_{prefix}": float(r2_score(actual, predicted)),
        f"bias_{prefix}": float(np.mean(predicted - actual)),
    }


def evaluate_prediction(
    model_name: str,
    actual_variation: pd.Series,
    predicted_variation: np.ndarray,
    current_deficit: pd.Series,
    actual_next_deficit: pd.Series,
    validation_mae: float | None = None,
    selected: bool = False,
) -> dict[str, object]:
    """Avalia variação e déficit absoluto reconstruído."""
    predicted_variation = np.asarray(predicted_variation, dtype=float)
    predicted_next_deficit = current_deficit.to_numpy(float) + predicted_variation
    result: dict[str, object] = {
        "model": model_name,
        "selected_by_validation": selected,
        "validation_mae_variation": validation_mae,
    }
    result.update(regression_metrics(actual_variation.to_numpy(), predicted_variation, "variation"))
    result.update(
        regression_metrics(
            actual_next_deficit.to_numpy(),
            predicted_next_deficit,
            "next_deficit",
        )
    )
    return result


def persistence_baseline(frame: pd.DataFrame) -> np.ndarray:
    """Baseline: a variação prevista é zero, então Dr(D+1) = Dr(D)."""
    return np.zeros(len(frame), dtype=float)
