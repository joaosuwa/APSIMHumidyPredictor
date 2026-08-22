"""Métricas e tabelas de avaliação para regressão de déficit hídrico."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    """Calcula as métricas oficiais com unidades consistentes em milímetros."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Observado e previsto devem possuir o mesmo formato")
    if actual_values.size == 0:
        raise ValueError("Não é possível avaliar vetores vazios")
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "r2": float(r2_score(actual_values, predicted_values)),
        "bias": float(np.mean(predicted_values - actual_values)),
    }


def persistence_prediction(current_deficit: Iterable[float]) -> np.ndarray:
    """Baseline operacional: prevê que o déficit de D+1 será igual ao de D."""
    return np.asarray(current_deficit, dtype=float).copy()
