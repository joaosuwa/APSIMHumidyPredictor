"""Métricas para a variação e o déficit absoluto reconstruído."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RAIN_BINS = (-np.inf, 0.1, 5.0, 20.0, np.inf)
RAIN_LABELS = ("sem_chuva", "fraca", "moderada", "forte")


def regression_metrics(
    actual: Iterable[float], predicted: Iterable[float]
) -> dict[str, float]:
    """Calcula as métricas básicas de regressão."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Observado e previsto devem possuir o mesmo formato")
    if actual_values.size == 0:
        raise ValueError("Não é possível avaliar vetores vazios")
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "r2": (
            float(r2_score(actual_values, predicted_values))
            if actual_values.size >= 2
            else float("nan")
        ),
        "bias": float(np.mean(predicted_values - actual_values)),
    }


def zero_variation_prediction(size_or_values: int | Iterable[float]) -> np.ndarray:
    """Baseline residual: nenhuma mudança prevista entre D e D+1."""
    if isinstance(size_or_values, int):
        if size_or_values < 0:
            raise ValueError("O tamanho da baseline não pode ser negativo")
        size = size_or_values
    else:
        size = np.asarray(list(size_or_values)).size
    return np.zeros(size, dtype=float)


def _prefixed(metrics: Mapping[str, float], suffix: str) -> dict[str, float]:
    return {f"{name}_{suffix}": value for name, value in metrics.items()}


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def evaluate_variation_prediction(
    actual_variation: Iterable[float],
    predicted_variation: Iterable[float],
    current_deficit: Iterable[float],
    actual_next_deficit: Iterable[float],
) -> dict[str, float]:
    """Avalia o target residual e a previsão absoluta reconstruída."""
    actual = np.asarray(actual_variation, dtype=float)
    predicted = np.asarray(predicted_variation, dtype=float)
    current = np.asarray(current_deficit, dtype=float)
    actual_next = np.asarray(actual_next_deficit, dtype=float)
    if not (actual.shape == predicted.shape == current.shape == actual_next.shape):
        raise ValueError("Todos os vetores de avaliação devem possuir o mesmo formato")
    predicted_next = current + predicted

    actual_reduction = actual < 0.0
    predicted_reduction = predicted < 0.0
    true_reduction = int((actual_reduction & predicted_reduction).sum())
    result = {
        **_prefixed(regression_metrics(actual, predicted), "variation"),
        **_prefixed(regression_metrics(actual_next, predicted_next), "next_deficit"),
        "direction_accuracy": float((actual_reduction == predicted_reduction).mean()),
        "reduction_precision": _safe_ratio(
            true_reduction, int(predicted_reduction.sum())
        ),
        "reduction_recall": _safe_ratio(true_reduction, int(actual_reduction.sum())),
    }
    return result


def event_metrics(
    model_name: str,
    actual_variation: Iterable[float],
    predicted_variation: Iterable[float],
) -> pd.DataFrame:
    """Detalha erro nos regimes que importam para a decisão de irrigação."""
    actual = np.asarray(actual_variation, dtype=float)
    predicted = np.asarray(predicted_variation, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Observado e previsto devem possuir o mesmo formato")
    masks = {
        "increase_or_stable": actual >= 0.0,
        "reduction": actual < 0.0,
        "large_reduction": actual <= -5.0,
    }
    records = []
    for event, mask in masks.items():
        if not mask.any():
            continue
        records.append(
            {
                "model": model_name,
                "event": event,
                "rows": int(mask.sum()),
                **regression_metrics(actual[mask], predicted[mask]),
                "direction_accuracy": float(
                    ((actual[mask] < 0.0) == (predicted[mask] < 0.0)).mean()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def rain_error_metrics(
    model_name: str,
    observed_next_day_rain: Iterable[float],
    actual_variation: Iterable[float],
    predicted_variation: Iterable[float],
) -> pd.DataFrame:
    """Resume o erro da variação por intensidade da chuva observada em D+1."""
    rain = np.asarray(observed_next_day_rain, dtype=float)
    actual = np.asarray(actual_variation, dtype=float)
    predicted = np.asarray(predicted_variation, dtype=float)
    if not (rain.shape == actual.shape == predicted.shape):
        raise ValueError("Chuva, observado e previsto devem possuir o mesmo formato")
    if rain.size == 0:
        raise ValueError("Não é possível avaliar chuva em vetores vazios")
    if not np.isfinite(rain).all() or (rain < 0).any():
        raise ValueError("Chuva observada deve ser finita e não negativa")

    categories = pd.cut(
        rain,
        bins=RAIN_BINS,
        labels=RAIN_LABELS,
        right=True,
        include_lowest=True,
    )
    records: list[dict[str, float | int | str]] = []
    for label in RAIN_LABELS:
        mask = np.asarray(categories == label)
        if not mask.any():
            records.append(
                {
                    "model": model_name,
                    "rain_intensity": label,
                    "rows": 0,
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "bias": float("nan"),
                }
            )
            continue
        metrics = regression_metrics(actual[mask], predicted[mask])
        records.append(
            {
                "model": model_name,
                "rain_intensity": label,
                "rows": int(mask.sum()),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "bias": metrics["bias"],
            }
        )
    return pd.DataFrame.from_records(records)
