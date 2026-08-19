"""Métricas de erro e de detecção para séries de previsão."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detection_metrics(
    forecast: pd.Series,
    observed: pd.Series,
    threshold: float,
) -> dict:
    """Calcula métricas de detecção de eventos acima de um limiar."""
    hits = int(((forecast >= threshold) & (observed >= threshold)).sum())
    misses = int(((forecast < threshold) & (observed >= threshold)).sum())
    false_alarms = int(((forecast >= threshold) & (observed < threshold)).sum())
    correct_neg = int(((forecast < threshold) & (observed < threshold)).sum())
    pod = hits / (hits + misses) if hits + misses else np.nan
    far = false_alarms / (hits + false_alarms) if hits + false_alarms else np.nan
    csi = hits / (hits + misses + false_alarms) if hits + misses + false_alarms else np.nan
    total = hits + misses + false_alarms + correct_neg
    accuracy = (hits + correct_neg) / total if total else np.nan
    return {
        "limiar": threshold,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_neg": correct_neg,
        "pod": pod,
        "far": far,
        "csi": csi,
        "acuracia": accuracy,
    }


def error_metrics(forecast: pd.Series, observed: pd.Series) -> dict:
    """Calcula MAE, RMSE, viés e correlação entre duas séries."""
    residual = forecast - observed
    n = int(observed.size)
    if n == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "vies": np.nan,
            "correlacao": np.nan,
        }
    correlation = float(np.corrcoef(forecast, observed)[0, 1]) if n > 2 else np.nan
    return {
        "n": n,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "vies": float(np.mean(residual)),
        "correlacao": correlation,
    }


def compare_forecast(
    forecast: pd.Series,
    observed: pd.Series,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """Calcula métricas gerais e agrupadas por mês para uma previsão."""
    merged = pd.DataFrame({"previsao": forecast, "observado": observed}).dropna().sort_index()
    if merged.empty:
        return pd.DataFrame()

    rows = [{
        "grupo": "geral",
        "mes": np.nan,
        **error_metrics(merged["previsao"], merged["observado"]),
        **detection_metrics(merged["previsao"], merged["observado"], threshold),
    }]
    merged["mes"] = merged.index.month
    for month, group in merged.groupby("mes"):
        rows.append({
            "grupo": "por_mes",
            "mes": int(month),
            **error_metrics(group["previsao"], group["observado"]),
            **detection_metrics(group["previsao"], group["observado"], threshold),
        })
    return pd.DataFrame(rows)
