"""Métricas estatísticas reutilizáveis do projeto."""

from .forecast import (
    compare_forecast,
    detection_metrics,
    error_metrics,
)

__all__ = ["compare_forecast", "detection_metrics", "error_metrics"]
