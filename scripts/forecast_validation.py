"""Orquestra a validação das previsões de chuva do GFS contra a NASA POWER."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .data_io import write_csv
    from .gfs_data_extractor.forecast import (
        build_daily_forecast,
        filter_future,
        read_forecast,
    )
    from .metrics.forecast import (
        compare_forecast,
        detection_metrics as _detection_metrics,
        error_metrics as _error_metrics,
    )
    from .nasa_power.processing import read_precipitation_series
    from .paths import PROCESSED_VALIDATION_DIR, RAW_GFS_DIR, RAW_NASA_POWER_DIR
except ImportError:  # Permite executar este arquivo diretamente.
    from data_io import write_csv
    from gfs_data_extractor.forecast import build_daily_forecast, filter_future, read_forecast
    from metrics.forecast import (
        compare_forecast,
        detection_metrics as _detection_metrics,
        error_metrics as _error_metrics,
    )
    from nasa_power.processing import read_precipitation_series
    from paths import PROCESSED_VALIDATION_DIR, RAW_GFS_DIR, RAW_NASA_POWER_DIR


FORECAST_PAIRS = [
    (
        RAW_GFS_DIR / "PrevisãoChuva24h_Alegrete_2019-2026.csv",
        RAW_NASA_POWER_DIR / "ClimaAlegrete2019-2026.csv",
        "Alegrete",
    ),
    (
        RAW_GFS_DIR / "PrevisãoChuva24H_NovaRamada_2019-2026.csv",
        RAW_NASA_POWER_DIR / "ClimaNovaRamada_2019-2026.csv",
        "NovaRamada",
    ),
]

DEFAULT_HORARIOS = ["00:00", "06:00", "12:00", "18:00"]


def read_observed(path: str | Path) -> pd.Series:
    """Alias compatível para a leitura da precipitação observada da NASA POWER."""
    return read_precipitation_series(path)


def compare_all(
    horario: str = "00:00",
    threshold: float = 1.0,
    reference_date: str | pd.Timestamp | None = None,
    horizon_days: int | None = None,
    output_dir: str | Path = PROCESSED_VALIDATION_DIR,
) -> dict[str, pd.DataFrame]:
    """Lê as fontes, calcula a validação e salva os resultados em CSV."""
    output_dir = Path(output_dir)
    series_frames = []
    metrics_frames = []
    future_info = {}

    for forecast_path, observed_path, local in FORECAST_PAIRS:
        forecast = read_forecast(forecast_path)
        observed = read_observed(observed_path)
        daily = build_daily_forecast(forecast, horario=horario)
        future_info[local] = len(filter_future(daily, reference_date, horizon_days))

        metrics = compare_forecast(daily, observed, threshold=threshold)
        if not metrics.empty:
            metrics.insert(0, "local", local)
            metrics_frames.append(metrics)

        series = pd.DataFrame({"previsao": daily, "observado": observed}).dropna().sort_index()
        series = series.reset_index()
        series.insert(0, "local", local)
        series.columns = ["local", "data", "previsao", "observado"]
        series_frames.append(series)

    series_all = pd.concat(series_frames, ignore_index=True)
    metrics_all = pd.concat(metrics_frames, ignore_index=True)
    metrics_all.insert(0, "horario", horario)
    series_path = output_dir / "serie_compare_previsao_chuva.csv"
    metrics_path = output_dir / "metricas_previsao_chuva.csv"
    write_csv(series_all, series_path)
    write_csv(metrics_all, metrics_path)

    print(f"horario={horario} limiar={threshold}mm")
    for local, count in future_info.items():
        print(f"  {local}: dias futuros a partir de {reference_date}: {count}")
    print(f"serie salva em: {series_path}")
    print(f"metricas salvas em: {metrics_path}")
    return {"serie": series_all, "metricas": metrics_all}


if __name__ == "__main__":
    result = compare_all(
        horario="00:00",
        threshold=1.0,
        reference_date="2026-08-01",
        horizon_days=7,
    )
    columns = [
        "local", "grupo", "mes", "n", "mae", "rmse", "vies",
        "correlacao", "pod", "far", "csi", "acuracia",
    ]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(result["metricas"].loc[result["metricas"]["grupo"] == "geral", columns].to_string(index=False))
