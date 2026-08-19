"""Orquestra a validação das previsões de chuva do GFS contra a NASA POWER."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:  # Compatibilidade com ``python scripts/forecast_validation.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.data_io import write_csv
    from scripts.gfs_data_extractor.gfs_config import SITES
    from scripts.gfs_data_extractor.gfs_data_processing import load_daily_24h_product
    from scripts.metrics.forecast import compare_forecast
    from scripts.nasa_power.processing import read_nasa_power_data
    from scripts.paths import PROCESSED_VALIDATION_DIR, RAW_NASA_POWER_DIR
else:
    from .data_io import write_csv
    from .gfs_data_extractor.gfs_config import SITES
    from .gfs_data_extractor.gfs_data_processing import load_daily_24h_product
    from .metrics.forecast import compare_forecast
    from .nasa_power.processing import read_nasa_power_data
    from .paths import PROCESSED_VALIDATION_DIR, RAW_NASA_POWER_DIR

DEFAULT_HORARIOS = ["00:00", "06:00", "12:00", "18:00"]


def _filter_future(
    forecast: pd.Series,
    reference_date: str | pd.Timestamp | None,
    horizon_days: int | None,
) -> pd.Series:
    if reference_date is None:
        return forecast
    start = pd.Timestamp(reference_date).normalize()
    end = None if horizon_days is None else start + pd.Timedelta(days=horizon_days - 1)
    return forecast.loc[start:end]


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

    hour = pd.to_datetime(horario, format="%H:%M").hour
    for site in SITES.values():
        rain = load_daily_24h_product("apcp_24", site, hour=hour)
        daily = rain.set_index("date")["previsao_chuva_24h_mm"]
        observed_data = read_nasa_power_data(RAW_NASA_POWER_DIR / site.nasa_raw_filename)
        observed = observed_data.set_index("date")["PRECTOTCORR"]
        future_info[site.name] = len(_filter_future(daily, reference_date, horizon_days))

        metrics = compare_forecast(daily, observed, threshold=threshold)
        if not metrics.empty:
            metrics.insert(0, "local", site.name)
            metrics_frames.append(metrics)

        series = pd.DataFrame({"previsao": daily, "observado": observed}).dropna().sort_index()
        series = series.reset_index()
        series.insert(0, "local", site.name)
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
