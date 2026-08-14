"""Validacao das previsoes de chuva do GFS contra a NASA POWER."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .feature_engineering import read_csv_files, read_nasa_power_data, write_csv
    from .paths import PROCESSED_VALIDATION_DIR, RAW_GFS_DIR, RAW_NASA_POWER_DIR
except ImportError:  # Permite executar este arquivo diretamente.
    from feature_engineering import read_csv_files, read_nasa_power_data, write_csv
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


def read_forecast(path: str | Path) -> pd.DataFrame:
    """Le um CSV de previsao e devolve timestamp, data, hora e precipitacao."""
    forecast = read_csv_files(path)
    forecast.columns = ["Date", "Time", "precip"]
    hour = pd.to_datetime(forecast["Time"], format="%H:%M").dt.hour
    forecast["ts"] = pd.to_datetime(forecast["Date"]) + pd.to_timedelta(hour, unit="h")
    forecast = forecast.sort_values("ts").reset_index(drop=True)
    forecast["date"] = forecast["ts"].dt.date
    forecast["hour"] = hour.astype(int)
    return forecast[["ts", "date", "hour", "precip"]]


def read_observed(path: str | Path) -> pd.Series:
    """Le a precipitacao observada da NASA POWER; -999 vira NaN."""
    observed = read_nasa_power_data(path)
    return observed.set_index("date")["PRECTOTCORR"]


def build_daily_forecast(forecast: pd.DataFrame, horario: str = "00:00") -> pd.Series:
    """Relaciona a previsao de D ao registro de D+1 no horario escolhido."""
    hour = int(pd.to_datetime(horario, format="%H:%M").hour)
    daily = forecast.pivot_table(index="date", columns="hour", values="precip")
    values = daily.get(hour).copy()
    values.index = pd.to_datetime(values.index) - pd.Timedelta(days=1)
    values.name = "previsao"
    return values


def filter_future(
    daily_forecast: pd.Series,
    reference_date: str | pd.Timestamp | None = None,
    horizon_days: int | None = None,
) -> pd.Series:
    """Mantem apenas previsoes a partir da data de referencia."""
    if reference_date is None:
        return daily_forecast.copy()
    start = pd.Timestamp(reference_date).normalize()
    mask = daily_forecast.index >= start
    if horizon_days is not None:
        end = start + pd.Timedelta(days=horizon_days - 1)
        mask &= daily_forecast.index <= end
    return daily_forecast.loc[mask]


def _detection_metrics(forecast: pd.Series, observed: pd.Series, threshold: float) -> dict:
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


def _error_metrics(forecast: pd.Series, observed: pd.Series) -> dict:
    residual = forecast - observed
    n = int(observed.size)
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "vies": np.nan, "correlacao": np.nan}
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
    """Calcula metricas de erro e deteccao gerais e por mes."""
    merged = pd.DataFrame({"previsao": forecast, "observado": observed}).dropna().sort_index()
    if merged.empty:
        return pd.DataFrame()

    rows = [{
        "grupo": "geral",
        "mes": np.nan,
        **_error_metrics(merged["previsao"], merged["observado"]),
        **_detection_metrics(merged["previsao"], merged["observado"], threshold),
    }]
    merged["mes"] = merged.index.month
    for mes, group in merged.groupby("mes"):
        rows.append({
            "grupo": "por_mes",
            "mes": int(mes),
            **_error_metrics(group["previsao"], group["observado"]),
            **_detection_metrics(group["previsao"], group["observado"], threshold),
        })
    return pd.DataFrame(rows)


def compare_all(
    horario: str = "00:00",
    threshold: float = 1.0,
    reference_date: str | pd.Timestamp | None = None,
    horizon_days: int | None = None,
    output_dir: str | Path = PROCESSED_VALIDATION_DIR,
) -> dict[str, pd.DataFrame]:
    """Le as fontes, calcula a validacao e salva os resultados em CSV."""
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
