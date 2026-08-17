"""Leitura e agregação diária de previsões do GFS."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from ..data_io import read_csv_files
except ImportError:  # Permite executar o módulo diretamente.
    from data_io import read_csv_files


def read_forecast(path: str | Path) -> pd.DataFrame:
    """Lê um CSV de previsão e devolve timestamp, data, hora e precipitação."""
    forecast = read_csv_files(path)
    forecast.columns = ["Date", "Time", "precip"]
    hour = pd.to_datetime(forecast["Time"], format="%H:%M").dt.hour
    forecast["ts"] = pd.to_datetime(forecast["Date"]) + pd.to_timedelta(hour, unit="h")
    forecast = forecast.sort_values("ts").reset_index(drop=True)
    forecast["date"] = forecast["ts"].dt.date
    forecast["hour"] = hour.astype(int)
    return forecast[["ts", "date", "hour", "precip"]]


def build_daily_forecast(forecast: pd.DataFrame, horario: str = "00:00") -> pd.Series:
    """Relaciona a previsão de D ao registro de D+1 no horário escolhido."""
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
    """Mantém apenas previsões a partir da data de referência."""
    if reference_date is None:
        return daily_forecast.copy()
    start = pd.Timestamp(reference_date).normalize()
    mask = daily_forecast.index >= start
    if horizon_days is not None:
        end = start + pd.Timedelta(days=horizon_days - 1)
        mask &= daily_forecast.index <= end
    return daily_forecast.loc[mask]
