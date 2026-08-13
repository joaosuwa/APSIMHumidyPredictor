from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FORECAST_PAIRS = [
    (
        DATA_DIR / "Chuva24h_2019_2026.csv",
        DATA_DIR / "AlegreteDados.csv",
        "Alegrete",
    ),
    (
        DATA_DIR / "PrevisãoChuva24H_NovaRamada_2019-2026.csv",
        DATA_DIR / "ClimaNovaRamada_2019-2026.csv",
        "NovaRamada",
    ),
]

DEFAULT_HORARIOS = ["00:00", "06:00", "12:00", "18:00"]


def read_forecast(path: str | Path) -> pd.DataFrame:
    """Le um arquivo de previsao (Date, Time, precip) e devolve ts + valor."""
    fc = pd.read_csv(path)
    fc.columns = ["Date", "Time", "precip"]
    hour = pd.to_datetime(fc["Time"], format="%H:%M").dt.hour
    fc["ts"] = pd.to_datetime(fc["Date"]) + pd.to_timedelta(hour, unit="h")
    fc = fc.sort_values("ts").reset_index(drop=True)
    fc["date"] = fc["ts"].dt.date
    fc["hour"] = hour.astype(int)
    return fc[["ts", "date", "hour", "precip"]]


def read_observed(path: str | Path) -> pd.Series:
    """Le PRECTOTCORR do NASA-POWER; -999 vira NaN. Index = data."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("YEAR,"))
    obs = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    obs["date"] = pd.to_datetime(
        obs["YEAR"].astype(int).astype(str) + "-" + obs["DOY"].astype(int).astype(str),
        format="%Y-%j",
    ).dt.date
    series = obs.set_index(pd.to_datetime(obs["date"]))["PRECTOTCORR"]
    series = series.replace(-999.0, np.nan)
    return series


def build_daily_forecast(
    forecast: pd.DataFrame, horario: str = "00:00"
) -> pd.Series:
    """Serie diaria de previsao: previsao(D) = valor(D+1, horario).

    Dias sem registro (linhas ausentes) ficam como NaN.
    """
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
    """Mantem apenas previsoes validas a partir da data de referencia."""
    if reference_date is None:
        return daily_forecast.copy()
    start = pd.Timestamp(reference_date).normalize()
    mask = daily_forecast.index >= start
    if horizon_days is not None:
        end = start + pd.Timedelta(days=horizon_days - 1)
        mask &= daily_forecast.index <= end
    return daily_forecast.loc[mask]


def _detection_metrics(forecast: pd.Series, observed: pd.Series, threshold: float):
    hits = int(((forecast >= threshold) & (observed >= threshold)).sum())
    misses = int(((forecast < threshold) & (observed >= threshold)).sum())
    false_alarms = int(((forecast >= threshold) & (observed < threshold)).sum())
    correct_neg = int(((forecast < threshold) & (observed < threshold)).sum())
    pod = hits / (hits + misses) if (hits + misses) else np.nan
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) else np.nan
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) else np.nan
    acc = (hits + correct_neg) / (hits + misses + false_alarms + correct_neg) if (
        hits + misses + false_alarms + correct_neg
    ) else np.nan
    return {
        "limiar": threshold,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_neg": correct_neg,
        "pod": pod,
        "far": far,
        "csi": csi,
        "acuracia": acc,
    }


def _error_metrics(forecast: pd.Series, observed: pd.Series) -> dict:
    resid = forecast - observed
    n = int(observed.size)
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "vies": np.nan, "correlacao": np.nan}
    corr = float(np.corrcoef(forecast, observed)[0, 1]) if n > 2 else np.nan
    return {
        "n": n,
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "vies": float(np.mean(resid)),
        "correlacao": corr,
    }


def compare_forecast(
    forecast: pd.Series,
    observed: pd.Series,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """Metricas previsao x observado (geral + por mes).

    Ambas as Series devem ter index de datas alinhadas; NaN e descartado.
    """
    merged = pd.DataFrame(
        {"previsao": forecast, "observado": observed}
    ).dropna().sort_index()
    if merged.empty:
        return pd.DataFrame()
    rows = []
    rows.append(
        {
            "grupo": "geral",
            "mes": np.nan,
            **_error_metrics(merged["previsao"], merged["observado"]),
            **_detection_metrics(merged["previsao"], merged["observado"], threshold),
        }
    )
    merged["mes"] = merged.index.to_series().apply(lambda d: d.month)
    for mes, g in merged.groupby("mes"):
        rows.append(
            {
                "grupo": "por_mes",
                "mes": int(mes),
                **_error_metrics(g["previsao"], g["observado"]),
                **_detection_metrics(g["previsao"], g["observado"], threshold),
            }
        )
    return pd.DataFrame(rows)


def compare_all(
    horario: str = "00:00",
    threshold: float = 1.0,
    reference_date: str | pd.Timestamp | None = None,
    horizon_days: int | None = None,
    output_dir: str | Path = DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Orquestra leitura, previsao diaria, filtro futuro e comparacao.

    Salva serie comparada (long format) e metricas em CSV. Devolve
    {"serie": df, "metricas": df}.
    """
    output_dir = Path(output_dir)
    series_frames = []
    metrics_frames = []
    future_info = {}
    for forecast_path, observed_path, local in FORECAST_PAIRS:
        forecast = read_forecast(forecast_path)
        observed = read_observed(observed_path)

        daily = build_daily_forecast(forecast, horario=horario)
        future = filter_future(daily, reference_date, horizon_days)
        future_info[local] = len(future)

        metrics = compare_forecast(daily, observed, threshold=threshold)
        if not metrics.empty:
            metrics.insert(0, "local", local)
            metrics_frames.append(metrics)

        serie = pd.DataFrame(
            {"previsao": daily, "observado": observed}
        ).dropna().sort_index().reset_index()
        serie.insert(0, "local", local)
        serie.columns = ["local", "data", "previsao", "observado"]
        series_frames.append(serie)

    serie_all = pd.concat(series_frames, ignore_index=True)
    metrics_all = pd.concat(metrics_frames, ignore_index=True)
    metrics_all.insert(0, "horario", horario)

    serie_path = output_dir / "serie_compare_previsao_chuva.csv"
    metrics_path = output_dir / "metricas_previsao_chuva.csv"
    serie_all.to_csv(serie_path, index=False)
    metrics_all.to_csv(metrics_path, index=False)

    print(f"horario={horario} limiar={threshold}mm")
    for local, n_future in future_info.items():
        print(f"  {local}: dias futuros a partir de {reference_date}: {n_future}")
    print(f"serie salva em: {serie_path}")
    print(f"metricas salvas em: {metrics_path}")
    return {"serie": serie_all, "metricas": metrics_all}


if __name__ == "__main__":
    result = compare_all(
        horario="00:00",
        threshold=1.0,
        reference_date="2026-08-01",
        horizon_days=7,
    )
    cols = [
        "local", "grupo", "mes", "n", "mae", "rmse", "vies",
        "correlacao", "pod", "far", "csi", "acuracia",
    ]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(result["metricas"].loc[result["metricas"]["grupo"] == "geral", cols].to_string(index=False))
