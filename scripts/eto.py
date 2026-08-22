"""Cálculo de ETo diária pelo método FAO-56 Penman--Monteith."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    import pyeto
except ImportError as exc:  # pragma: no cover - depende do ambiente externo.
    raise ImportError(
        "PyETo não está instalado. Execute: "
        "python -m pip install -r requirements-eto.txt"
    ) from exc


FORECAST_WIND_2M_COLUMN = "velocidade_vento_prevista_2m_m_s"


def calculate_forecast_wind_speed_2m(forecast: pd.DataFrame) -> pd.Series:
    """Combina U/V a 10 m e converte a velocidade para 2 m pela FAO-56."""
    required = {"u_grd_10m_m_s", "v_grd_10m_m_s"}
    missing = sorted(required - set(forecast.columns))
    if missing:
        raise ValueError(f"Dados insuficientes para calcular vento a 2 m: {missing}")

    u_component = pd.to_numeric(forecast["u_grd_10m_m_s"], errors="coerce")
    v_component = pd.to_numeric(forecast["v_grd_10m_m_s"], errors="coerce")
    wind_10m = np.hypot(u_component, v_component)
    # Equação 47 da FAO-56, também usada por ``pyeto.wind_speed_2m``.
    wind_2m = wind_10m * 4.87 / np.log(67.8 * 10.0 - 5.42)
    result = pd.Series(wind_2m, index=forecast.index, dtype=float)
    result.loc[~np.isfinite(result)] = np.nan
    result.name = FORECAST_WIND_2M_COLUMN
    return result


def _calculate_row_eto(
    row: pd.Series,
    *,
    latitude_deg: float,
    elevation_m: float,
) -> float:
    """Calcula a ETo de uma linha já alinhada para a data de previsão."""
    required = [
        "date",
        "previsao_temperatura_maxima_C",
        "previsao_temperatura_minima_C",
        "previsao_radiacao_solar_MJ_m2_dia",
        "u_grd_10m_m_s",
        "v_grd_10m_m_s",
    ]
    if any(pd.isna(row.get(column)) for column in required):
        return np.nan

    tmax = float(row["previsao_temperatura_maxima_C"])
    tmin = float(row["previsao_temperatura_minima_C"])
    solar_rad = float(row["previsao_radiacao_solar_MJ_m2_dia"])
    u10 = math.hypot(
        float(row["u_grd_10m_m_s"]),
        float(row["v_grd_10m_m_s"]),
    )
    if not np.isfinite([tmax, tmin, solar_rad, u10]).all():
        return np.nan
    if tmax < tmin or solar_rad < 0:
        return np.nan

    rh = row.get("umidade_relativa_prevista_pct")
    if not pd.isna(rh):
        rh = float(rh)
        if not 0 <= rh <= 100:
            return np.nan

    tdew = row.get("ponto_orvalho_previsto_C")
    if pd.isna(tdew):
        if pd.isna(rh):
            return np.nan
        actual_vapour_pressure = pyeto.avp_from_rhmean(
            pyeto.svp_from_t(tmin),
            pyeto.svp_from_t(tmax),
            rh,
        )
    else:
        actual_vapour_pressure = pyeto.avp_from_tdew(float(tdew))

    tmean = pyeto.daily_mean_t(tmin, tmax)
    saturation_vapour_pressure = pyeto.mean_svp(tmin, tmax)
    slope_svp = pyeto.delta_svp(tmean)
    atmospheric_pressure = pyeto.atm_pressure(float(elevation_m))
    psychrometric_constant = pyeto.psy_const(atmospheric_pressure)
    wind_2m = pyeto.wind_speed_2m(u10, 10.0)

    date = pd.Timestamp(row["date"])
    latitude_rad = math.radians(float(latitude_deg))
    day_of_year = int(date.dayofyear)
    solar_declination = pyeto.sol_dec(day_of_year)
    sunset_hour_angle = pyeto.sunset_hour_angle(latitude_rad, solar_declination)
    inverse_distance = pyeto.inv_rel_dist_earth_sun(day_of_year)
    extraterrestrial_radiation = pyeto.et_rad(
        latitude_rad,
        solar_declination,
        sunset_hour_angle,
        inverse_distance,
    )
    clear_sky_radiation = pyeto.cs_rad(elevation_m, extraterrestrial_radiation)
    net_shortwave = pyeto.net_in_sol_rad(solar_rad, albedo=0.23)
    net_longwave = pyeto.net_out_lw_rad(
        tmin + 273.15,
        tmax + 273.15,
        solar_rad,
        clear_sky_radiation,
        actual_vapour_pressure,
    )
    net_radiation = pyeto.net_rad(net_shortwave, net_longwave)
    eto = pyeto.fao56_penman_monteith(
        net_radiation,
        tmean + 273.15,
        wind_2m,
        saturation_vapour_pressure,
        actual_vapour_pressure,
        slope_svp,
        psychrometric_constant,
        shf=0.0,
    )
    return max(0.0, float(eto)) if np.isfinite(eto) else np.nan


def calculate_fao56_eto(
    forecast: pd.DataFrame,
    *,
    latitude_deg: float,
    elevation_m: float,
) -> pd.Series:
    """Calcula ETo [mm/dia] para previsões GFS alinhadas em ``date``."""
    required = {
        "date",
        "previsao_temperatura_maxima_C",
        "previsao_temperatura_minima_C",
        "previsao_radiacao_solar_MJ_m2_dia",
        "u_grd_10m_m_s",
        "v_grd_10m_m_s",
    }
    missing = sorted(required - set(forecast.columns))
    if missing:
        raise ValueError(f"Dados insuficientes para calcular ETo: {missing}")

    frame = forecast.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    result = frame.apply(
        _calculate_row_eto,
        axis=1,
        latitude_deg=latitude_deg,
        elevation_m=elevation_m,
    )
    result.name = "previsao_eto_mm_dia"
    return result


def add_fao56_eto(
    forecast: pd.DataFrame,
    *,
    latitude_deg: float,
    elevation_m: float,
) -> pd.DataFrame:
    """Adiciona vento a 2 m e ETo prevista sem remover colunas do forecast."""
    result = forecast.copy()
    result[FORECAST_WIND_2M_COLUMN] = calculate_forecast_wind_speed_2m(result)
    result["previsao_eto_mm_dia"] = calculate_fao56_eto(
        result,
        latitude_deg=latitude_deg,
        elevation_m=elevation_m,
    )
    return result
