"""Monta o dataset final para previsão do déficit hídrico de milho."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .apsim.processing import DEFAULT_REPORT, add_apsim_features, filter_to_crop_window, read_apsim_report
    from .data_io import write_model_dataset
    from .gfs_data_extractor.forecast import build_daily_forecast, read_forecast
    from .nasa_power.processing import read_nasa_power_data
    from .paths import MODEL_DATA_DIR, PROCESSED_GFS_DIR, PROCESSED_NASA_POWER_DIR, RAW_GFS_DIR, RAW_NASA_POWER_DIR
except ImportError:  # Permite executar o módulo pelo caminho do arquivo.
    from apsim.processing import DEFAULT_REPORT, add_apsim_features, filter_to_crop_window, read_apsim_report
    from data_io import write_model_dataset
    from gfs_data_extractor.forecast import build_daily_forecast, read_forecast
    from nasa_power.processing import read_nasa_power_data
    from paths import MODEL_DATA_DIR, PROCESSED_GFS_DIR, PROCESSED_NASA_POWER_DIR, RAW_GFS_DIR, RAW_NASA_POWER_DIR


DEFAULT_OUTPUT = MODEL_DATA_DIR / "training_dataset.csv"
DOCUMENTATION = Path(__file__).resolve().parents[1] / "dataset_treinamento.md"
GROUP_COLS = ["SimulationName", "cycle_id"]
TARGET_COLUMN = "deficit_agua_proximo_dia_mm"
VARIATION_TARGET_COLUMN = "variacao_deficit_proximo_dia_mm"

METADATA_COLUMNS = [
    "data",
    "data_alvo",
    "simulation_name",
    "cycle_id",
    "ano_semeadura",
    "local",
    "cenario_irrigado",
]

MAIZE_LOCATION = {
    "Alegrete": "Alegrete",
    "AlegreteClay": "Alegrete",
    "AlegreteClayIrrigation": "Alegrete",
    "AlegreteIrrigation": "Alegrete",
    "Simulation": "NovaRamada",
    "SimulationClay": "NovaRamada",
    "SimulationClayIrrigation": "NovaRamada",
    "SimulationIrrigation": "NovaRamada",
}

FEATURE_COLUMNS = [
    "umidade_solo_mm",
    "umidade_solo_passada_1d_mm",
    "umidade_solo_passada_2d_mm",
    "umidade_solo_passada_3d_mm",
    "precipitacao_observada_mm",
    "irrigacao_aplicada_mm",
    "irrigacao_aplicada_dia_posterior_mm",
    "etreal_mm_dia",
    "previsao_chuva_24h_mm",
    "previsao_temperatura_maxima_C",
    "previsao_temperatura_minima_C",
    "previsao_radiacao_solar_MJ_m2_dia",
    "temperatura_media_C",
    "temperatura_maxima_C",
    "temperatura_minima_C",
    "umidade_relativa_pct",
    "velocidade_vento_m_s",
    "radiacao_solar_MJ_m2_dia",
    "dr_mm",
    "profundidade_radicular_mm",
    "gdd_acumulado_C_dia",
    "etr_acumulado_mm",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
    "escoamento_superficial_mm",
    "drenagem_mm",
    "taw_mm",
    "dias_desde_semeadura",
]
TARGET_COLUMNS = [TARGET_COLUMN, VARIATION_TARGET_COLUMN]
FINAL_MODEL_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS


def _load_nasa(location: str) -> pd.DataFrame:
    processed_paths = {
        "Alegrete": PROCESSED_NASA_POWER_DIR / "ClimaAlegrete2019-2026.processed.csv",
        "NovaRamada": PROCESSED_NASA_POWER_DIR / "ClimaNovaRamada_2019-2026.processed.csv",
    }
    raw_paths = {
        "Alegrete": RAW_NASA_POWER_DIR / "ClimaAlegrete2019-2026.csv",
        "NovaRamada": RAW_NASA_POWER_DIR / "ClimaNovaRamada_2019-2026.csv",
    }
    path = processed_paths[location]
    df = pd.read_csv(path) if path.exists() else read_nasa_power_data(raw_paths[location])
    df["date"] = pd.to_datetime(df["date"])
    return df[
        [
            "date", "PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M",
            "WS2M", "ALLSKY_SFC_SW_DWN",
        ]
    ].copy()


def _find_forecast_rain_file(location: str) -> Path:
    matches = [
        path for path in sorted(RAW_GFS_DIR.glob("Previs*"))
        if location.lower() in path.name.lower()
    ]
    if not matches:
        raise FileNotFoundError(f"Previsão de chuva GFS não encontrada para {location}")
    return matches[0]


def _load_gfs_forecasts(location: str) -> pd.DataFrame:
    """Retorna as previsões disponíveis em D para o dia D+1."""
    rain = build_daily_forecast(
        read_forecast(_find_forecast_rain_file(location)),
        horario="00:00",
    ).rename("previsao_chuva_24h_mm").rename_axis("date").reset_index()
    rain["date"] = pd.to_datetime(rain["date"])

    suffix = "Alegrete" if location == "Alegrete" else "Nova_Ramada"
    temperature = pd.read_csv(PROCESSED_GFS_DIR / f"gfs_temperature_20190613_20260813_{suffix}.csv")
    radiation = pd.read_csv(PROCESSED_GFS_DIR / f"gfs_radiation_20190613_20260813_{suffix}.csv")
    forecast = temperature.merge(radiation, on="datetime", how="inner")
    # O produto diário é válido na data de datetime; a previsão de D+1
    # disponível no registro D é alinhada deslocando a data um dia para trás.
    forecast["date"] = pd.to_datetime(forecast["datetime"]) - pd.Timedelta(days=1)
    forecast = forecast.rename(
        columns={
            "Tmax_24h_C": "previsao_temperatura_maxima_C",
            "Tmin_24h_C": "previsao_temperatura_minima_C",
            "DSWRF_24h_MJ_m2": "previsao_radiacao_solar_MJ_m2_dia",
        }
    )
    forecast = forecast[
        [
            "date",
            "previsao_temperatura_maxima_C",
            "previsao_temperatura_minima_C",
            "previsao_radiacao_solar_MJ_m2_dia",
        ]
    ]
    return rain.merge(forecast, on="date", how="outer")


def validate_dataset_schema(
    df: pd.DataFrame,
    documentation_path: str | Path = DOCUMENTATION,
) -> None:
    """Garante que o dataset e o Markdown têm exatamente o esquema esperado."""
    missing = [column for column in FINAL_MODEL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset sem colunas obrigatórias: {missing}")
    text = Path(documentation_path).read_text(encoding="utf-8").lower()
    undocumented = [column for column in FINAL_MODEL_COLUMNS if column.lower() not in text]
    if undocumented:
        raise ValueError(f"Colunas não documentadas em {documentation_path}: {undocumented}")


def add_next_day_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona os alvos de D+1 sem atravessar cenários, ciclos ou lacunas."""
    result = df.copy()
    grouped = result.groupby(GROUP_COLS, sort=False)
    result["data_alvo"] = grouped["Clock.Today"].shift(-1)
    result[TARGET_COLUMN] = grouped["Dr_root"].shift(-1)
    consecutive = result["data_alvo"] == result["Clock.Today"] + pd.Timedelta(days=1)
    result.loc[~consecutive, ["data_alvo", TARGET_COLUMN]] = pd.NA
    result[VARIATION_TARGET_COLUMN] = result[TARGET_COLUMN] - result["Dr_root"]
    return result


def build_training_dataset(
    report_path: str | Path = DEFAULT_REPORT,
    output_path: str | Path = DEFAULT_OUTPUT,
    documentation_path: str | Path = DOCUMENTATION,
) -> pd.DataFrame:
    """Gera, valida e salva o dataset central de treinamento."""
    apsim = add_apsim_features(filter_to_crop_window(read_apsim_report(report_path)))
    apsim["date"] = pd.to_datetime(apsim["Clock.Today"])
    apsim["local"] = apsim["SimulationName"].map(MAIZE_LOCATION)
    if apsim["local"].isna().any():
        raise ValueError("Existe simulação APSIM sem mapeamento para Alegrete/Nova Ramada")
    apsim = add_next_day_targets(apsim)
    apsim["ano_semeadura"] = apsim["Maize.SowingDate"].dt.year.astype("Int64")
    apsim["cenario_irrigado"] = apsim["SimulationName"].str.contains(
        "Irrigation",
        case=False,
        regex=False,
    )

    climate_frames = []
    forecast_frames = []
    for location in sorted(apsim["local"].unique()):
        climate = _load_nasa(location)
        climate["local"] = location
        climate_frames.append(climate)
        forecast = _load_gfs_forecasts(location)
        forecast["local"] = location
        forecast_frames.append(forecast)

    df = apsim.merge(
        pd.concat(climate_frames, ignore_index=True),
        on=["local", "date"],
        how="left",
        validate="many_to_one",
    )
    df = df.merge(
        pd.concat(forecast_frames, ignore_index=True),
        on=["local", "date"],
        how="left",
        validate="many_to_one",
    )
    df = df.rename(
        columns={
            "SoilWater_root": "umidade_solo_mm",
            "Umidade_solo_passada_1d": "umidade_solo_passada_1d_mm",
            "Umidade_solo_passada_2d": "umidade_solo_passada_2d_mm",
            "Umidade_solo_passada_3d": "umidade_solo_passada_3d_mm",
            "PRECTOTCORR": "precipitacao_observada_mm",
            "Irrigation.IrrigationApplied": "irrigacao_aplicada_mm",
            "Irrigacao_dia_posterior": "irrigacao_aplicada_dia_posterior_mm",
            "ETreal": "etreal_mm_dia",
            "T2M": "temperatura_media_C",
            "T2M_MAX": "temperatura_maxima_C",
            "T2M_MIN": "temperatura_minima_C",
            "RH2M": "umidade_relativa_pct",
            "WS2M": "velocidade_vento_m_s",
            "ALLSKY_SFC_SW_DWN": "radiacao_solar_MJ_m2_dia",
            "Dr_root": "dr_mm",
            "Maize.Root.Depth": "profundidade_radicular_mm",
            "GDD_acumulado": "gdd_acumulado_C_dia",
            "ETr_acumulado": "etr_acumulado_mm",
            "DOY_sin": "doy_sin",
            "DOY_cos": "doy_cos",
            "Month_sin": "month_sin",
            "Month_cos": "month_cos",
            "Soil.SoilWater.Runoff": "escoamento_superficial_mm",
            "Soil.SoilWater.Drainage": "drenagem_mm",
            "TAW_root": "taw_mm",
            "Maize.DaysAfterSowing": "dias_desde_semeadura",
            "date": "data",
            "SimulationName": "simulation_name",
        }
    )
    final = df[FINAL_MODEL_COLUMNS].dropna().reset_index(drop=True)
    validate_dataset_schema(final, documentation_path)
    write_model_dataset(final, output_path)
    return final


def main() -> pd.DataFrame:
    result = build_training_dataset()
    print(f"dataset salvo em: {DEFAULT_OUTPUT}")
    print(f"linhas: {len(result)}")
    print(f"colunas: {len(result.columns)}")
    print(f"target do modelo: {VARIATION_TARGET_COLUMN}")
    return result


if __name__ == "__main__":
    main()
