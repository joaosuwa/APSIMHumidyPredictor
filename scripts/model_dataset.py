"""Monta o dataset final para previsão do déficit hídrico de milho."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:  # Compatibilidade com ``python scripts/model_dataset.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.apsim.processing import DEFAULT_REPORT, add_apsim_features, filter_to_crop_window, read_apsim_report
    from scripts.data_io import write_csv
    from scripts.eto import add_fao56_eto
    from scripts.gfs_data_extractor.gfs_config import get_site, processed_forecast_path
    from scripts.gfs_data_extractor.gfs_data_processing import generate_site_forecast
    from scripts.nasa_power.processing import read_nasa_power_data
    from scripts.paths import MODEL_DATA_DIR, PROCESSED_NASA_POWER_DIR, RAW_NASA_POWER_DIR
else:
    from .apsim.processing import DEFAULT_REPORT, add_apsim_features, filter_to_crop_window, read_apsim_report
    from .data_io import write_csv
    from .eto import add_fao56_eto
    from .gfs_data_extractor.gfs_config import get_site, processed_forecast_path
    from .gfs_data_extractor.gfs_data_processing import generate_site_forecast
    from .nasa_power.processing import read_nasa_power_data
    from .paths import MODEL_DATA_DIR, PROCESSED_NASA_POWER_DIR, RAW_NASA_POWER_DIR


DEFAULT_OUTPUT = MODEL_DATA_DIR / "training_dataset.csv"
DOCUMENTATION = Path(__file__).resolve().parents[1] / "dataset_treinamento.md"
GROUP_COLS = ["SimulationName", "cycle_id"]
TARGET_COLUMN = "deficit_agua_proximo_dia_mm"
METADATA_COLUMNS = ["SimulationName", "Clock_today", "cycle_id"]

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
    "previsao_eto_mm_dia",
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
FINAL_MODEL_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]


def _load_nasa(location: str) -> pd.DataFrame:
    site = get_site(location)
    raw_path = RAW_NASA_POWER_DIR / site.nasa_raw_filename
    processed_path = PROCESSED_NASA_POWER_DIR / f"{raw_path.stem}.processed.csv"
    df = pd.read_csv(processed_path) if processed_path.exists() else read_nasa_power_data(raw_path)
    df["date"] = pd.to_datetime(df["date"])
    return df[
        [
            "date", "PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M",
            "WS2M", "ALLSKY_SFC_SW_DWN",
        ]
    ].copy()


def _load_gfs_forecasts(location: str) -> pd.DataFrame:
    """Retorna as previsões disponíveis em D para o dia D+1."""
    site = get_site(location)
    path = processed_forecast_path(site)
    if not path.exists():
        generate_site_forecast(site)
    forecast = pd.read_csv(path)
    forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
    if forecast["date"].isna().any() or forecast["date"].duplicated().any():
        raise ValueError(f"Datas inválidas ou duplicadas no forecast GFS: {path}")
    return add_fao56_eto(
        forecast,
        latitude_deg=site.latitude,
        elevation_m=site.elevation_m,
    )


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


def build_training_dataset(
    report_path: str | Path = DEFAULT_REPORT,
    output_path: str | Path = DEFAULT_OUTPUT,
    documentation_path: str | Path = DOCUMENTATION,
) -> pd.DataFrame:
    """Gera, valida e salva o dataset central de treinamento."""
    apsim = add_apsim_features(filter_to_crop_window(read_apsim_report(report_path)))
    apsim["Clock_today"] = pd.to_datetime(apsim["Clock.Today"]).dt.strftime("%Y-%m-%d")
    apsim["date"] = pd.to_datetime(apsim["Clock_today"])
    apsim["local"] = apsim["SimulationName"].map(MAIZE_LOCATION)
    if apsim["local"].isna().any():
        raise ValueError("Existe simulação APSIM sem mapeamento para Alegrete/Nova Ramada")
    apsim[TARGET_COLUMN] = apsim.groupby(GROUP_COLS, sort=False)["Dr_root"].shift(-1)

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
        }
    )
    final = df[FINAL_MODEL_COLUMNS].dropna().reset_index(drop=True)
    validate_dataset_schema(final, documentation_path)
    write_csv(final, output_path)
    return final


def main() -> pd.DataFrame:
    result = build_training_dataset()
    print(f"dataset salvo em: {DEFAULT_OUTPUT}")
    print(f"linhas: {len(result)}")
    print(f"colunas: {len(result.columns)}")
    print(f"target: {TARGET_COLUMN}")
    return result


if __name__ == "__main__":
    main()
