"""Processamento local dos arquivos brutos baixados do GFS/GDEX."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from ..data_io import read_csv_files, write_csv
    from ..paths import ensure_data_directories
    from .gfs_config import (
        ALEGRETE_COORDINATES,
        FINAL_COLUMNS,
        FINAL_OUTPUT_FILE,
        INTERVAL_END_HOURS,
        LATITUDE,
        LONGITUDE,
        NOVA_RAMADA_COORDINATES,
        PRODUCTS,
        PROCESSED_OUTPUT_DIR,
        RADIATION_FINAL_COLUMNS,
        RADIATION_FINAL_OUTPUT_FILE,
        RADIATION_PRODUCTS,
        RAW_OUTPUT_DIR,
        coordinate_file_pattern,
    )
except ImportError:  # Permite executar este arquivo diretamente.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_io import read_csv_files, write_csv
    from paths import ensure_data_directories
    from gfs_config import (
        ALEGRETE_COORDINATES,
        FINAL_COLUMNS,
        FINAL_OUTPUT_FILE,
        INTERVAL_END_HOURS,
        LATITUDE,
        LONGITUDE,
        NOVA_RAMADA_COORDINATES,
        PRODUCTS,
        PROCESSED_OUTPUT_DIR,
        RADIATION_FINAL_COLUMNS,
        RADIATION_FINAL_OUTPUT_FILE,
        RADIATION_PRODUCTS,
        RAW_OUTPUT_DIR,
        coordinate_file_pattern,
    )

ensure_data_directories()


def read_gfs_data(
    source: str | Path | list[str | Path],
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """Lê e concatena CSVs brutos exportados pelo GFS/GDEX."""
    return read_csv_files(source, pattern=pattern)


def find_temperature_column(df: pd.DataFrame) -> str:
    """Identifica a coluna de temperatura de um resultado GFS."""
    candidates = [
        column
        for column in df.columns
        if any(term in column.lower() for term in ("temperature", "t max", "t min"))
    ]
    if not candidates:
        raise ValueError(
            "Coluna de temperatura não encontrada. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )
    return candidates[-1]


def process_gfs_temperature_data(
    source: str | Path,
    variable_name: str,
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """Converte temperatura GFS de Kelvin para Celsius e cria ``datetime``."""
    df = read_gfs_data(source, pattern=pattern)
    temperature_column = find_temperature_column(df)
    df[temperature_column] = pd.to_numeric(df[temperature_column], errors="coerce")
    df["temperature_C"] = df[temperature_column] - 273.15
    if {"Date", "Time"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
        )
    df["variable"] = variable_name
    return df


def find_radiation_column(df: pd.DataFrame) -> str:
    """Identifica a coluna DSWRF do resultado GFS."""
    candidates = [
        column
        for column in df.columns
        if any(term in column.lower() for term in ("dswrf", "shortwave radiation", "radiation flux"))
    ]
    if not candidates:
        raise ValueError(
            "Coluna DSWRF não encontrada. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )
    return candidates[-1]


def process_gfs_radiation_data(
    source: str | Path,
    variable_name: str,
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """Lê DSWRF bruto em W/m² e cria ``datetime`` sem alterar a unidade."""
    df = read_gfs_data(source, pattern=pattern)
    radiation_column = find_radiation_column(df)
    df["dswrf_W_m2"] = pd.to_numeric(df[radiation_column], errors="coerce")
    if {"Date", "Time"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
        )
    df["variable"] = variable_name
    return df


def process_radiation_product(
    product_info: dict,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    file_pattern: str | None = None,
) -> pd.DataFrame:
    """Processa um produto DSWRF já baixado, sem fazer requisições externas."""
    file_pattern = file_pattern or coordinate_file_pattern(latitude, longitude)
    raw_dir = RAW_OUTPUT_DIR / product_info["name"]
    return process_gfs_radiation_data(
        raw_dir,
        variable_name=product_info["name"],
        pattern=file_pattern,
    )


def load_or_process_radiation_product(
    product_info: dict,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    file_pattern: str | None = None,
) -> pd.DataFrame:
    """Carrega um produto DSWRF existente em ``data/raw/gfs/downloads``."""
    file_pattern = file_pattern or coordinate_file_pattern(latitude, longitude)
    raw_dir = RAW_OUTPUT_DIR / product_info["name"]
    if not raw_dir.exists() or not list(raw_dir.rglob(file_pattern)):
        raise FileNotFoundError(
            f"Dados brutos de radiação não encontrados para {product_info['name']} em {raw_dir}. "
            "Execute primeiro get_gfs_data.py."
        )
    return process_radiation_product(
        product_info,
        latitude=latitude,
        longitude=longitude,
        file_pattern=file_pattern,
    )


def build_daily_radiation_dataframe(
    all_results: dict[str, pd.DataFrame],
    products: list[dict] = RADIATION_PRODUCTS,
) -> pd.DataFrame:
    """Consolida médias de fluxo DSWRF de seis horas em energia diária."""
    daily_df = None
    radiation_columns = []

    for product_info in products:
        name = product_info["name"]
        output_name = f"{name}_W_m2"
        radiation_columns.append(output_name)
        radiation = all_results[name][["datetime", "dswrf_W_m2"]].copy()
        radiation["datetime"] = pd.to_datetime(radiation["datetime"], errors="coerce")
        radiation["dswrf_W_m2"] = pd.to_numeric(radiation["dswrf_W_m2"], errors="coerce")
        radiation = radiation.dropna(subset=["datetime"])
        radiation = radiation[radiation["datetime"].dt.hour == INTERVAL_END_HOURS[name]]
        radiation = radiation.sort_values("datetime").drop_duplicates("datetime")
        radiation["date"] = radiation["datetime"].dt.normalize()

        if name.endswith("18_24"):
            radiation["date"] -= pd.Timedelta(days=1)

        radiation = radiation.sort_values("date").drop_duplicates("date")
        radiation = radiation[["date", "dswrf_W_m2"]].rename(
            columns={"dswrf_W_m2": output_name}
        )
        daily_df = radiation if daily_df is None else daily_df.merge(
            radiation,
            on="date",
            how="outer",
        )

    if daily_df is None:
        raise RuntimeError("Nenhum dado de radiação foi consolidado.")

    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    # Não exporta dias parciais: a radiação diária precisa dos quatro
    # intervalos de seis horas.
    daily_df = daily_df.dropna(subset=radiation_columns).reset_index(drop=True)
    daily_df["DSWRF_24h_mean_W_m2"] = daily_df[radiation_columns].mean(axis=1)

    # Cada coluna é uma média de fluxo ao longo de seis horas.
    # W/m² × 6 h × 3.600 s/h ÷ 1.000.000 = MJ/m².
    daily_df["DSWRF_24h_MJ_m2"] = daily_df[radiation_columns].sum(axis=1) * 0.0216

    daily_df = daily_df.rename(columns={"date": "datetime"})
    daily_df["datetime"] = daily_df["datetime"].dt.strftime("%Y-%m-%d")
    return daily_df.reindex(columns=RADIATION_FINAL_COLUMNS)


def generate_gfs_radiation_file(
    output_file: str | Path = RADIATION_FINAL_OUTPUT_FILE,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    file_pattern: str | None = None,
    products: list[dict] = RADIATION_PRODUCTS,
) -> pd.DataFrame:
    """Processa DSWRF bruto local e grava a série diária consolidada."""
    all_results = {
        product["name"]: load_or_process_radiation_product(
            product,
            latitude=latitude,
            longitude=longitude,
            file_pattern=file_pattern,
        )
        for product in products
    }
    final_df = build_daily_radiation_dataframe(all_results, products=products)
    write_csv(final_df, output_file)
    return final_df


def process_product(
    product_info: dict,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    file_pattern: str | None = None,
) -> pd.DataFrame:
    """Processa um produto GFS já baixado, sem fazer requisições externas."""
    file_pattern = file_pattern or coordinate_file_pattern(latitude, longitude)
    raw_dir = RAW_OUTPUT_DIR / product_info["name"]
    return process_gfs_temperature_data(
        raw_dir,
        variable_name=product_info["name"],
        pattern=file_pattern,
    )


def load_or_process_product(
    product_info: dict,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    file_pattern: str | None = None,
) -> pd.DataFrame:
    """Carrega e processa um produto existente em ``data/raw/gfs/downloads``."""
    file_pattern = file_pattern or coordinate_file_pattern(latitude, longitude)
    raw_dir = RAW_OUTPUT_DIR / product_info["name"]
    if not raw_dir.exists() or not list(raw_dir.rglob(file_pattern)):
        raise FileNotFoundError(
            f"Dados brutos não encontrados para {product_info['name']} em {raw_dir}. "
            "Execute primeiro get_gfs_data.py."
        )
    return process_product(
        product_info,
        latitude=latitude,
        longitude=longitude,
        file_pattern=file_pattern,
    )


def build_daily_temperature_dataframe(
    all_results: dict[str, pd.DataFrame],
    products: list[dict] = PRODUCTS,
) -> pd.DataFrame:
    """Consolida os produtos de seis horas em uma série diária."""
    daily_df = None
    for product_info in products:
        name = product_info["name"]
        temperature = all_results[name][["datetime", "temperature_C"]].copy()
        temperature["datetime"] = pd.to_datetime(temperature["datetime"], errors="coerce")
        temperature["temperature_C"] = pd.to_numeric(temperature["temperature_C"], errors="coerce")
        temperature = temperature.dropna(subset=["datetime"])
        temperature = temperature[temperature["datetime"].dt.hour == INTERVAL_END_HOURS[name]]
        temperature = temperature.sort_values("datetime").drop_duplicates("datetime")
        temperature["date"] = temperature["datetime"].dt.normalize()
        if name.endswith("18_24"):
            temperature["date"] -= pd.Timedelta(days=1)
        temperature = temperature.sort_values("date").drop_duplicates("date")
        temperature = temperature[["date", "temperature_C"]].rename(columns={"temperature_C": name})
        daily_df = temperature if daily_df is None else daily_df.merge(temperature, on="date", how="outer")

    if daily_df is None:
        raise RuntimeError("Nenhum dado de temperatura foi consolidado.")

    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    max_columns = ["Tmax_0_6", "Tmax_6_12", "Tmax_12_18", "Tmax_18_24"]
    min_columns = ["Tmin_0_6", "Tmin_6_12", "Tmin_12_18", "Tmin_18_24"]
    daily_df["Tmax_24h_C"] = daily_df[max_columns].max(axis=1)
    daily_df["Tmin_24h_C"] = daily_df[min_columns].min(axis=1)
    daily_df["Tmean_24h_C"] = (daily_df["Tmax_24h_C"] - daily_df["Tmin_24h_C"]) / 2.0
    daily_df = daily_df.rename(columns={"date": "datetime"})
    daily_df["datetime"] = daily_df["datetime"].dt.strftime("%Y-%m-%d")
    return daily_df.reindex(columns=FINAL_COLUMNS)


def generate_gfs_temperature_file(
    output_file: str | Path = FINAL_OUTPUT_FILE,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    file_pattern: str | None = None,
    products: list[dict] = PRODUCTS,
) -> pd.DataFrame:
    """Processa downloads locais e grava a série diária consolidada."""
    all_results = {
        product["name"]: load_or_process_product(
            product,
            latitude=latitude,
            longitude=longitude,
            file_pattern=file_pattern,
        )
        for product in products
    }
    final_df = build_daily_temperature_dataframe(all_results, products=products)
    write_csv(final_df, output_file)
    return final_df


def main() -> pd.DataFrame:
    return generate_gfs_temperature_file(
        latitude=ALEGRETE_COORDINATES[0],
        longitude=ALEGRETE_COORDINATES[1],
    )


def main_nova_ramada() -> pd.DataFrame:
    output_file = PROCESSED_OUTPUT_DIR / "gfs_temperature_20190613_20260813_Nova_Ramada.csv"
    return generate_gfs_temperature_file(
        output_file=output_file,
        latitude=NOVA_RAMADA_COORDINATES[0],
        longitude=NOVA_RAMADA_COORDINATES[1],
    )


def main_radiation() -> pd.DataFrame:
    """Gera a radiação diária de Alegrete a partir dos RAW DSWRF."""
    output_file = PROCESSED_OUTPUT_DIR / "gfs_radiation_20190613_20260813_Alegrete.csv"
    return generate_gfs_radiation_file(
        output_file=output_file,
        latitude=ALEGRETE_COORDINATES[0],
        longitude=ALEGRETE_COORDINATES[1],
    )


def main_radiation_nova_ramada() -> pd.DataFrame:
    """Gera a radiação diária de Nova Ramada a partir dos RAW DSWRF."""
    output_file = PROCESSED_OUTPUT_DIR / "gfs_radiation_20190613_20260813_Nova_Ramada.csv"
    return generate_gfs_radiation_file(
        output_file=output_file,
        latitude=NOVA_RAMADA_COORDINATES[0],
        longitude=NOVA_RAMADA_COORDINATES[1],
    )


if __name__ == "__main__":
    main()
