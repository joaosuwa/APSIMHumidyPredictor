"""Extracao e consolidacao de multiplos produtos do dataset GFS/GDEX."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

try:
    from ..feature_engineering import process_gfs_temperature_data, write_csv
    from ..paths import PROCESSED_GFS_DIR, RAW_GFS_DIR, ensure_data_directories
    from . import gdex_client as rc
except ImportError:  # Permite executar este arquivo diretamente.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from feature_engineering import process_gfs_temperature_data, write_csv
    from paths import PROCESSED_GFS_DIR, RAW_GFS_DIR, ensure_data_directories
    import gdex_client as rc


DATASET = "d084001"
LATITUDE = -28
LONGITUDE = -53.75
START_DATE = "201906130000"
END_DATE = "202608011200"

RAW_OUTPUT_DIR = RAW_GFS_DIR / "downloads"
PROCESSED_OUTPUT_DIR = PROCESSED_GFS_DIR

PRODUCTS = [
    {"name": "Tmax_0_6", "param": "T MAX", "product": "6-hour Maximum (initial+0 to initial+6)"},
    {"name": "Tmax_6_12", "param": "T MAX", "product": "6-hour Maximum (initial+6 to initial+12)"},
    {"name": "Tmax_12_18", "param": "T MAX", "product": "6-hour Maximum (initial+12 to initial+18)"},
    {"name": "Tmax_18_24", "param": "T MAX", "product": "6-hour Maximum (initial+18 to initial+24)"},
    {"name": "Tmin_0_6", "param": "T MIN", "product": "6-hour Minimum (initial+0 to initial+6)"},
    {"name": "Tmin_6_12", "param": "T MIN", "product": "6-hour Minimum (initial+6 to initial+12)"},
    {"name": "Tmin_12_18", "param": "T MIN", "product": "6-hour Minimum (initial+12 to initial+18)"},
    {"name": "Tmin_18_24", "param": "T MIN", "product": "6-hour Minimum (initial+18 to initial+24)"},
]

INTERVAL_END_HOURS = {
    product["name"]: hour
    for product, hour in zip(PRODUCTS, [6, 12, 18, 0, 6, 12, 18, 0])
}
FINAL_COLUMNS = [
    "datetime", "Tmax_0_6", "Tmax_6_12", "Tmax_12_18", "Tmax_18_24",
    "Tmin_0_6", "Tmin_6_12", "Tmin_12_18", "Tmin_18_24",
    "Tmax_24h_C", "Tmin_24h_C", "Tmean_24h_C",
]
FINAL_OUTPUT_FILE = PROCESSED_OUTPUT_DIR / "gfs_temperature_20190613_20260813.csv"
ALEGRETE_FILE_PATTERN = "*29.75S_55.75W.csv"
NOVA_RAMADA_FILE_PATTERN = "*28.0S_53.75W.csv"

ensure_data_directories()


def submit_request(param: str, product: str) -> str:
    """Submete uma consulta pontual ao GDEX."""
    control = {
        "dataset": DATASET,
        "date": f"{START_DATE}/to/{END_DATE}",
        "datetype": "init",
        "param": param,
        "level": "HTGL:2",
        "product": product,
        "oformat": "csv",
        "nlat": LATITUDE,
        "slat": LATITUDE,
        "wlon": LONGITUDE,
        "elon": LONGITUDE,
    }
    print(json.dumps(control, indent=4))
    response = rc.submit_json(control)
    if response.get("http_response") != 200:
        raise RuntimeError("Erro ao submeter request:\n" + json.dumps(response, indent=4))
    request_id = response["data"]["request_id"]
    print(f"Request ID: {request_id}")
    return request_id


def wait_for_request(request_id: str, interval: int = 20) -> dict:
    """Aguarda a conclusao de uma consulta do GDEX."""
    while True:
        status = rc.get_status(request_id)
        request_status = status["data"]["status"]
        print(f"Status: {request_status}")
        if request_status == "Completed":
            return status
        if request_status == "Error":
            raise RuntimeError(f"Request {request_id} terminou com erro.")
        time.sleep(interval)


def download_request(request_id: str, output_dir: str | Path):
    """Baixa os CSVs brutos para a pasta de downloads do GFS."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return rc.download(str(request_id), out_dir=str(output_dir) + "/")


def purge_request(request_id: str):
    """Remove a request finalizada do GDEX."""
    return rc.purge_request(str(request_id))


def process_product(product_info: dict, file_pattern: str = "*.csv") -> pd.DataFrame:
    """Baixa, normaliza e devolve um produto GFS."""
    name = product_info["name"]
    raw_dir = RAW_OUTPUT_DIR / name
    request_id = submit_request(product_info["param"], product_info["product"])
    try:
        wait_for_request(request_id)
        download_request(request_id, raw_dir)
        return process_gfs_temperature_data(raw_dir, name, pattern=file_pattern)
    finally:
        purge_request(request_id)


def load_or_process_product(
    product_info: dict,
    fetch_missing: bool = False,
    file_pattern: str = "*.csv",
) -> pd.DataFrame:
    """Reutiliza downloads locais ou busca o produto quando solicitado."""
    name = product_info["name"]
    raw_dir = RAW_OUTPUT_DIR / name
    if raw_dir.exists() and list(raw_dir.rglob(file_pattern)):
        return process_gfs_temperature_data(raw_dir, name, pattern=file_pattern)
    if not fetch_missing:
        raise FileNotFoundError(
            f"Dados brutos não encontrados para {name}. "
            "Use fetch_missing=True para baixar do GFS."
        )
    return process_product(product_info, file_pattern=file_pattern)


def build_daily_temperature_dataframe(
    all_results: dict[str, pd.DataFrame],
    products: list[dict] = PRODUCTS,
) -> pd.DataFrame:
    """Consolida os produtos de seis horas em uma serie diaria."""
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
    fetch_missing: bool = False,
    output_file: str | Path = FINAL_OUTPUT_FILE,
    file_pattern: str = ALEGRETE_FILE_PATTERN,
    products: list[dict] = PRODUCTS,
) -> pd.DataFrame:
    """Gera um CSV diario para a lista de produtos selecionada."""
    all_results = {
        product["name"]: load_or_process_product(product, fetch_missing, file_pattern)
        for product in products
    }
    final_df = build_daily_temperature_dataframe(all_results, products=products)
    write_csv(final_df, output_file)
    return final_df


def main(fetch_missing: bool = False) -> pd.DataFrame:
    return generate_gfs_temperature_file(fetch_missing=fetch_missing)


def main_nova_ramada(fetch_missing: bool = False) -> pd.DataFrame:
    output_file = PROCESSED_OUTPUT_DIR / "gfs_temperature_20190613_20260813_Nova_Ramada.csv"
    return generate_gfs_temperature_file(
        fetch_missing=fetch_missing,
        output_file=output_file,
        file_pattern=NOVA_RAMADA_FILE_PATTERN,
    )


if __name__ == "__main__":
    main()
