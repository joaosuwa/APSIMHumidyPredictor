"""Configuração compartilhada entre o downloader e o processador GFS."""

from pathlib import Path

try:
    from ..paths import PROCESSED_GFS_DIR, RAW_GFS_DIR
except ImportError:  # Permite executar módulos diretamente.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import PROCESSED_GFS_DIR, RAW_GFS_DIR


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
