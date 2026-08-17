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
ALEGRETE_COORDINATES = (-29.75, -55.75)
NOVA_RAMADA_COORDINATES = (-28.0, -53.75)
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

RADIATION_PRODUCTS = [
    {"name": "DSWRF_0_6", "param": "DSWRF", "product": "6-hour Average (initial+0 to initial+6)"},
    {"name": "DSWRF_6_12", "param": "DSWRF", "product": "6-hour Average (initial+6 to initial+12)"},
    {"name": "DSWRF_12_18", "param": "DSWRF", "product": "6-hour Average (initial+12 to initial+18)"},
    {"name": "DSWRF_18_24", "param": "DSWRF", "product": "6-hour Average (initial+18 to initial+24)"},
]

ALL_PRODUCTS = PRODUCTS + RADIATION_PRODUCTS

INTERVAL_END_HOURS = {
    product["name"]: hour
    for product, hour in zip(PRODUCTS, [6, 12, 18, 0, 6, 12, 18, 0])
}
INTERVAL_END_HOURS.update(
    dict(zip(
        [product["name"] for product in RADIATION_PRODUCTS],
        [6, 12, 18, 0],
    ))
)
FINAL_COLUMNS = [
    "datetime", "Tmax_0_6", "Tmax_6_12", "Tmax_12_18", "Tmax_18_24",
    "Tmin_0_6", "Tmin_6_12", "Tmin_12_18", "Tmin_18_24",
    "Tmax_24h_C", "Tmin_24h_C", "Tmean_24h_C",
]
FINAL_OUTPUT_FILE = PROCESSED_OUTPUT_DIR / "gfs_temperature_20190613_20260813.csv"


def format_coordinate(value: float, positive_suffix: str, negative_suffix: str) -> str:
    """Formata uma coordenada no padrão usado nos nomes do GDEX."""
    numeric_value = float(value)
    if numeric_value.is_integer():
        number = f"{abs(numeric_value):.1f}"
    else:
        number = f"{abs(numeric_value):.6f}".rstrip("0").rstrip(".")
    suffix = positive_suffix if numeric_value >= 0 else negative_suffix
    return f"{number}{suffix}"


def coordinate_file_pattern(latitude: float, longitude: float) -> str:
    """Retorna o glob para arquivos GFS de uma coordenada específica."""
    latitude_name = format_coordinate(latitude, "N", "S")
    longitude_name = format_coordinate(longitude, "E", "W")
    return f"*_{latitude_name}_{longitude_name}.csv"


# Aliases mantidos para compatibilidade com chamadas antigas.
ALEGRETE_FILE_PATTERN = coordinate_file_pattern(*ALEGRETE_COORDINATES)
NOVA_RAMADA_FILE_PATTERN = coordinate_file_pattern(*NOVA_RAMADA_COORDINATES)
RADIATION_ALEGRETE_FILE_PATTERN = ALEGRETE_FILE_PATTERN
RADIATION_NOVA_RAMADA_FILE_PATTERN = NOVA_RAMADA_FILE_PATTERN

RADIATION_FINAL_COLUMNS = [
    "datetime",
    "DSWRF_0_6_W_m2",
    "DSWRF_6_12_W_m2",
    "DSWRF_12_18_W_m2",
    "DSWRF_18_24_W_m2",
    "DSWRF_24h_mean_W_m2",
    "DSWRF_24h_MJ_m2",
]
RADIATION_FINAL_OUTPUT_FILE = PROCESSED_OUTPUT_DIR / "gfs_radiation_20190613_20260813.csv"
