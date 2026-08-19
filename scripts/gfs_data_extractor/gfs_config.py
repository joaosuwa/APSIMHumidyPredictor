"""Configuração compartilhada pelo download e processamento do GFS/GDEX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..paths import PROCESSED_GFS_DIR, RAW_GFS_DIR


DATASET = "d084001"
DEFAULT_START_DATE = "201906130000"
DEFAULT_END_DATE = "202608191200"
RAW_OUTPUT_DIR = RAW_GFS_DIR / "downloads"
PROCESSED_OUTPUT_DIR = PROCESSED_GFS_DIR

ProductFamily = Literal["temperature_max", "temperature_min", "radiation", "forecast_24h"]


@dataclass(frozen=True, slots=True)
class SiteConfig:
    """Metadados de uma localidade usada pelo pipeline meteorológico."""

    key: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float
    output_suffix: str
    nasa_raw_filename: str


@dataclass(frozen=True, slots=True)
class ProductConfig:
    """Metadados de download e transformação de um produto GFS."""

    key: str
    raw_directory: str
    param: str
    product: str
    family: ProductFamily
    output_column: str
    level: str | None
    interval_end_hour: int | None = None
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE


SITES: dict[str, SiteConfig] = {
    "alegrete": SiteConfig(
        key="alegrete",
        name="Alegrete",
        latitude=-29.75,
        longitude=-55.75,
        elevation_m=102.0,
        output_suffix="Alegrete",
        nasa_raw_filename="ClimaAlegrete2019-2026.csv",
    ),
    "nova_ramada": SiteConfig(
        key="nova_ramada",
        name="NovaRamada",
        latitude=-28.0,
        longitude=-53.75,
        elevation_m=511.0,
        output_suffix="Nova_Ramada",
        nasa_raw_filename="ClimaNovaRamada_2019-2026.csv",
    ),
}


def _six_hour_product(
    key: str,
    param: str,
    product: str,
    family: ProductFamily,
    interval_end_hour: int,
    level: str = "HTGL:2",
) -> ProductConfig:
    return ProductConfig(
        key=key.lower(),
        raw_directory=key,
        param=param,
        product=product,
        family=family,
        output_column=key,
        level=level,
        interval_end_hour=interval_end_hour,
    )


GFS_PRODUCTS: dict[str, ProductConfig] = {
    product.key: product
    for product in (
        _six_hour_product("Tmax_0_6", "T MAX", "6-hour Maximum (initial+0 to initial+6)", "temperature_max", 6),
        _six_hour_product("Tmax_6_12", "T MAX", "6-hour Maximum (initial+6 to initial+12)", "temperature_max", 12),
        _six_hour_product("Tmax_12_18", "T MAX", "6-hour Maximum (initial+12 to initial+18)", "temperature_max", 18),
        _six_hour_product("Tmax_18_24", "T MAX", "6-hour Maximum (initial+18 to initial+24)", "temperature_max", 0),
        _six_hour_product("Tmin_0_6", "T MIN", "6-hour Minimum (initial+0 to initial+6)", "temperature_min", 6),
        _six_hour_product("Tmin_6_12", "T MIN", "6-hour Minimum (initial+6 to initial+12)", "temperature_min", 12),
        _six_hour_product("Tmin_12_18", "T MIN", "6-hour Minimum (initial+12 to initial+18)", "temperature_min", 18),
        _six_hour_product("Tmin_18_24", "T MIN", "6-hour Minimum (initial+18 to initial+24)", "temperature_min", 0),
        _six_hour_product("DSWRF_0_6", "DSWRF", "6-hour Average (initial+0 to initial+6)", "radiation", 6),
        _six_hour_product("DSWRF_6_12", "DSWRF", "6-hour Average (initial+6 to initial+12)", "radiation", 12),
        _six_hour_product("DSWRF_12_18", "DSWRF", "6-hour Average (initial+12 to initial+18)", "radiation", 18),
        _six_hour_product("DSWRF_18_24", "DSWRF", "6-hour Average (initial+18 to initial+24)", "radiation", 0),
        ProductConfig(
            key="apcp_24",
            raw_directory="A PCP_24",
            param="A PCP",
            product="24-hour Accumulation (initial+0 to initial+24)",
            family="forecast_24h",
            output_column="previsao_chuva_24h_mm",
            level=None,
            start_date="201906121200",
            end_date="202608191200",
        ),
        ProductConfig("u_grd_24", "U GRD_24", "U GRD", "24-hour Forecast", "forecast_24h", "u_grd_10m_m_s", "HTGL:10"),
        ProductConfig("v_grd_24", "V GRD_24", "V GRD", "24-hour Forecast", "forecast_24h", "v_grd_10m_m_s", "HTGL:10"),
        ProductConfig("rh_24", "R H_24", "R H", "24-hour Forecast", "forecast_24h", "umidade_relativa_prevista_pct", "HTGL:2"),
        ProductConfig("dpt_24", "DPT_24", "DPT", "24-hour Forecast", "forecast_24h", "ponto_orvalho_previsto_C", "HTGL:2"),
    )
}

TEMPERATURE_MAX_KEYS = ("tmax_0_6", "tmax_6_12", "tmax_12_18", "tmax_18_24")
TEMPERATURE_MIN_KEYS = ("tmin_0_6", "tmin_6_12", "tmin_12_18", "tmin_18_24")
RADIATION_KEYS = ("dswrf_0_6", "dswrf_6_12", "dswrf_12_18", "dswrf_18_24")
FORECAST_24H_KEYS = ("apcp_24", "u_grd_24", "v_grd_24", "rh_24", "dpt_24")
PRODUCT_GROUPS: dict[str, tuple[str, ...]] = {
    "forecast_24h": FORECAST_24H_KEYS,
    "temperature": TEMPERATURE_MAX_KEYS + TEMPERATURE_MIN_KEYS,
    "radiation": RADIATION_KEYS,
    "all": tuple(GFS_PRODUCTS),
}


def get_site(site: str | SiteConfig) -> SiteConfig:
    """Resolve uma chave ou nome de localidade para sua configuração."""
    if isinstance(site, SiteConfig):
        return site
    normalized = site.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {config.name.lower(): key for key, config in SITES.items()}
    normalized = aliases.get(normalized.replace("_", ""), normalized)
    try:
        return SITES[normalized]
    except KeyError as exc:
        raise ValueError(f"Localidade GFS não configurada: {site}") from exc


def get_product(product: str | ProductConfig) -> ProductConfig:
    """Resolve a chave de um produto GFS."""
    if isinstance(product, ProductConfig):
        return product
    try:
        return GFS_PRODUCTS[product.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Produto GFS não configurado: {product}") from exc


def resolve_product_keys(values: list[str] | tuple[str, ...]) -> list[str]:
    """Expande grupos da CLI e elimina produtos repetidos mantendo a ordem."""
    resolved: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        keys = PRODUCT_GROUPS.get(normalized, (normalized,))
        for key in keys:
            get_product(key)
            if key not in resolved:
                resolved.append(key)
    return resolved


def format_coordinate(value: float, positive_suffix: str, negative_suffix: str) -> str:
    """Formata uma coordenada no padrão usado nos nomes do GDEX."""
    numeric_value = float(value)
    number = (
        f"{abs(numeric_value):.1f}"
        if numeric_value.is_integer()
        else f"{abs(numeric_value):.6f}".rstrip("0").rstrip(".")
    )
    suffix = positive_suffix if numeric_value >= 0 else negative_suffix
    return f"{number}{suffix}"


def coordinate_file_pattern(latitude: float, longitude: float) -> str:
    """Retorna o glob para arquivos GFS de uma coordenada específica."""
    latitude_name = format_coordinate(latitude, "N", "S")
    longitude_name = format_coordinate(longitude, "E", "W")
    return f"*_{latitude_name}_{longitude_name}.csv"


def processed_forecast_path(site: str | SiteConfig) -> Path:
    """Caminho estável do forecast diário consolidado de uma localidade."""
    config = get_site(site)
    return PROCESSED_OUTPUT_DIR / f"gfs_daily_forecast_{config.output_suffix}.csv"
