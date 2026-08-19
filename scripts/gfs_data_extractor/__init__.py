"""Integração com o GDEX/GFS e consolidação dos dados meteorológicos."""

from .gfs_config import ProductConfig, SiteConfig
from .gfs_data_processing import (
    generate_site_forecast,
    load_daily_24h_product,
    load_six_hour_intervals,
)

__all__ = [
    "ProductConfig",
    "SiteConfig",
    "generate_site_forecast",
    "load_daily_24h_product",
    "load_six_hour_intervals",
]
