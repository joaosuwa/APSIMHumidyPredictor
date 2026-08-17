"""Leitura e normalização de dados da NASA POWER."""

from .processing import (
    process_nasa_power_data,
    read_nasa_power_data,
    read_precipitation_series,
)

__all__ = [
    "process_nasa_power_data",
    "read_nasa_power_data",
    "read_precipitation_series",
]
