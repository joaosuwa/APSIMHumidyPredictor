"""Processamento de dados meteorológicos da NASA POWER."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ..data_io import write_csv
    from ..paths import PROCESSED_NASA_POWER_DIR, ensure_data_directories
except ImportError:  # Permite executar o ponto de entrada diretamente.
    from data_io import write_csv
    from paths import PROCESSED_NASA_POWER_DIR, ensure_data_directories


ensure_data_directories()


def read_nasa_power_data(path: str | Path) -> pd.DataFrame:
    """Lê um CSV diário da NASA POWER e adiciona a coluna ``date``."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.startswith("YEAR,"))
    except StopIteration as exc:
        raise ValueError(f"Cabeçalho YEAR não encontrado em {path}") from exc

    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df["date"] = pd.to_datetime(
        df["YEAR"].astype(int).astype(str)
        + "-"
        + df["DOY"].astype(int).astype(str),
        format="%Y-%j",
    )
    numeric_columns = [column for column in df.columns if column != "date"]
    df[numeric_columns] = df[numeric_columns].replace(-999, np.nan)
    return df


def process_nasa_power_data(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Normaliza um arquivo NASA POWER e opcionalmente salva o resultado."""
    df = read_nasa_power_data(input_path)
    if output_path is None:
        output_path = PROCESSED_NASA_POWER_DIR / f"{Path(input_path).stem}.processed.csv"
    write_csv(df, output_path)
    return df


def read_precipitation_series(path: str | Path) -> pd.Series:
    """Lê a precipitação observada da NASA POWER como série indexada por data."""
    observed = read_nasa_power_data(path)
    return observed.set_index("date")["PRECTOTCORR"]
