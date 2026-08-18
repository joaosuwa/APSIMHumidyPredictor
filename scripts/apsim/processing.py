"""Processamento de relatórios do APSIM NG e criação de features."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ..data_io import write_csv
    from ..paths import MODEL_DATA_DIR, PROCESSED_APSIM_DIR, RAW_APSIM_DIR, ensure_data_directories
except ImportError:  # Permite executar o ponto de entrada diretamente.
    from data_io import write_csv
    from paths import MODEL_DATA_DIR, PROCESSED_APSIM_DIR, RAW_APSIM_DIR, ensure_data_directories


ensure_data_directories()

DEFAULT_REPORT = RAW_APSIM_DIR / "milho.Report.csv"
DEFAULT_OUTPUT = PROCESSED_APSIM_DIR / "milho.Report.processed.csv"
DEFAULT_MODEL_DATASET = MODEL_DATA_DIR / "training_dataset.csv"

N_LAYERS = 7

TEXT_COLUMNS = {
    "SimulationName",
    "CheckpointName",
    "Clock.Today",
    "Maize.SowingDate",
    "Zone",
} | {f"Soil.Physical.Depth({i})" for i in range(1, N_LAYERS + 1)}

INT_COLUMNS = {
    "SimulationID",
    "CheckpointID",
    "Maize.DaysAfterSowing",
    "Maize.IsReadyForHarvesting",
}

LAYER_COLS = {
    "deficit": [f"deficit({i})" for i in range(1, N_LAYERS + 1)],
    "DUL": [f"DULreal({i})" for i in range(1, N_LAYERS + 1)],
    "LL": [f"LLreal({i})" for i in range(1, N_LAYERS + 1)],
    "SW": [f"Soil.Water.MM({i})" for i in range(1, N_LAYERS + 1)],
    "depth": [f"Soil.Physical.Depth({i})" for i in range(1, N_LAYERS + 1)],
}

GROUP_COLS = ["SimulationName", "cycle_id"]

DEFAULT_COLUMNS_TO_DROP = [
    "SimulationID",
    "CheckpointID",
    "CheckpointName",
    *LAYER_COLS["deficit"],
    *LAYER_COLS["DUL"],
    *LAYER_COLS["LL"],
    "Maize.IsReadyForHarvesting",
    "Maize.SowingDate",
    *[f"SATreal({i})" for i in range(1, N_LAYERS + 1)],
    *LAYER_COLS["depth"],
    "Maize.Leaf.Transpiration",
    "Soil.SoilWater.Es",
    *LAYER_COLS["SW"],
    "Yield",
    "Zone",
    "cycle_id",
]


def _column_types(header: list[str]) -> list[str]:
    """Classifica cada coluna do header como texto, inteiro ou real."""
    types = []
    for name in header:
        if name in TEXT_COLUMNS:
            types.append("text")
        elif name in INT_COLUMNS:
            types.append("int")
        else:
            types.append("real")
    return types


def _parse_row(tokens: list[str], types: list[str]) -> list[object]:
    """Converte uma linha do Report.csv, tratando decimais separados por vírgula."""
    i = 0
    n = len(tokens)
    values = []
    for column_index, value_type in enumerate(types):
        columns_left = len(types) - column_index
        tokens_left = n - i
        if value_type == "text":
            values.append(tokens[i])
            i += 1
        elif value_type == "int":
            values.append(int(tokens[i]))
            i += 1
        else:
            if tokens_left > columns_left:
                values.append(float(f"{tokens[i]}.{tokens[i + 1]}"))
                i += 2
            else:
                values.append(float(tokens[i]))
                i += 1
    if i != n:
        raise ValueError(f"tokens não consumidos na linha: {tokens[i:]}")
    return values


def _parse_sowing_date(value: str) -> pd.Timestamp:
    """Converte a data de semeadura; 0001-01-01 representa ausência de cultivo."""
    if value in ("0001-01-01", ""):
        return pd.NaT
    return pd.Timestamp(value)


def read_apsim_report(path: str | Path = DEFAULT_REPORT) -> pd.DataFrame:
    """Lê o Report.csv do APSIM NG e devolve um DataFrame tipado."""
    with io.open(path, "r", encoding="utf-8-sig", newline="") as fh:
        lines = [line.strip() for line in fh.read().splitlines() if line.strip()]
    header = lines[0].split(",")
    types = _column_types(header)
    records = [_parse_row(line.split(","), types) for line in lines[1:]]

    df = pd.DataFrame.from_records(records, columns=header)
    for name, value_type in zip(header, types):
        if value_type == "real":
            df[name] = df[name].astype(float)
        elif value_type == "int":
            df[name] = df[name].astype("int64")
    df["Clock.Today"] = pd.to_datetime(df["Clock.Today"])
    df["Maize.SowingDate"] = df["Maize.SowingDate"].map(_parse_sowing_date)
    return df


def filter_to_crop_window(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém apenas os registros dentro da janela de cultivo."""
    df = df.copy()
    df = df.loc[df["Maize.SowingDate"].notna()].reset_index(drop=True)
    df["cycle_id"] = (
        df.groupby("SimulationName")["Maize.SowingDate"]
        .transform(lambda sowing_date: (sowing_date != sowing_date.shift(1)).cumsum())
        - 1
    )
    return df


def _layer_bounds(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extrai os limites superior e inferior de cada camada do solo."""
    tops, bottoms = [], []
    for column in LAYER_COLS["depth"]:
        top, bottom = df[column].iloc[0].split("-")
        tops.append(float(top))
        bottoms.append(float(bottom))
    return np.asarray(tops), np.asarray(bottoms)


def _root_fractions(
    df: pd.DataFrame,
    tops: np.ndarray,
    bottoms: np.ndarray,
) -> np.ndarray:
    """Calcula a fração da zona radicular presente em cada camada."""
    thickness = bottoms - tops
    root_depth = df["Maize.Root.Depth"].to_numpy(float)[:, None]
    return np.clip((root_depth - tops[None, :]) / thickness[None, :], 0.0, 1.0)


def add_apsim_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features calculadas exclusivamente com dados do APSIM NG."""
    df = df.copy()
    tops, bottoms = _layer_bounds(df)
    fractions = _root_fractions(df, tops, bottoms)

    soil_water = df[LAYER_COLS["SW"]].to_numpy(float)
    deficit = df[LAYER_COLS["deficit"]].to_numpy(float)
    taw_layer = (
        df[LAYER_COLS["DUL"]].to_numpy(float)
        - df[LAYER_COLS["LL"]].to_numpy(float)
    )

    df["SoilWater_root"] = np.sum(soil_water * fractions, axis=1)
    df["Dr_root"] = np.sum(deficit * fractions, axis=1)
    df["TAW_root"] = np.sum(taw_layer * fractions, axis=1)

    df["ETreal"] = df["Soil.SoilWater.Es"] + df["Maize.Leaf.Transpiration"]
    df["ETr_acumulado"] = df.groupby(GROUP_COLS, sort=False)["ETreal"].cumsum()
    df["Irrigacao_dia_posterior"] = df.groupby(GROUP_COLS, sort=False)[
        "Irrigation.IrrigationApplied"
    ].shift(-1)
    soil_group = df.groupby(GROUP_COLS, sort=False)["SoilWater_root"]
    for days in range(1, 4):
        df[f"Umidade_solo_passada_{days}d"] = soil_group.shift(days)

    water_input = (
        df["Weather.Rain"].fillna(0.0)
        + df["Irrigation.IrrigationApplied"].fillna(0.0)
    )
    water_group = water_input.groupby(
        [df[column] for column in GROUP_COLS],
        sort=False,
    )
    for days in range(1, 4):
        df[f"Chuva_Irrigacao_passada_{days}d"] = water_group.shift(days)

    doy = df["Clock.Today"].dt.dayofyear.to_numpy(float)
    month = df["Clock.Today"].dt.month.to_numpy(float)
    df["DOY_sin"] = np.sin(2.0 * np.pi * doy / 365.0)
    df["DOY_cos"] = np.cos(2.0 * np.pi * doy / 365.0)
    df["Month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    df["Month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    return df


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove as colunas informadas, ignorando nomes inexistentes."""
    return df.drop(columns=[column for column in columns if column in df.columns])


def build_report_features(
    report_path: str | Path = DEFAULT_REPORT,
    output_path: str | Path = DEFAULT_OUTPUT,
    columns_to_drop: list[str] | None = None,
) -> pd.DataFrame:
    """Lê, enriquece, reduz e salva o relatório do APSIM NG.

    Por padrão, remove as colunas brutas e de identificação que não serão
    utilizadas pelo modelo. Para manter todas as colunas, passe
    ``columns_to_drop=[]``. Uma lista própria substitui a lista padrão.
    """
    df = read_apsim_report(report_path)
    df = filter_to_crop_window(df)
    df = add_apsim_features(df)
    columns = DEFAULT_COLUMNS_TO_DROP if columns_to_drop is None else columns_to_drop
    if columns:
        df = drop_columns(df, columns)
    df = df.dropna().reset_index(drop=True)
    write_csv(df, output_path)
    return df
