from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_REPORT = DATA_DIR / "milho_nova_ramada.Report.csv"
DEFAULT_OUTPUT = DATA_DIR / "milho_nova_ramada.Report.processed.csv"

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


def _column_types(header: list[str]) -> list[str]:
    """Classifica cada coluna do header como 'text', 'int' ou 'real'."""
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
    """Converte uma linha em valores, reconstruindo numeros com virgula decimal.

    No arquivo original as casas decimais usam virgula (ex.: 7,441 = 7.441),
    entao cada valor real vira dois tokens (parte inteira + fracao de 3 digitos).
    A regra 'tokens restantes > colunas restantes' define quando consumir o par.
    """
    i = 0
    n = len(tokens)
    values = []
    for c, t in enumerate(types):
        cols_left = len(types) - c
        toks_left = n - i
        if t == "text":
            values.append(tokens[i])
            i += 1
        elif t == "int":
            values.append(int(tokens[i]))
            i += 1
        else:
            if toks_left > cols_left:
                values.append(float(f"{tokens[i]}.{tokens[i + 1]}"))
                i += 2
            else:
                values.append(float(tokens[i]))
                i += 1
    if i != n:
        raise ValueError(f"tokens nao consumidos na linha: {tokens[i:]}")
    return values


def _parse_sowing_date(value: str) -> pd.Timestamp:
    """Converte a data de semeadura; '0001-01-01' (sem cultivo) vira NaT."""
    if value in ("0001-01-01", ""):
        return pd.NaT
    return pd.Timestamp(value)


def read_apsim_report(path: str | Path = DEFAULT_REPORT) -> pd.DataFrame:
    """Le o Report.csv do APSIM NG e devolve um DataFrame com tipos corretos."""
    with io.open(path, "r", encoding="utf-8-sig", newline="") as fh:
        lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
    header = lines[0].split(",")
    types = _column_types(header)
    records = [_parse_row(ln.split(","), types) for ln in lines[1:]]

    df = pd.DataFrame.from_records(records, columns=header)
    for name, t in zip(header, types):
        if t == "real":
            df[name] = df[name].astype(float)
        elif t == "int":
            df[name] = df[name].astype("int64")
    df["Clock.Today"] = pd.to_datetime(df["Clock.Today"])
    df["Maize.SowingDate"] = df["Maize.SowingDate"].map(_parse_sowing_date)
    return df


def filter_to_crop_window(df: pd.DataFrame) -> pd.DataFrame:
    """Mantem apenas a janela semeadura -> colheita (dias com cultivo)."""
    df = df.copy()
    df = df.loc[df["Maize.SowingDate"].notna()].reset_index(drop=True)
    df["cycle_id"] = (
        df.groupby("SimulationName")["Maize.SowingDate"]
        .transform(lambda sd: (sd != sd.shift(1)).cumsum())
        - 1
    )
    return df


def _layer_bounds(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extrai os limites superior/inferior (mm) de cada camada do solo."""
    tops, bottoms = [], []
    for col in LAYER_COLS["depth"]:
        top, bottom = df[col].iloc[0].split("-")
        tops.append(float(top))
        bottoms.append(float(bottom))
    return np.asarray(tops), np.asarray(bottoms)


def _root_fractions(
    df: pd.DataFrame, tops: np.ndarray, bottoms: np.ndarray
) -> np.ndarray:
    """Fracao da zona radicular em cada camada (0 = acima, 1 = coberta, 0-1 = parcial)."""
    thickness = bottoms - tops
    zr = df["Maize.Root.Depth"].to_numpy(float)[:, None]
    return np.clip((zr - tops[None, :]) / thickness[None, :], 0.0, 1.0)


def add_apsim_features(df: pd.DataFrame, include_relative: bool = True) -> pd.DataFrame:
    """Adiciona features calculadas com dados do APSIM NG (previsorDeficitHidrico.md)."""
    df = df.copy()

    tops, bottoms = _layer_bounds(df)
    frac = _root_fractions(df, tops, bottoms)

    sw = df[LAYER_COLS["SW"]].to_numpy(float)
    deficit = df[LAYER_COLS["deficit"]].to_numpy(float)
    taw_layer = (
        df[LAYER_COLS["DUL"]].to_numpy(float) - df[LAYER_COLS["LL"]].to_numpy(float)
    )

    df["SoilWater_root"] = np.sum(sw * frac, axis=1)
    df["Dr_root"] = np.sum(deficit * frac, axis=1)
    df["TAW_root"] = np.sum(taw_layer * frac, axis=1)

    df["ETreal"] = df["Soil.SoilWater.Es"] + df["Maize.Leaf.Transpiration"]
    df["ETr_acumulado"] = df.groupby(GROUP_COLS, sort=False)["ETreal"].cumsum()
    df["Irrigacao_dia_posterior"] = df.groupby(GROUP_COLS, sort=False)[
        "Irrigation.IrrigationApplied"
    ].shift(-1)
    df["Umidade_solo_passada_1d"] = df.groupby(GROUP_COLS, sort=False)[
        "SoilWater_root"
    ].shift(1)

    doy = df["Clock.Today"].dt.dayofyear.to_numpy(float)
    month = df["Clock.Today"].dt.month.to_numpy(float)
    df["DOY_sin"] = np.sin(2.0 * np.pi * doy / 365.0)
    df["DOY_cos"] = np.cos(2.0 * np.pi * doy / 365.0)
    df["Month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    df["Month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    if include_relative:
        df["SW_frac_TAW"] = df["SoilWater_root"] / df["TAW_root"]
        df["Dr_frac_TAW"] = df["Dr_root"] / df["TAW_root"]

    return df


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove as colunas da lista do DataFrame (ignora nomes inexistentes)."""
    return df.drop(columns=[c for c in columns if c in df.columns])


def build_report_features(
    report_path: str | Path = DEFAULT_REPORT,
    output_path: str | Path = DEFAULT_OUTPUT,
    include_relative: bool = True,
    columns_to_drop: list[str] | None = None,
) -> pd.DataFrame:
    """Le, filtra e enriquece o Report.csv, salvando o resultado em CSV.

    Se columns_to_drop for informado, remove essas colunas antes de salvar.
    """
    df = read_apsim_report(report_path)
    df = filter_to_crop_window(df)
    df = add_apsim_features(df, include_relative=include_relative)
    if columns_to_drop:
        df = drop_columns(df, columns_to_drop)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    full = read_apsim_report()
    result = build_report_features()
    print(f"linhas no arquivo original: {len(full)}")
    print(f"linhas na janela de cultivo: {len(result)}")
    print(f"simulacoes: {sorted(result['SimulationName'].unique())}")
    print(f"ciclos (semeadura->colheita): {int(result['cycle_id'].nunique())}")
    print(f"colunas novas: {sorted(set(result.columns) - set(full.columns))}")
    print(result[["SoilWater_root", "Dr_root", "TAW_root", "ETreal"]].describe())
