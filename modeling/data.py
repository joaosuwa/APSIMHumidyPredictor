"""Filtragem, holdout e cross-validation agrupados por ciclo."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from .config import DataConfig


SIMULATION_COLUMN = "SimulationName"
CYCLE_COLUMN = "cycle_id"
DATE_COLUMN = "Clock_today"
SOWING_DATE_COLUMN = "sowing_date"
TARGET_COLUMN = "deficit_agua_proximo_dia_mm"
CYCLE_COLUMNS = (SIMULATION_COLUMN, CYCLE_COLUMN)

NON_IRRIGATED_SIMULATIONS = frozenset(
    {"Alegrete", "AlegreteClay", "Simulation", "SimulationClay"}
)
DIRECT_IRRIGATION_COLUMNS = (
    "irrigacao_aplicada_mm",
    "irrigacao_aplicada_dia_posterior_mm",
)
METADATA_COLUMNS = (
    SIMULATION_COLUMN,
    DATE_COLUMN,
    SOWING_DATE_COLUMN,
    CYCLE_COLUMN,
)
MODELING_FEATURE_COLUMNS = (
    "umidade_solo_mm",
    "umidade_solo_passada_1d_mm",
    "umidade_solo_passada_2d_mm",
    "umidade_solo_passada_3d_mm",
    "precipitacao_observada_mm",
    "chuva_irrigacao_passada_1d_mm",
    "chuva_irrigacao_passada_2d_mm",
    "chuva_irrigacao_passada_3d_mm",
    "etreal_mm_dia",
    "previsao_chuva_24h_mm",
    "previsao_temperatura_maxima_C",
    "previsao_temperatura_minima_C",
    "previsao_radiacao_solar_MJ_m2_dia",
    "previsao_eto_mm_dia",
    "temperatura_media_C",
    "temperatura_maxima_C",
    "temperatura_minima_C",
    "umidade_relativa_pct",
    "velocidade_vento_m_s",
    "radiacao_solar_MJ_m2_dia",
    "dr_mm",
    "profundidade_radicular_mm",
    "gdd_acumulado_C_dia",
    "etr_acumulado_mm",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
    "escoamento_superficial_mm",
    "drenagem_mm",
    "taw_mm",
    "dias_desde_semeadura",
)


@dataclass(frozen=True, slots=True)
class CycleFold:
    """Posições de um fold relativas ao DataFrame de desenvolvimento."""

    validation_cycle_id: int
    train_indices: pd.Index
    validation_indices: pd.Index


@dataclass(frozen=True, slots=True)
class PreparedData:
    """Dados e metadados prontos para o treinamento futuro."""

    filtered: pd.DataFrame
    development: pd.DataFrame
    test: pd.DataFrame
    folds: tuple[CycleFold, ...]
    feature_columns: tuple[str, ...]
    target_column: str


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset sem colunas obrigatórias: {missing}")


def _duplicates(values: Iterable[object]) -> list[object]:
    return [value for value, count in Counter(values).items() if count > 1]


def load_training_dataset(path: str | Path) -> pd.DataFrame:
    """Carrega o CSV central e tipa suas colunas de data."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset de treinamento não encontrado: {dataset_path}")
    df = pd.read_csv(dataset_path)
    _require_columns(df, (*METADATA_COLUMNS, TARGET_COLUMN))
    for column in (DATE_COLUMN, SOWING_DATE_COLUMN):
        df[column] = pd.to_datetime(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"{column} contém valores ausentes ou inválidos")
    return df


def filter_simulations(df: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """Retorna uma cópia apenas com as simulações solicitadas, na ordem original."""
    _require_columns(df, (SIMULATION_COLUMN,))
    requested = tuple(names)
    if not requested:
        raise ValueError("A lista de simulações não pode ser vazia")
    duplicate_names = _duplicates(requested)
    if duplicate_names:
        raise ValueError(f"Simulações duplicadas na configuração: {duplicate_names}")

    available = set(df[SIMULATION_COLUMN].dropna().unique())
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(f"Simulações não encontradas no dataset: {unknown}")
    return df.loc[df[SIMULATION_COLUMN].isin(requested)].copy().reset_index(drop=True)


def _validate_cycle_alignment(df: pd.DataFrame) -> None:
    """Garante que todo cycle_id existe em todas as simulações filtradas."""
    _require_columns(df, CYCLE_COLUMNS)
    expected_simulations = set(df[SIMULATION_COLUMN].unique())
    simulations_by_cycle = df.groupby(CYCLE_COLUMN)[SIMULATION_COLUMN].agg(set)
    misaligned = {
        int(cycle_id): sorted(expected_simulations - simulations)
        for cycle_id, simulations in simulations_by_cycle.items()
        if simulations != expected_simulations
    }
    if misaligned:
        raise ValueError(f"cycle_id sem todas as simulações selecionadas: {misaligned}")


def split_test_cycle_ids(
    df: pd.DataFrame,
    cycle_ids: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa os mesmos ciclos completos de todas as simulações para teste."""
    _require_columns(df, CYCLE_COLUMNS)
    requested = tuple(cycle_ids)
    if not requested:
        raise ValueError("A lista de cycle_ids de teste não pode ser vazia")
    if any(isinstance(cycle_id, bool) or not isinstance(cycle_id, int) for cycle_id in requested):
        raise TypeError("Cada cycle_id de teste deve ser um inteiro")
    duplicate_ids = _duplicates(requested)
    if duplicate_ids:
        raise ValueError(f"cycle_ids de teste duplicados: {duplicate_ids}")

    _validate_cycle_alignment(df)
    available_ids = set(int(value) for value in df[CYCLE_COLUMN].unique())
    missing_ids = sorted(set(requested) - available_ids)
    if missing_ids:
        raise ValueError(f"cycle_ids de teste não encontrados: {missing_ids}")

    test_mask = df[CYCLE_COLUMN].isin(requested)
    development = df.loc[~test_mask].copy().reset_index(drop=True)
    test = df.loc[test_mask].copy().reset_index(drop=True)
    if development.empty or test.empty:
        raise ValueError("A divisão deve produzir conjuntos de desenvolvimento e teste não vazios")

    development_ids = set(int(value) for value in development[CYCLE_COLUMN].unique())
    if min(requested) <= max(development_ids):
        raise ValueError(
            "Todos os cycle_ids de teste devem ser posteriores aos ciclos de desenvolvimento"
        )
    return development, test


def build_cycle_folds(development: pd.DataFrame) -> tuple[CycleFold, ...]:
    """Cria um fold por cycle_id, mantendo cada ciclo completamente agrupado."""
    _require_columns(development, CYCLE_COLUMNS)
    _validate_cycle_alignment(development)
    groups = development[CYCLE_COLUMN]
    if groups.nunique() < 2:
        raise ValueError("São necessários pelo menos dois cycle_ids para cross-validation")

    folds = []
    splitter = LeaveOneGroupOut()
    for train_positions, validation_positions in splitter.split(development, groups=groups):
        validation_ids = groups.iloc[validation_positions].unique()
        validation_cycle_id = int(validation_ids[0])
        folds.append(
            CycleFold(
                validation_cycle_id=validation_cycle_id,
                train_indices=development.index.take(train_positions),
                validation_indices=development.index.take(validation_positions),
            )
        )
    return tuple(folds)


def prepare_data(config: DataConfig) -> PreparedData:
    """Carrega, valida, filtra e divide o dataset conforme a configuração."""
    configured_simulations = set(config.included_simulations)
    unsupported = sorted(configured_simulations - NON_IRRIGATED_SIMULATIONS)
    if unsupported:
        raise ValueError(
            "Esta etapa aceita somente simulações não irrigadas; "
            f"recebidas: {unsupported}"
        )

    df = load_training_dataset(config.dataset_path)
    _require_columns(
        df,
        (*MODELING_FEATURE_COLUMNS, *DIRECT_IRRIGATION_COLUMNS, TARGET_COLUMN),
    )
    filtered = filter_simulations(df, config.included_simulations)
    _validate_cycle_alignment(filtered)

    irrigation = filtered.loc[:, list(DIRECT_IRRIGATION_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    if irrigation.isna().any().any() or irrigation.ne(0.0).any().any():
        raise ValueError("As simulações não irrigadas possuem valores de irrigação inesperados")

    development, test = split_test_cycle_ids(filtered, config.test_cycle_ids)
    folds = build_cycle_folds(development)
    return PreparedData(
        filtered=filtered,
        development=development,
        test=test,
        folds=folds,
        feature_columns=MODELING_FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )
