"""Leitura, validação e cortes temporais do dataset de treinamento."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.model_dataset import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    TARGET_COLUMNS,
    VARIATION_TARGET_COLUMN,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "processed" / "model" / "training_dataset.csv"


@dataclass(frozen=True)
class TemporalSplits:
    tuning_train: pd.DataFrame
    validation: pd.DataFrame
    final_train: pd.DataFrame
    test: pd.DataFrame


def load_dataset(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    """Carrega o CSV e valida esquema, alvos e metadados temporais."""
    df = pd.read_csv(path, parse_dates=["data", "data_alvo"])
    required = METADATA_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset sem colunas obrigatórias: {missing}")
    if df[required].isna().any().any():
        nulls = df[required].isna().sum()
        raise ValueError(f"Dataset contém valores ausentes: {nulls[nulls > 0].to_dict()}")

    expected_variation = df[TARGET_COLUMN] - df["dr_mm"]
    if not np.allclose(df[VARIATION_TARGET_COLUMN], expected_variation):
        raise ValueError("Target de variação não corresponde a Dr(D+1) - Dr(D)")
    if not ((df["data_alvo"] - df["data"]) == pd.Timedelta(days=1)).all():
        raise ValueError("Há targets que não correspondem exatamente ao dia D+1")
    return df.sort_values(["ano_semeadura", "data", "simulation_name", "cycle_id"]).reset_index(drop=True)


def make_temporal_splits(df: pd.DataFrame) -> TemporalSplits:
    """Separa tuning, validação, treino final e teste por ano de semeadura."""
    tuning_train = df[df["ano_semeadura"].between(2019, 2023)].copy()
    validation = df[df["ano_semeadura"] == 2024].copy()
    final_train = df[df["ano_semeadura"].between(2019, 2024)].copy()
    test = df[df["ano_semeadura"] == 2025].copy()

    frames = {
        "tuning_train": tuning_train,
        "validation": validation,
        "final_train": final_train,
        "test": test,
    }
    empty = [name for name, frame in frames.items() if frame.empty]
    if empty:
        raise ValueError(f"Cortes temporais vazios: {empty}")
    if final_train["ano_semeadura"].max() >= test["ano_semeadura"].min():
        raise ValueError("Treino final e teste não respeitam a ordem temporal")

    train_keys = set(zip(final_train["simulation_name"], final_train["cycle_id"], final_train["data"]))
    test_keys = set(zip(test["simulation_name"], test["cycle_id"], test["data"]))
    if train_keys & test_keys:
        raise ValueError("Há observações repetidas entre treino final e teste")
    return TemporalSplits(**frames)


def feature_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Retorna somente features numéricas e o target de variação."""
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    forbidden = set(METADATA_COLUMNS + TARGET_COLUMNS)
    leaked = forbidden.intersection(features.columns)
    if leaked:
        raise ValueError(f"Metadados ou targets presentes nas features: {sorted(leaked)}")
    return features, frame[VARIATION_TARGET_COLUMN].copy()


def split_summary(splits: TemporalSplits) -> pd.DataFrame:
    """Resume tamanho, período e presença de cenários irrigados em cada corte."""
    records = []
    for name, frame in (
        ("tuning_train", splits.tuning_train),
        ("validation", splits.validation),
        ("final_train", splits.final_train),
        ("test", splits.test),
    ):
        records.append(
            {
                "split": name,
                "rows": len(frame),
                "sowing_year_min": int(frame["ano_semeadura"].min()),
                "sowing_year_max": int(frame["ano_semeadura"].max()),
                "date_min": frame["data"].min().date().isoformat(),
                "date_max": frame["data"].max().date().isoformat(),
                "irrigated_rows": int(frame["cenario_irrigado"].sum()),
                "future_irrigation_events": int((frame["irrigacao_aplicada_dia_posterior_mm"] > 0).sum()),
            }
        )
    return pd.DataFrame.from_records(records)
