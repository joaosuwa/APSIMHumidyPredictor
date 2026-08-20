"""Ponto de entrada da preparação dos dados de modelagem."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:  # Compatibilidade com ``python modeling/main.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modeling.config import DEFAULT_CONFIG
from modeling.data import CYCLE_COLUMNS, PreparedData, prepare_data


def _cycle_count(data) -> int:
    return len(data.loc[:, list(CYCLE_COLUMNS)].drop_duplicates())


def print_summary(prepared: PreparedData) -> None:
    """Exibe os cortes preparados sem realizar treinamento."""
    simulations = sorted(prepared.filtered["SimulationName"].unique())
    print(f"simulações: {simulations}")
    print(
        "dados filtrados: "
        f"{len(prepared.filtered)} linhas, {_cycle_count(prepared.filtered)} ciclos"
    )
    print(
        "desenvolvimento: "
        f"{len(prepared.development)} linhas, {_cycle_count(prepared.development)} ciclos"
    )
    print(f"teste: {len(prepared.test)} linhas, {_cycle_count(prepared.test)} ciclos")
    test_cycle_ids = sorted(int(value) for value in prepared.test["cycle_id"].unique())
    development_cycle_ids = sorted(
        int(value) for value in prepared.development["cycle_id"].unique()
    )
    print(f"cycle_ids de desenvolvimento: {development_cycle_ids}")
    print(f"cycle_ids de teste: {test_cycle_ids}")
    print(f"features: {len(prepared.feature_columns)}; target: {prepared.target_column}")
    print("folds por ciclo:")
    for fold in prepared.folds:
        train_cycle_ids = sorted(
            int(value)
            for value in prepared.development.loc[fold.train_indices, "cycle_id"].unique()
        )
        print(
            f"  treino {train_cycle_ids} ({len(fold.train_indices)} linhas) -> "
            f"validação cycle_id={fold.validation_cycle_id} "
            f"({len(fold.validation_indices)} linhas)"
        )


def main() -> PreparedData:
    prepared = prepare_data(DEFAULT_CONFIG)
    print_summary(prepared)
    return prepared


if __name__ == "__main__":
    main()
