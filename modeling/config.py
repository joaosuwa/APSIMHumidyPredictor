"""Configuração tipada da preparação dos dados de modelagem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "model" / "training_dataset.csv"


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Parâmetros necessários para filtrar e dividir o dataset."""

    included_simulations: tuple[str, ...]
    test_cycle_ids: tuple[int, ...] = (6,)
    dataset_path: Path = DEFAULT_DATASET_PATH

    def __post_init__(self) -> None:
        if not self.included_simulations:
            raise ValueError("included_simulations não pode ser vazio")
        if len(set(self.included_simulations)) != len(self.included_simulations):
            raise ValueError("included_simulations contém nomes duplicados")
        if not self.test_cycle_ids:
            raise ValueError("test_cycle_ids não pode ser vazio")
        if any(
            isinstance(cycle_id, bool) or not isinstance(cycle_id, int)
            for cycle_id in self.test_cycle_ids
        ):
            raise TypeError("Cada test_cycle_id deve ser um inteiro")
        if any(cycle_id < 0 for cycle_id in self.test_cycle_ids):
            raise ValueError("test_cycle_ids não pode conter valores negativos")
        if len(set(self.test_cycle_ids)) != len(self.test_cycle_ids):
            raise ValueError("test_cycle_ids contém valores duplicados")


DEFAULT_CONFIG = DataConfig(
    included_simulations=(
        "Alegrete",
        "AlegreteClay",
        "Simulation",
        "SimulationClay",
    ),
    test_cycle_ids=(6,),
)
