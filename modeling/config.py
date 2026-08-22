"""Configuração tipada da preparação e do treinamento dos modelos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "model" / "training_dataset.csv"
DEFAULT_ARTIFACTS_PATH = PROJECT_ROOT / "modeling" / "artifacts" / "default"


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


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Parâmetros reproduzíveis do tuning e do treinamento final."""

    n_trials: int = 50
    seed: int = 42
    max_iterations: int = 3000
    early_stopping_rounds: int = 100
    output_dir: Path = DEFAULT_ARTIFACTS_PATH

    def __post_init__(self) -> None:
        for name in ("n_trials", "max_iterations", "early_stopping_rounds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} deve ser um inteiro")
            if value <= 0:
                raise ValueError(f"{name} deve ser maior que zero")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed deve ser um inteiro")
        object.__setattr__(self, "output_dir", Path(self.output_dir))


DEFAULT_CONFIG = DataConfig(
    included_simulations=(
        "Alegrete",
        "AlegreteClay",
        "Simulation",
        "SimulationClay",
    ),
    test_cycle_ids=(6,),
)

DEFAULT_TRAINING_CONFIG = TrainingConfig()
