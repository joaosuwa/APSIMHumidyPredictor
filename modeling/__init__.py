"""Preparação, otimização e treinamento dos modelos de déficit hídrico."""

from .config import (
    DEFAULT_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DataConfig,
    TrainingConfig,
)
from .data import (
    NEXT_DEFICIT_COLUMN,
    NEXT_DAY_OBSERVED_RAIN_COLUMN,
    TARGET_COLUMN,
    VARIATION_TARGET_COLUMN,
    CycleFold,
    PreparedData,
    prepare_data,
)
from .pipeline import (
    MODEL_NAMES,
    CrossValidationResult,
    ModelName,
    TuningResult,
    load_trained_model,
)

__all__ = [
    "CycleFold",
    "DataConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    "MODEL_NAMES",
    "ModelName",
    "NEXT_DEFICIT_COLUMN",
    "NEXT_DAY_OBSERVED_RAIN_COLUMN",
    "PreparedData",
    "TARGET_COLUMN",
    "TrainingConfig",
    "CrossValidationResult",
    "TuningResult",
    "VARIATION_TARGET_COLUMN",
    "load_trained_model",
    "prepare_data",
]
