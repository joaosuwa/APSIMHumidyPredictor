from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from modeling.config import DEFAULT_CONFIG, TrainingConfig
from modeling.data import prepare_data
from modeling.evaluation import persistence_prediction, regression_metrics
from modeling.main import run_pipeline
from modeling.models import (
    MODEL_NAMES,
    build_model,
    load_trained_model,
    suggest_parameters,
)
from modeling.training import build_training_fingerprint


FIXED_PARAMETERS = {
    "xgboost": {
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 2.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.01,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
    },
    "lightgbm": {
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
    },
    "catboost": {
        "learning_rate": 0.1,
        "depth": 5,
        "l2_leaf_reg": 3.0,
        "random_strength": 0.1,
        "bagging_temperature": 1.0,
    },
}


def test_metrics_and_persistence() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    predicted = persistence_prediction([1.0, 1.0, 4.0])
    metrics = regression_metrics(actual, predicted)
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["bias"] == pytest.approx(0.0)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_search_spaces_and_model_factories(model_name: str) -> None:
    trial = optuna.trial.FixedTrial(FIXED_PARAMETERS[model_name])
    parameters = suggest_parameters(model_name, trial)
    assert parameters == FIXED_PARAMETERS[model_name]
    model = build_model(
        model_name,
        parameters,
        seed=42,
        iterations=5,
        early_stopping_rounds=2,
    )
    assert model is not None


def test_fingerprint_changes_with_training_decisions(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("x,y\n1,2\n", encoding="utf-8")
    common = dict(
        feature_columns=("x",),
        target_column="y",
        included_simulations=("A",),
        test_cycle_ids=(6,),
        seed=42,
        max_iterations=10,
        early_stopping_rounds=2,
    )
    first, dataset_hash = build_training_fingerprint(dataset, **common)
    second, _ = build_training_fingerprint(
        dataset, **{**common, "feature_columns": ("x", "z")}
    )
    assert first != second
    assert len(dataset_hash) == 64


def test_complete_smoke_pipeline_and_native_models(tmp_path: Path) -> None:
    config = TrainingConfig(
        n_trials=1,
        seed=42,
        max_iterations=8,
        early_stopping_rounds=3,
        output_dir=tmp_path / "artifacts",
    )
    result = run_pipeline(DEFAULT_CONFIG, config)

    assert result.champion in MODEL_NAMES
    assert set(result.model_paths) == set(MODEL_NAMES)
    assert all(path.is_file() for path in result.model_paths.values())
    results_dir = result.artifact_root / "results"
    plots_dir = result.artifact_root / "plots"
    predictions = pd.read_csv(results_dir / "test_predictions.csv")
    oof = pd.read_csv(results_dir / "oof_predictions.csv")
    assert len(predictions) == 614
    assert len(oof) == 3422
    assert not oof[[f"predicted_{name}" for name in MODEL_NAMES]].isna().any().any()
    assert (results_dir / "manifest.json").is_file()
    assert (plots_dir / "optuna_history.png").is_file()

    prepared = prepare_data(DEFAULT_CONFIG)
    X_sample = prepared.test.loc[:9, list(prepared.feature_columns)]
    for model_name, model_path in result.model_paths.items():
        loaded = load_trained_model(model_name, model_path)
        expected = predictions.loc[:9, f"predicted_{model_name}"].to_numpy()
        assert np.allclose(loaded.predict(X_sample), expected, rtol=1e-5, atol=1e-5)
