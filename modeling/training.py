"""Cross-validation agrupada, estudos Optuna e treino final."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
import pandas as pd

from .config import TrainingConfig
from .data import CycleFold
from .evaluation import regression_metrics
from .models import (
    SEARCH_SPACE_VERSION,
    ModelName,
    best_iteration_count,
    build_model,
    fit_final_model,
    fit_with_validation,
    suggest_parameters,
)


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    """Previsões OOF, métricas por fold e contagens ótimas de árvores."""

    predictions: pd.Series
    fold_metrics: pd.DataFrame
    aggregate_metrics: dict[str, float]
    best_iterations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TuningResult:
    """Resultado completo do estudo vencedor de um algoritmo."""

    model_name: ModelName
    study_name: str
    best_params: dict[str, Any]
    best_value: float
    final_iterations: int
    cv: CrossValidationResult
    study: optuna.Study


def build_training_fingerprint(
    dataset_path: str | Path,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    included_simulations: Sequence[str],
    test_cycle_ids: Sequence[int],
    seed: int,
    max_iterations: int,
    early_stopping_rounds: int,
) -> tuple[str, str]:
    """Identifica de forma estável os dados e as decisões que afetam um estudo."""
    path = Path(dataset_path)
    dataset_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "dataset_sha256": dataset_hash,
        "features": list(feature_columns),
        "target": target_column,
        "included_simulations": list(included_simulations),
        "test_cycle_ids": list(test_cycle_ids),
        "seed": seed,
        "max_iterations": max_iterations,
        "early_stopping_rounds": early_stopping_rounds,
        "search_space_version": SEARCH_SPACE_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), dataset_hash


def cross_validate_parameters(
    model_name: ModelName,
    parameters: Mapping[str, Any],
    development: pd.DataFrame,
    folds: Sequence[CycleFold],
    feature_columns: Sequence[str],
    target_column: str,
    config: TrainingConfig,
    *,
    trial: optuna.Trial | None = None,
) -> CrossValidationResult:
    """Avalia parâmetros sem receber ou acessar o holdout final."""
    if not folds:
        raise ValueError("A cross-validation requer pelo menos um fold")
    X = development.loc[:, list(feature_columns)]
    y = development.loc[:, target_column]
    predictions = pd.Series(np.nan, index=development.index, dtype=float, name=model_name)
    records: list[dict[str, float | int | str]] = []
    iterations: list[int] = []
    absolute_error_sum = 0.0
    evaluated_rows = 0

    for step, fold in enumerate(folds):
        model = build_model(
            model_name,
            dict(parameters),
            seed=config.seed,
            iterations=config.max_iterations,
            early_stopping_rounds=config.early_stopping_rounds,
        )
        fit_with_validation(
            model_name,
            model,
            X.loc[fold.train_indices],
            y.loc[fold.train_indices],
            X.loc[fold.validation_indices],
            y.loc[fold.validation_indices],
            early_stopping_rounds=config.early_stopping_rounds,
        )
        fold_prediction = np.asarray(
            model.predict(X.loc[fold.validation_indices]), dtype=float
        )
        predictions.loc[fold.validation_indices] = fold_prediction
        metrics = regression_metrics(y.loc[fold.validation_indices], fold_prediction)
        iteration_count = best_iteration_count(model_name, model)
        iterations.append(iteration_count)
        records.append(
            {
                "model": model_name,
                "validation_cycle_id": fold.validation_cycle_id,
                "rows": len(fold.validation_indices),
                "best_iterations": iteration_count,
                **metrics,
            }
        )

        absolute_error_sum += float(
            np.abs(y.loc[fold.validation_indices].to_numpy(float) - fold_prediction).sum()
        )
        evaluated_rows += len(fold.validation_indices)
        if trial is not None:
            trial.report(absolute_error_sum / evaluated_rows, step=step)
            if trial.should_prune():
                trial.set_user_attr("best_iterations_partial", iterations)
                raise optuna.TrialPruned()

    if predictions.isna().any():
        missing = predictions.index[predictions.isna()].tolist()
        raise ValueError(f"Cross-validation não cobriu todas as linhas: {missing[:10]}")
    aggregate = regression_metrics(y, predictions)
    return CrossValidationResult(
        predictions=predictions,
        fold_metrics=pd.DataFrame.from_records(records),
        aggregate_metrics=aggregate,
        best_iterations=tuple(iterations),
    )


def tune_model(
    model_name: ModelName,
    development: pd.DataFrame,
    folds: Sequence[CycleFold],
    feature_columns: Sequence[str],
    target_column: str,
    config: TrainingConfig,
    fingerprint: str,
) -> TuningResult:
    """Cria ou retoma o estudo e reconstrói as previsões do melhor trial."""
    studies_dir = config.output_dir / "studies"
    results_dir = config.output_dir / "results"
    studies_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    database_path = (studies_dir / "optuna.sqlite3").resolve().as_posix()
    study_name = f"water_deficit_{model_name}_{fingerprint[:16]}"
    sampler = optuna.samplers.TPESampler(seed=config.seed, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{database_path}",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_parameters(model_name, trial)
        cv = cross_validate_parameters(
            model_name,
            parameters,
            development,
            folds,
            feature_columns,
            target_column,
            config,
            trial=trial,
        )
        trial.set_user_attr("best_iterations", list(cv.best_iterations))
        return cv.aggregate_metrics["mae"]

    terminal_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    terminal_trials = sum(trial.state in terminal_states for trial in study.trials)
    remaining = max(0, config.n_trials - terminal_trials)
    if remaining:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=remaining, gc_after_trial=True, n_jobs=1)
    if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        raise RuntimeError(f"O estudo {study_name} não possui trials completos")

    study.trials_dataframe().to_csv(
        results_dir / f"optuna_trials_{model_name}.csv", index=False
    )
    best_params = dict(study.best_params)
    cv = cross_validate_parameters(
        model_name,
        best_params,
        development,
        folds,
        feature_columns,
        target_column,
        config,
    )
    final_iterations = max(1, int(round(median(cv.best_iterations))))
    return TuningResult(
        model_name=model_name,
        study_name=study_name,
        best_params=best_params,
        best_value=float(cv.aggregate_metrics["mae"]),
        final_iterations=final_iterations,
        cv=cv,
        study=study,
    )


def train_from_tuning_result(
    result: TuningResult,
    development: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    config: TrainingConfig,
) -> Any:
    """Refaz o ajuste vencedor em todas as linhas de desenvolvimento."""
    return fit_final_model(
        result.model_name,
        result.best_params,
        development.loc[:, list(feature_columns)],
        development.loc[:, target_column],
        seed=config.seed,
        iterations=result.final_iterations,
    )
