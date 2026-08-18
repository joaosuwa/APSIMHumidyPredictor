"""Otimização Optuna para XGBoost, LightGBM e CatBoost."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


MODEL_NAMES = ("xgboost", "lightgbm", "catboost")


def suggest_parameters(model_name: str, trial: optuna.Trial) -> dict[str, Any]:
    """Define espaços de busca compactos e comparáveis."""
    common = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
    }
    if model_name == "xgboost":
        return {
            **common,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        }
    if model_name == "lightgbm":
        return {
            **common,
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        }
    if model_name == "catboost":
        return {
            **common,
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        }
    raise ValueError(f"Modelo desconhecido: {model_name}")


def build_model(
    model_name: str,
    parameters: dict[str, Any],
    seed: int,
    iterations: int = 3000,
    early_stopping: bool = True,
):
    """Instancia um regressor com configuração reproduzível."""
    if model_name == "xgboost":
        return XGBRegressor(
            **parameters,
            n_estimators=iterations,
            objective="reg:squarederror",
            eval_metric="mae",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            early_stopping_rounds=100 if early_stopping else None,
        )
    if model_name == "lightgbm":
        return lgb.LGBMRegressor(
            **parameters,
            n_estimators=iterations,
            objective="regression_l1",
            verbosity=-1,
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "catboost":
        return CatBoostRegressor(
            **parameters,
            iterations=iterations,
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"Modelo desconhecido: {model_name}")


def fit_with_validation(model_name, model, X_train, y_train, X_valid, y_valid):
    """Treina com early stopping usando a API de cada biblioteca."""
    if model_name == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    elif model_name == "lightgbm":
        model.fit(
            X_train,
            y_train,
            eval_X=X_valid,
            eval_y=y_valid,
            eval_metric="mae",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
    else:
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100)
    return model


def best_iteration(model_name: str, model) -> int:
    """Normaliza a melhor quantidade de árvores para o treino final."""
    if model_name == "xgboost":
        value = getattr(model, "best_iteration", None)
        return int(value + 1) if value is not None else int(model.n_estimators)
    if model_name == "lightgbm":
        value = getattr(model, "best_iteration_", None)
        return int(value) if value else int(model.n_estimators)
    value = model.get_best_iteration()
    return int(value + 1) if value is not None and value >= 0 else int(model.get_params()["iterations"])


def optimize_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    results_dir: Path,
    n_trials: int = 50,
    seed: int = 42,
) -> tuple[optuna.Study, dict[str, Any]]:
    """Executa ou retoma um estudo TPE e salva o histórico."""
    results_dir.mkdir(parents=True, exist_ok=True)
    storage_path = (results_dir / "optuna_studies.db").resolve().as_posix()
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
    study = optuna.create_study(
        study_name=f"water_deficit_{model_name}",
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_parameters(model_name, trial)
        model = build_model(model_name, parameters, seed=seed)
        fit_with_validation(model_name, model, X_train, y_train, X_valid, y_valid)
        prediction = model.predict(X_valid)
        mae = float(mean_absolute_error(y_valid, prediction))
        trial.set_user_attr("best_iteration", best_iteration(model_name, model))
        return mae

    completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    remaining = max(0, n_trials - completed)
    if remaining:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    trials_path = results_dir / f"optuna_trials_{model_name}.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)
    result = {
        "model": model_name,
        "validation_mae": float(study.best_value),
        "best_iteration": int(study.best_trial.user_attrs["best_iteration"]),
        "parameters": study.best_params,
    }
    (results_dir / f"best_params_{model_name}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return study, result


def train_final_model(
    model_name: str,
    optimization_result: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models_dir: Path,
    seed: int = 42,
):
    """Refaz o treino com todos os dados 2019-2024 e salva o modelo."""
    model = build_model(
        model_name,
        optimization_result["parameters"],
        seed=seed,
        iterations=optimization_result["best_iteration"],
        early_stopping=False,
    )
    model.fit(X_train, y_train)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / f"{model_name}.joblib")
    return model
