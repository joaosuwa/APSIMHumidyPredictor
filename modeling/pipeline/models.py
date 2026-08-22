"""Fábricas, ajuste e persistência dos regressores de árvores."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import lightgbm as lgb
import optuna
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


ModelName = Literal["xgboost", "lightgbm", "catboost"]
ObjectiveMetric = Literal["mae", "rmse"]
MODEL_NAMES: tuple[ModelName, ...] = ("xgboost", "lightgbm", "catboost")
SEARCH_SPACE_VERSION = "2026-08-plan7-weather-rmse-v1"


def suggest_parameters(model_name: ModelName, trial: optuna.Trial) -> dict[str, Any]:
    common = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True)
    }
    if model_name == "xgboost":
        return {
            **common,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
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
            "random_strength": trial.suggest_float(
                "random_strength", 1e-3, 10.0, log=True
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.0, 10.0
            ),
        }
    raise ValueError(f"Modelo desconhecido: {model_name}")


def build_model(
    model_name: ModelName,
    parameters: dict[str, Any],
    *,
    seed: int,
    iterations: int,
    early_stopping_rounds: int | None,
    objective_metric: ObjectiveMetric,
) -> Any:
    if objective_metric not in {"mae", "rmse"}:
        raise ValueError(f"Métrica de objetivo desconhecida: {objective_metric}")
    if model_name == "xgboost":
        return XGBRegressor(
            **parameters,
            n_estimators=iterations,
            objective=(
                "reg:absoluteerror"
                if objective_metric == "mae"
                else "reg:squarederror"
            ),
            eval_metric=objective_metric,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            early_stopping_rounds=early_stopping_rounds,
            verbosity=0,
        )
    if model_name == "lightgbm":
        return LGBMRegressor(
            **parameters,
            n_estimators=iterations,
            objective="l1" if objective_metric == "mae" else "regression_l2",
            subsample_freq=1,
            n_jobs=-1,
            random_state=seed,
            verbosity=-1,
        )
    if model_name == "catboost":
        return CatBoostRegressor(
            **parameters,
            iterations=iterations,
            loss_function="MAE" if objective_metric == "mae" else "RMSE",
            eval_metric="MAE" if objective_metric == "mae" else "RMSE",
            bootstrap_type="Bayesian",
            random_seed=seed,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"Modelo desconhecido: {model_name}")


def fit_with_validation(
    model_name: ModelName,
    model: Any,
    X_train: Any,
    y_train: Any,
    X_validation: Any,
    y_validation: Any,
    *,
    early_stopping_rounds: int,
    objective_metric: ObjectiveMetric,
) -> Any:
    if model_name == "xgboost":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_validation, y_validation)],
            verbose=False,
        )
    elif model_name == "lightgbm":
        model.fit(
            X_train,
            y_train,
            eval_X=X_validation,
            eval_y=y_validation,
            eval_metric=objective_metric,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    elif model_name == "catboost":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_validation, y_validation),
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True,
            verbose=False,
        )
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    return model


def best_iteration_count(model_name: ModelName, model: Any) -> int:
    if model_name == "xgboost":
        value = getattr(model, "best_iteration", None)
        return int(value) + 1 if value is not None else int(model.n_estimators)
    if model_name == "lightgbm":
        value = getattr(model, "best_iteration_", None)
        return int(value) if value else int(model.n_estimators)
    if model_name == "catboost":
        value = model.get_best_iteration()
        return int(value) + 1 if value is not None and value >= 0 else int(
            model.get_param("iterations")
        )
    raise ValueError(f"Modelo desconhecido: {model_name}")


def fit_final_model(
    model_name: ModelName,
    parameters: dict[str, Any],
    X: Any,
    y: Any,
    *,
    seed: int,
    iterations: int,
    objective_metric: ObjectiveMetric,
) -> Any:
    model = build_model(
        model_name,
        parameters,
        seed=seed,
        iterations=iterations,
        early_stopping_rounds=None,
        objective_metric=objective_metric,
    )
    model.fit(X, y)
    return model


def save_model(model_name: ModelName, model: Any, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    suffix = {"xgboost": ".ubj", "lightgbm": ".txt", "catboost": ".cbm"}[
        model_name
    ]
    path = models_dir / f"{model_name}{suffix}"
    if model_name == "xgboost":
        model.save_model(path)
    elif model_name == "lightgbm":
        model.booster_.save_model(str(path))
    elif model_name == "catboost":
        model.save_model(str(path))
    return path


def load_trained_model(model_name: ModelName, path: str | Path) -> Any:
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if model_name == "xgboost":
        model = XGBRegressor()
        model.load_model(model_path)
        return model
    if model_name == "lightgbm":
        return lgb.Booster(model_file=str(model_path))
    if model_name == "catboost":
        model = CatBoostRegressor()
        model.load_model(str(model_path))
        return model
    raise ValueError(f"Modelo desconhecido: {cast(str, model_name)}")
