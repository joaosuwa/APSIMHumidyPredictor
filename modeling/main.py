"""Orquestra tuning, treino final, persistência, avaliação e gráficos."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:  # Compatibilidade com ``python modeling/main.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from modeling.config import (
    DEFAULT_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DataConfig,
    TrainingConfig,
)
from modeling.data import METADATA_COLUMNS, PreparedData, prepare_data
from modeling.evaluation import persistence_prediction, regression_metrics
from modeling.models import MODEL_NAMES, save_model
from modeling.plots import (
    permutation_feature_importance,
    plot_fold_mae,
    plot_metric_comparison,
    plot_observed_predicted,
    plot_optimization_history,
    plot_residuals,
)
from modeling.training import (
    TuningResult,
    build_training_fingerprint,
    train_from_tuning_result,
    tune_model,
)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Resumo público de uma execução completa do pipeline."""

    champion: str
    cv_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    artifact_root: Path
    model_paths: dict[str, Path]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _metric_row(model_name: str, actual: Any, predicted: Any) -> dict[str, Any]:
    return {"model": model_name, **regression_metrics(actual, predicted)}


def _print_data_summary(prepared: PreparedData) -> None:
    print(
        f"Dados: {len(prepared.development)} linhas de desenvolvimento, "
        f"{len(prepared.test)} linhas de teste, {len(prepared.folds)} folds e "
        f"{len(prepared.feature_columns)} features."
    )


def run_pipeline(
    data_config: DataConfig = DEFAULT_CONFIG,
    training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
) -> PipelineResult:
    """Executa o plano 5; o holdout só é lido após a seleção por OOF."""
    output_dir = training_config.output_dir
    results_dir = output_dir / "results"
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    for directory in (results_dir, models_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    prepared = prepare_data(data_config)
    _print_data_summary(prepared)
    fingerprint, dataset_hash = build_training_fingerprint(
        data_config.dataset_path,
        feature_columns=prepared.feature_columns,
        target_column=prepared.target_column,
        included_simulations=data_config.included_simulations,
        test_cycle_ids=data_config.test_cycle_ids,
        seed=training_config.seed,
        max_iterations=training_config.max_iterations,
        early_stopping_rounds=training_config.early_stopping_rounds,
    )

    tuning_results: dict[str, TuningResult] = {}
    for model_name in MODEL_NAMES:
        print(f"[{model_name}] tuning até {training_config.n_trials} trials...")
        result = tune_model(
            model_name,
            prepared.development,
            prepared.folds,
            prepared.feature_columns,
            prepared.target_column,
            training_config,
            fingerprint,
        )
        tuning_results[model_name] = result
        print(
            f"[{model_name}] MAE OOF={result.best_value:.6f}; "
            f"iterações finais={result.final_iterations}"
        )

    # A escolha é deliberadamente concluída antes de qualquer acesso ao teste.
    champion = min(tuning_results, key=lambda name: tuning_results[name].best_value)
    print(f"Campeão escolhido pela cross-validation: {champion}")

    development = prepared.development
    target_column = prepared.target_column
    metadata = list(METADATA_COLUMNS)
    oof_predictions = development.loc[:, [*metadata, "dr_mm", target_column]].copy()
    fold_tables = []
    cv_rows = []
    for model_name, result in tuning_results.items():
        oof_predictions[f"predicted_{model_name}"] = result.cv.predictions
        fold_tables.append(result.cv.fold_metrics)
        cv_rows.append({"model": model_name, **result.cv.aggregate_metrics})
    baseline_oof = persistence_prediction(development["dr_mm"])
    oof_predictions["predicted_persistence"] = baseline_oof
    cv_rows.append(_metric_row("persistence", development[target_column], baseline_oof))
    cv_metrics = pd.DataFrame(cv_rows).sort_values("mae").reset_index(drop=True)
    fold_metrics = pd.concat(fold_tables, ignore_index=True)
    cv_metrics.to_csv(results_dir / "cv_metrics.csv", index=False)
    fold_metrics.to_csv(results_dir / "cv_fold_metrics.csv", index=False)
    oof_predictions.to_csv(results_dir / "oof_predictions.csv", index=False)

    final_models: dict[str, Any] = {}
    model_paths: dict[str, Path] = {}
    for model_name, result in tuning_results.items():
        model = train_from_tuning_result(
            result,
            development,
            prepared.feature_columns,
            target_column,
            training_config,
        )
        final_models[model_name] = model
        model_paths[model_name] = save_model(model_name, model, models_dir)

    # Fronteira explícita: daqui em diante o holdout do ciclo 6 pode ser avaliado.
    test = prepared.test
    X_test = test.loc[:, list(prepared.feature_columns)]
    y_test = test[target_column]
    test_predictions = test.loc[:, [*metadata, "dr_mm", target_column]].copy()
    test_rows = []
    for model_name, model in final_models.items():
        prediction = np.asarray(model.predict(X_test), dtype=float)
        test_predictions[f"predicted_{model_name}"] = prediction
        test_rows.append(_metric_row(model_name, y_test, prediction))
    baseline_test = persistence_prediction(test["dr_mm"])
    test_predictions["predicted_persistence"] = baseline_test
    test_rows.append(_metric_row("persistence", y_test, baseline_test))
    test_metrics = pd.DataFrame(test_rows).sort_values("mae").reset_index(drop=True)
    test_metrics.to_csv(results_dir / "test_metrics.csv", index=False)
    test_predictions.to_csv(results_dir / "test_predictions.csv", index=False)

    best_parameters = {
        "champion": champion,
        "models": {
            name: {
                "study_name": result.study_name,
                "cv_mae": result.best_value,
                "final_iterations": result.final_iterations,
                "best_iterations_by_fold": list(result.cv.best_iterations),
                "parameters": result.best_params,
            }
            for name, result in tuning_results.items()
        },
    }
    _write_json(results_dir / "best_params.json", best_parameters)

    plot_metric_comparison(
        cv_metrics,
        "Comparação no desenvolvimento (previsões OOF)",
        plots_dir / "cv_model_comparison.png",
        champion,
    )
    plot_fold_mae(fold_metrics, plots_dir / "cv_mae_by_fold.png")
    plot_metric_comparison(
        test_metrics,
        "Comparação no teste final (cycle_id=6)",
        plots_dir / "test_model_comparison.png",
        champion,
    )
    champion_prediction = test_predictions[f"predicted_{champion}"]
    plot_observed_predicted(
        y_test,
        champion_prediction,
        champion,
        plots_dir / "test_observed_vs_predicted.png",
    )
    plot_residuals(
        y_test,
        champion_prediction,
        champion,
        plots_dir / "test_residuals.png",
    )
    plot_optimization_history(
        {name: result.study for name, result in tuning_results.items()},
        plots_dir / "optuna_history.png",
    )
    permutation_feature_importance(
        final_models[champion],
        X_test,
        y_test,
        seed=training_config.seed,
        results_path=results_dir / "permutation_importance.csv",
        plot_path=plots_dir / "permutation_importance.png",
    )

    champion_test_mae = float(
        test_metrics.loc[test_metrics["model"] == champion, "mae"].iloc[0]
    )
    baseline_test_mae = float(
        test_metrics.loc[test_metrics["model"] == "persistence", "mae"].iloc[0]
    )
    package_names = (
        "pandas",
        "scikit-learn",
        "optuna",
        "xgboost",
        "lightgbm",
        "catboost",
        "matplotlib",
    )
    manifest = {
        "fingerprint": fingerprint,
        "dataset_sha256": dataset_hash,
        "dataset_path": str(Path(data_config.dataset_path).resolve()),
        "features": list(prepared.feature_columns),
        "target": target_column,
        "data_config": asdict(data_config),
        "training_config": asdict(training_config),
        "library_versions": {name: version(name) for name in package_names},
        "champion_selected_by_oof_mae": champion,
        "champion_test_mae": champion_test_mae,
        "persistence_test_mae": baseline_test_mae,
        "baseline_outperformed_champion_on_test": baseline_test_mae < champion_test_mae,
        "model_paths": {name: str(path.resolve()) for name, path in model_paths.items()},
        "test_was_not_used_for_selection": True,
    }
    _write_json(results_dir / "manifest.json", manifest)
    print(f"Resultados gravados em: {output_dir.resolve()}")
    return PipelineResult(champion, cv_metrics, test_metrics, output_dir, model_paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.n_trials,
        help="Total de trials Optuna por algoritmo; trials retomados contam no total.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_CONFIG.seed)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_TRAINING_CONFIG.output_dir
    )
    return parser


def main(argv: Sequence[str] | None = None) -> PipelineResult:
    args = build_parser().parse_args(argv)
    config = replace(
        DEFAULT_TRAINING_CONFIG,
        n_trials=args.trials,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    return run_pipeline(DEFAULT_CONFIG, config)


if __name__ == "__main__":
    main()
