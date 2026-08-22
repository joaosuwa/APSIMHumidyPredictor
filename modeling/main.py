"""Orquestra tuning residual, treino, persistência, avaliação e gráficos."""

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
from modeling.data import (
    METADATA_COLUMNS,
    NEXT_DAY_OBSERVED_RAIN_COLUMN,
    NEXT_DEFICIT_COLUMN,
    PreparedData,
    VARIATION_TARGET_COLUMN,
    prepare_data,
)
from modeling.pipeline.artifacts import write_json, write_parameter_artifacts
from modeling.pipeline.evaluation import (
    evaluate_variation_prediction,
    event_metrics,
    rain_error_metrics,
    zero_variation_prediction,
)
from modeling.pipeline.models import MODEL_NAMES, save_model
from modeling.pipeline.optimization import (
    TuningResult,
    build_training_fingerprint,
    train_from_tuning_result,
    tune_model,
)
from modeling.pipeline.plots import (
    permutation_feature_importance,
    plot_feature_importance_comparison,
    plot_fold_metric,
    plot_metric_comparison,
    plot_observed_predicted,
    plot_optimization_history,
    plot_residuals,
    plot_residuals_by_observed_rain,
    plot_rain_rmse,
)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    champion: str
    cv_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    artifact_root: Path
    model_paths: dict[str, Path]
    parameter_paths: dict[str, Path]


def _metric_row(
    model_name: str,
    frame: pd.DataFrame,
    predicted_variation: Any,
) -> dict[str, Any]:
    return {
        "model": model_name,
        **evaluate_variation_prediction(
            frame[VARIATION_TARGET_COLUMN],
            predicted_variation,
            frame["dr_mm"],
            frame[NEXT_DEFICIT_COLUMN],
        ),
    }


def _add_prediction_columns(
    destination: pd.DataFrame,
    model_name: str,
    predicted_variation: Any,
    current_deficit: pd.Series,
) -> np.ndarray:
    variation = np.asarray(predicted_variation, dtype=float)
    destination[f"predicted_variation_{model_name}"] = variation
    destination[f"predicted_next_deficit_{model_name}"] = (
        current_deficit.to_numpy(float) + variation
    )
    return variation


def _historical_comparison(
    output_path: Path,
    new_metrics: pd.DataFrame,
    new_champion: str,
) -> None:
    """Compara abordagens somente depois que a nova seleção já terminou."""
    absolute_root = output_path.parents[2] / "default" / "results"
    old_metrics_path = absolute_root / "test_metrics.csv"
    old_manifest_path = absolute_root / "manifest.json"
    if not (old_metrics_path.is_file() and old_manifest_path.is_file()):
        return
    old_metrics = pd.read_csv(old_metrics_path)
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_champion = old_manifest["champion_selected_by_oof_mae"]
    records = [
        {
            "approach": "absolute_target",
            "model": row.model,
            "selected_by_oof": row.model == old_champion,
            "mae_next_deficit": row.mae,
            "rmse_next_deficit": row.rmse,
            "r2_next_deficit": row.r2,
            "bias_next_deficit": row.bias,
        }
        for row in old_metrics.itertuples(index=False)
    ]
    records.extend(
        {
            "approach": "variation_target",
            "model": row.model,
            "selected_by_oof": row.model == new_champion,
            "mae_next_deficit": row.mae_next_deficit,
            "rmse_next_deficit": row.rmse_next_deficit,
            "r2_next_deficit": row.r2_next_deficit,
            "bias_next_deficit": row.bias_next_deficit,
        }
        for row in new_metrics.itertuples(index=False)
    )
    pd.DataFrame.from_records(records).to_csv(output_path, index=False)


def run_pipeline(
    data_config: DataConfig = DEFAULT_CONFIG,
    training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
) -> PipelineResult:
    output_dir = training_config.output_dir
    results_dir = output_dir / "results"
    models_dir = output_dir / "models"
    parameters_dir = output_dir / "parameters"
    plots_dir = output_dir / "plots"
    for directory in (results_dir, models_dir, parameters_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    prepared = prepare_data(data_config)
    print(
        f"Dados: {len(prepared.development)} desenvolvimento, "
        f"{len(prepared.test)} teste, {len(prepared.folds)} folds, "
        f"target={prepared.target_column}."
    )
    fingerprint, dataset_hash = build_training_fingerprint(
        data_config.dataset_path,
        feature_columns=prepared.feature_columns,
        target_column=prepared.target_column,
        included_simulations=data_config.included_simulations,
        test_cycle_ids=data_config.test_cycle_ids,
        seed=training_config.seed,
        max_iterations=training_config.max_iterations,
        early_stopping_rounds=training_config.early_stopping_rounds,
        objective_metric=training_config.objective_metric,
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
            f"[{model_name}] {training_config.objective_metric.upper()} OOF "
            f"da variação={result.best_value:.6f}; "
            f"iterações finais={result.final_iterations}"
        )

    champion = min(tuning_results, key=lambda name: tuning_results[name].best_value)
    print(
        "Campeão escolhido pela variação OOF "
        f"({training_config.objective_metric.upper()}): {champion}"
    )
    parameter_paths = write_parameter_artifacts(
        parameters_dir,
        tuning_results,
        champion=champion,
        target_column=prepared.target_column,
        objective_metric=training_config.objective_metric,
    )

    development = prepared.development
    prediction_columns = [
        *METADATA_COLUMNS,
        "dr_mm",
        NEXT_DEFICIT_COLUMN,
        VARIATION_TARGET_COLUMN,
        NEXT_DAY_OBSERVED_RAIN_COLUMN,
    ]
    oof_predictions = development.loc[:, prediction_columns].copy()
    fold_tables = []
    cv_rows = []
    cv_event_tables = []
    cv_rain_tables = []
    for model_name, result in tuning_results.items():
        prediction = _add_prediction_columns(
            oof_predictions,
            model_name,
            result.cv.predictions,
            development["dr_mm"],
        )
        fold_tables.append(result.cv.fold_metrics)
        cv_rows.append(_metric_row(model_name, development, prediction))
        cv_event_tables.append(
            event_metrics(model_name, development[VARIATION_TARGET_COLUMN], prediction)
        )
        cv_rain_tables.append(
            rain_error_metrics(
                model_name,
                development[NEXT_DAY_OBSERVED_RAIN_COLUMN],
                development[VARIATION_TARGET_COLUMN],
                prediction,
            ).assign(split="development_oof")
        )
    baseline_oof = _add_prediction_columns(
        oof_predictions,
        "zero_variation",
        zero_variation_prediction(len(development)),
        development["dr_mm"],
    )
    cv_rows.append(_metric_row("zero_variation", development, baseline_oof))
    cv_event_tables.append(
        event_metrics(
            "zero_variation", development[VARIATION_TARGET_COLUMN], baseline_oof
        )
    )
    cv_rain_tables.append(
        rain_error_metrics(
            "zero_variation",
            development[NEXT_DAY_OBSERVED_RAIN_COLUMN],
            development[VARIATION_TARGET_COLUMN],
            baseline_oof,
        ).assign(split="development_oof")
    )
    selection_column = f"{training_config.objective_metric}_variation"
    cv_metrics = (
        pd.DataFrame(cv_rows).sort_values(selection_column).reset_index(drop=True)
    )
    fold_metrics = pd.concat(fold_tables, ignore_index=True)
    cv_events = pd.concat(cv_event_tables, ignore_index=True)
    cv_metrics.to_csv(results_dir / "cv_metrics.csv", index=False)
    fold_metrics.to_csv(results_dir / "cv_fold_metrics.csv", index=False)
    cv_events.to_csv(results_dir / "cv_event_metrics.csv", index=False)
    oof_predictions.to_csv(results_dir / "oof_predictions.csv", index=False)

    final_models: dict[str, Any] = {}
    model_paths: dict[str, Path] = {}
    for model_name, result in tuning_results.items():
        model = train_from_tuning_result(
            result,
            development,
            prepared.feature_columns,
            prepared.target_column,
            training_config,
        )
        final_models[model_name] = model
        model_paths[model_name] = save_model(model_name, model, models_dir)

    # O holdout só é acessado após seleção, parâmetros e treino final.
    test = prepared.test
    X_test = test.loc[:, list(prepared.feature_columns)]
    test_predictions = test.loc[:, prediction_columns].copy()
    test_rows = []
    test_event_tables = []
    test_rain_tables = []
    test_model_predictions: dict[str, np.ndarray] = {}
    for model_name, model in final_models.items():
        prediction = _add_prediction_columns(
            test_predictions, model_name, model.predict(X_test), test["dr_mm"]
        )
        test_rows.append(_metric_row(model_name, test, prediction))
        test_event_tables.append(
            event_metrics(model_name, test[VARIATION_TARGET_COLUMN], prediction)
        )
        test_model_predictions[model_name] = prediction
        test_rain_tables.append(
            rain_error_metrics(
                model_name,
                test[NEXT_DAY_OBSERVED_RAIN_COLUMN],
                test[VARIATION_TARGET_COLUMN],
                prediction,
            ).assign(split="test")
        )
    baseline_test = _add_prediction_columns(
        test_predictions,
        "zero_variation",
        zero_variation_prediction(len(test)),
        test["dr_mm"],
    )
    test_rows.append(_metric_row("zero_variation", test, baseline_test))
    test_event_tables.append(
        event_metrics("zero_variation", test[VARIATION_TARGET_COLUMN], baseline_test)
    )
    test_rain_tables.append(
        rain_error_metrics(
            "zero_variation",
            test[NEXT_DAY_OBSERVED_RAIN_COLUMN],
            test[VARIATION_TARGET_COLUMN],
            baseline_test,
        ).assign(split="test")
    )
    test_metrics = (
        pd.DataFrame(test_rows).sort_values(selection_column).reset_index(drop=True)
    )
    test_events = pd.concat(test_event_tables, ignore_index=True)
    rain_metrics = pd.concat([*cv_rain_tables, *test_rain_tables], ignore_index=True)
    test_metrics.to_csv(results_dir / "test_metrics.csv", index=False)
    test_events.to_csv(results_dir / "test_event_metrics.csv", index=False)
    rain_metrics.to_csv(results_dir / "rain_error_metrics.csv", index=False)
    test_predictions.to_csv(results_dir / "test_predictions.csv", index=False)
    _historical_comparison(
        results_dir / "historical_comparison.csv", test_metrics, champion
    )

    plot_metric_comparison(
        cv_metrics,
        "Comparação OOF — target de variação",
        plots_dir / "cv_model_comparison.png",
        champion,
        metric_column=selection_column,
        metric_label=f"{training_config.objective_metric.upper()} da variação (mm)",
    )
    plot_fold_metric(
        fold_metrics,
        plots_dir / f"cv_{training_config.objective_metric}_by_fold.png",
        metric=training_config.objective_metric,
    )
    plot_metric_comparison(
        test_metrics,
        "Comparação no teste — target de variação",
        plots_dir / "test_model_comparison.png",
        champion,
        metric_column=selection_column,
        metric_label=f"{training_config.objective_metric.upper()} da variação (mm)",
    )
    champion_variation = test_predictions[f"predicted_variation_{champion}"]
    champion_next = test_predictions[f"predicted_next_deficit_{champion}"]
    plot_observed_predicted(
        test[VARIATION_TARGET_COLUMN],
        champion_variation,
        champion,
        plots_dir / "test_variation_observed_vs_predicted.png",
        value_label="Variação do déficit",
    )
    plot_observed_predicted(
        test[NEXT_DEFICIT_COLUMN],
        champion_next,
        champion,
        plots_dir / "test_next_deficit_observed_vs_predicted.png",
        value_label="Déficit em D+1",
    )
    plot_residuals(
        test[VARIATION_TARGET_COLUMN],
        champion_variation,
        champion,
        plots_dir / "test_variation_residuals.png",
    )
    plot_optimization_history(
        {name: result.study for name, result in tuning_results.items()},
        plots_dir / "optuna_history.png",
        metric=training_config.objective_metric,
    )
    test_rain_metrics = rain_metrics.loc[
        rain_metrics["split"].eq("test") & rain_metrics["model"].isin(MODEL_NAMES)
    ]
    plot_rain_rmse(test_rain_metrics, plots_dir / "test_rmse_by_observed_rain.png")
    plot_residuals_by_observed_rain(
        test[NEXT_DAY_OBSERVED_RAIN_COLUMN],
        test[VARIATION_TARGET_COLUMN],
        test_model_predictions,
        plots_dir / "test_residuals_by_observed_rain.png",
    )

    importance_tables: dict[str, pd.DataFrame] = {}
    for model_name, model in final_models.items():
        importance_tables[model_name] = permutation_feature_importance(
            model,
            X_test,
            test[VARIATION_TARGET_COLUMN],
            seed=training_config.seed,
            model_name=model_name,
            metric=training_config.objective_metric,
            results_path=results_dir / f"permutation_importance_{model_name}.csv",
            plot_path=plots_dir / f"permutation_importance_{model_name}.png",
        )
    plot_feature_importance_comparison(
        importance_tables,
        plots_dir / "permutation_importance_comparison.png",
    )

    champion_test_objective = float(
        test_metrics.loc[test_metrics["model"] == champion, selection_column].iloc[0]
    )
    baseline_test_objective = float(
        test_metrics.loc[
            test_metrics["model"] == "zero_variation", selection_column
        ].iloc[0]
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
        "target": prepared.target_column,
        "next_deficit_column": NEXT_DEFICIT_COLUMN,
        "reconstruction": "predicted_next_deficit = dr_mm + predicted_variation",
        "data_config": asdict(data_config),
        "training_config": asdict(training_config),
        "library_versions": {name: version(name) for name in package_names},
        "objective_metric": training_config.objective_metric,
        "champion_selected_by_oof": champion,
        "champion_selection_metric": selection_column,
        "champion_test_objective_value": champion_test_objective,
        "zero_variation_test_objective_value": baseline_test_objective,
        "baseline_outperformed_champion_on_test": baseline_test_objective
        < champion_test_objective,
        "model_paths": {name: str(path.resolve()) for name, path in model_paths.items()},
        "parameter_paths": {
            name: str(path.resolve()) for name, path in parameter_paths.items()
        },
        "test_was_not_used_for_selection": True,
    }
    write_json(results_dir / "manifest.json", manifest)
    print(f"Resultados gravados em: {output_dir.resolve()}")
    return PipelineResult(
        champion,
        cv_metrics,
        test_metrics,
        output_dir,
        model_paths,
        parameter_paths,
    )


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
        "--metric",
        choices=("mae", "rmse"),
        default=DEFAULT_TRAINING_CONFIG.objective_metric,
        help="Loss, métrica do early stopping, objetivo Optuna e seleção do campeão.",
    )
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
        objective_metric=args.metric,
        output_dir=args.output_dir,
    )
    return run_pipeline(DEFAULT_CONFIG, config)


if __name__ == "__main__":
    main()
