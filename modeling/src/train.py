"""Executa tuning, treino final, teste temporal e geração de artefatos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.model_dataset import METADATA_COLUMNS, TARGET_COLUMN, VARIATION_TARGET_COLUMN

from .data import feature_target, load_dataset, make_temporal_splits, split_summary
from .evaluate import evaluate_prediction, persistence_baseline
from .optimize import MODEL_NAMES, optimize_model, train_final_model
from .plots import (
    permutation_feature_importance,
    plot_model_comparison,
    plot_observed_predicted,
    plot_residuals,
    shap_feature_importance,
)


MODELING_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = MODELING_ROOT / "results"
MODELS_DIR = MODELING_ROOT / "models"
PLOTS_DIR = MODELING_ROOT / "plots"


def _summary_text(metrics: pd.DataFrame, splits_summary: pd.DataFrame, selected: str) -> str:
    lines = [
        "Treinamento do previsor de déficit hídrico",
        "==========================================",
        "",
        "Target: variacao_deficit_proximo_dia_mm",
        f"Modelo selecionado pela validação: {selected}",
        "Critério de seleção: menor MAE geral da variação nos ciclos de 2024",
        "",
        "Cortes temporais:",
        splits_summary.to_string(index=False),
        "",
        "Métricas no teste temporal (safra 2025):",
        metrics.to_string(index=False),
        "",
        "A baseline persistence prevê variação zero: Dr(D+1) = Dr(D).",
    ]
    return "\n".join(lines) + "\n"


def run_pipeline(n_trials: int = 50, seed: int = 42) -> dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    splits = make_temporal_splits(dataset)
    splits_table = split_summary(splits)
    splits_table.to_csv(RESULTS_DIR / "split_summary.csv", index=False)

    X_tuning, y_tuning = feature_target(splits.tuning_train)
    X_validation, y_validation = feature_target(splits.validation)
    X_final, y_final = feature_target(splits.final_train)
    X_test, y_test = feature_target(splits.test)

    optimization_results: dict[str, dict] = {}
    models = {}
    for model_name in MODEL_NAMES:
        print(f"[{model_name}] otimizando {n_trials} trials...")
        _, result = optimize_model(
            model_name,
            X_tuning,
            y_tuning,
            X_validation,
            y_validation,
            RESULTS_DIR,
            n_trials=n_trials,
            seed=seed,
        )
        optimization_results[model_name] = result
        print(
            f"[{model_name}] MAE validação={result['validation_mae']:.6f}; "
            f"árvores={result['best_iteration']}"
        )
        models[model_name] = train_final_model(
            model_name,
            result,
            X_final,
            y_final,
            MODELS_DIR,
            seed=seed,
        )

    selected = min(
        optimization_results,
        key=lambda name: optimization_results[name]["validation_mae"],
    )
    (RESULTS_DIR / "best_params.json").write_text(
        json.dumps(
            {"selected_model": selected, "models": optimization_results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    prediction_columns = [
        *METADATA_COLUMNS,
        "dr_mm",
        TARGET_COLUMN,
        VARIATION_TARGET_COLUMN,
        "irrigacao_aplicada_dia_posterior_mm",
    ]
    predictions = splits.test[prediction_columns].reset_index(drop=True).copy()
    metric_rows = []
    model_predictions = {}
    for model_name, model in models.items():
        predicted_variation = model.predict(X_test)
        predicted_deficit = splits.test["dr_mm"].to_numpy() + predicted_variation
        predictions[f"variacao_prevista_{model_name}"] = predicted_variation
        predictions[f"deficit_previsto_{model_name}"] = predicted_deficit
        model_predictions[model_name] = (predicted_variation, predicted_deficit)
        metric_rows.append(
            evaluate_prediction(
                model_name,
                y_test,
                predicted_variation,
                splits.test["dr_mm"],
                splits.test[TARGET_COLUMN],
                validation_mae=optimization_results[model_name]["validation_mae"],
                selected=model_name == selected,
            )
        )

    baseline_prediction = persistence_baseline(splits.test)
    predictions["variacao_prevista_persistence"] = baseline_prediction
    predictions["deficit_previsto_persistence"] = splits.test["dr_mm"].to_numpy()
    metric_rows.append(
        evaluate_prediction(
            "persistence",
            y_test,
            baseline_prediction,
            splits.test["dr_mm"],
            splits.test[TARGET_COLUMN],
        )
    )
    metrics = pd.DataFrame(metric_rows).sort_values("mae_variation").reset_index(drop=True)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    (RESULTS_DIR / "summary.txt").write_text(
        _summary_text(metrics, splits_table, selected),
        encoding="utf-8",
    )
    (RESULTS_DIR / "selected_model.txt").write_text(selected + "\n", encoding="utf-8")

    selected_model = models[selected]
    selected_variation, selected_deficit = model_predictions[selected]
    plot_model_comparison(metrics, PLOTS_DIR)
    plot_observed_predicted(splits.test[TARGET_COLUMN], selected_deficit, PLOTS_DIR)
    plot_residuals(selected_deficit, splits.test[TARGET_COLUMN], PLOTS_DIR)
    permutation_feature_importance(
        selected_model,
        X_test,
        y_test,
        RESULTS_DIR,
        PLOTS_DIR,
        seed=seed,
    )
    shap_feature_importance(
        selected_model,
        X_test,
        RESULTS_DIR,
        PLOTS_DIR,
        seed=seed,
    )
    print(f"Modelo selecionado: {selected}")
    print(f"Resultados: {RESULTS_DIR}")
    return {
        "selected_model": selected,
        "metrics": metrics,
        "split_summary": splits_table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=50, help="Trials por modelo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(n_trials=args.trials, seed=args.seed)


if __name__ == "__main__":
    main()
