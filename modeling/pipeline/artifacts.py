"""Persistência dos parâmetros e metadados de uma execução."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .optimization import TuningResult


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def write_parameter_artifacts(
    parameters_dir: Path,
    tuning_results: Mapping[str, TuningResult],
    *,
    champion: str,
    target_column: str,
    objective_metric: str,
) -> dict[str, Path]:
    """Grava um arquivo autocontido por modelo e um índice do campeão."""
    paths: dict[str, Path] = {}
    for model_name, result in tuning_results.items():
        path = parameters_dir / f"{model_name}.json"
        paths[model_name] = write_json(
            path,
            {
                "model": model_name,
                "target": target_column,
                "objective_metric": objective_metric,
                "study_name": result.study_name,
                "oof_metrics": result.cv.aggregate_metrics,
                "final_iterations": result.final_iterations,
                "best_iterations_by_fold": list(result.cv.best_iterations),
                "parameters": result.best_params,
            },
        )
    paths["champion"] = write_json(
        parameters_dir / "champion.json",
        {
            "champion": champion,
            "selection_metric": f"{objective_metric}_variation_oof",
            "target": target_column,
            "parameter_files": {
                name: str(path.resolve())
                for name, path in paths.items()
                if name != "champion"
            },
        },
    )
    return paths
