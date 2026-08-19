"""Ponto de entrada compatível para engenharia de features.

As implementações específicas ficam em ``scripts.apsim`` e
``scripts.nasa_power``. Os nomes abaixo continuam disponíveis para preservar
os imports existentes no projeto e nos scripts antigos.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:  # Compatibilidade com ``python scripts/feature_engineering.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.apsim.processing import DEFAULT_OUTPUT, DEFAULT_REPORT, build_report_features
else:
    from .apsim.processing import (
        DEFAULT_OUTPUT,
        DEFAULT_REPORT,
        build_report_features,
    )


def main(
    report_path=DEFAULT_REPORT,
    output_path=DEFAULT_OUTPUT,
    columns_to_drop: list[str] | None = None,
):
    """Executa o processamento padrão do relatório do APSIM NG."""
    result = build_report_features(
        report_path=report_path,
        output_path=output_path,
        columns_to_drop=columns_to_drop,
    )
    print(f"linhas na janela de cultivo: {len(result)}")
    print(f"simulacoes: {sorted(result['SimulationName'].unique())}")
    if "cycle_id" in result.columns:
        print(f"ciclos (semeadura->colheita): {int(result['cycle_id'].nunique())}")
    print(result[["SoilWater_root", "Dr_root", "TAW_root", "ETreal"]].describe())
    return result


if __name__ == "__main__":
    main()
