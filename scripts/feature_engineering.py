"""Ponto de entrada compatível para engenharia de features.

As implementações específicas ficam em ``scripts.apsim`` e
``scripts.nasa_power``. Os nomes abaixo continuam disponíveis para preservar
os imports existentes no projeto e nos scripts antigos.
"""

from __future__ import annotations

try:
    from .apsim.processing import (
        DEFAULT_COLUMNS_TO_DROP,
        DEFAULT_MODEL_DATASET,
        DEFAULT_OUTPUT,
        DEFAULT_REPORT,
        add_apsim_features,
        build_report_features,
        drop_columns,
        filter_to_crop_window,
        read_apsim_report,
    )
    from .data_io import read_csv_files, write_csv, write_model_dataset
    from .nasa_power.processing import process_nasa_power_data, read_nasa_power_data
except ImportError:  # Permite executar este arquivo diretamente.
    from apsim.processing import (
        DEFAULT_COLUMNS_TO_DROP,
        DEFAULT_MODEL_DATASET,
        DEFAULT_OUTPUT,
        DEFAULT_REPORT,
        add_apsim_features,
        build_report_features,
        drop_columns,
        filter_to_crop_window,
        read_apsim_report,
    )
    from data_io import read_csv_files, write_csv, write_model_dataset
    from nasa_power.processing import process_nasa_power_data, read_nasa_power_data


def main(
    report_path=DEFAULT_REPORT,
    output_path=DEFAULT_OUTPUT,
    columns_to_drop: list[str] | None = None,
):
    """Executa o processamento padrão do relatório do APSIM NG."""
    full = read_apsim_report(report_path)
    result = build_report_features(
        report_path=report_path,
        output_path=output_path,
        columns_to_drop=columns_to_drop,
    )
    print(f"linhas no arquivo original: {len(full)}")
    print(f"linhas na janela de cultivo: {len(result)}")
    print(f"simulacoes: {sorted(result['SimulationName'].unique())}")
    if "cycle_id" in result.columns:
        print(f"ciclos (semeadura->colheita): {int(result['cycle_id'].nunique())}")
    print(f"colunas novas: {sorted(set(result.columns) - set(full.columns))}")
    print(result[["SoilWater_root", "Dr_root", "TAW_root", "ETreal"]].describe())
    return result


if __name__ == "__main__":
    main()
