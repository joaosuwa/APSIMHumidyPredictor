"""Feature importance baseada na correlação de Pearson com o target."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from modeling.config import DEFAULT_CONFIG, DEFAULT_DATASET_PATH
from modeling.data import (
    METADATA_COLUMNS,
    MODELING_FEATURE_COLUMNS,
    TARGET_COLUMN,
    filter_simulations,
    load_training_dataset,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR_NAME = "plots"
CSV_NAME = "feature_importance.csv"
SUMMARY_NAME = "summary.txt"


@dataclass(frozen=True, slots=True)
class AnalysisArtifacts:
    """Caminhos e dados produzidos por uma execução da análise."""

    ranking: pd.DataFrame
    sample_size: int
    summary_path: Path
    csv_path: Path
    plot_paths: tuple[Path, ...]


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset sem colunas obrigatórias: {missing}")


def prepare_analysis_data(
    df: pd.DataFrame,
    simulations: Sequence[str],
    feature_columns: Sequence[str] = MODELING_FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Filtra simulações e devolve features e target validados como numéricos."""
    if not feature_columns:
        raise ValueError("A lista de features não pode ser vazia")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("A lista de features contém nomes duplicados")
    if target_column in feature_columns:
        raise ValueError("O target não pode fazer parte da lista de features")

    _require_columns(df, [*METADATA_COLUMNS, *feature_columns, target_column])
    filtered = filter_simulations(df, simulations)
    selected_columns = [*feature_columns, target_column]
    numeric = filtered.loc[:, selected_columns].apply(pd.to_numeric, errors="coerce")

    invalid = numeric.columns[numeric.isna().any()].tolist()
    if invalid:
        raise ValueError(f"Colunas com valores ausentes ou não numéricos: {invalid}")
    if len(numeric) < 2:
        raise ValueError("São necessárias pelo menos duas observações para a correlação")
    numeric.attrs["cycle_ids"] = sorted(int(value) for value in filtered["cycle_id"].unique())
    numeric.attrs["simulations"] = sorted(filtered["SimulationName"].unique().tolist())
    return numeric


def load_analysis_data(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    simulations: Sequence[str] = DEFAULT_CONFIG.included_simulations,
    feature_columns: Sequence[str] = MODELING_FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Carrega o CSV central e prepara a amostra usada na análise."""
    dataset = load_training_dataset(dataset_path)
    return prepare_analysis_data(dataset, simulations, feature_columns, target_column)


def calculate_correlations(
    data: pd.DataFrame,
    feature_columns: Sequence[str] = MODELING_FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Calcula e ordena as correlações de Pearson pela magnitude absoluta."""
    _require_columns(data, [*feature_columns, target_column])
    target = data[target_column]
    records: list[dict[str, object]] = []

    for feature in feature_columns:
        values = data[feature]
        is_constant = values.nunique(dropna=False) <= 1
        pearson_r = np.nan if is_constant else values.corr(target, method="pearson")
        if pd.isna(pearson_r):
            direction = "indefinida"
            status = "feature constante ou correlação indefinida"
            importance = np.nan
        elif pearson_r > 0:
            direction = "positiva"
            status = "ok"
            importance = abs(float(pearson_r))
        elif pearson_r < 0:
            direction = "negativa"
            status = "ok"
            importance = abs(float(pearson_r))
        else:
            direction = "neutra"
            status = "ok"
            importance = 0.0
        records.append(
            {
                "feature": feature,
                "pearson_r": pearson_r,
                "importance_abs": importance,
                "direction": direction,
                "status": status,
            }
        )

    ranking = pd.DataFrame.from_records(records)
    ranking = ranking.sort_values(
        ["importance_abs", "feature"],
        ascending=[False, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def _format_coefficient(value: float) -> str:
    return "indefinida" if pd.isna(value) else f"{value:+.6f}"


def build_summary(
    ranking: pd.DataFrame,
    data: pd.DataFrame,
    simulations: Sequence[str],
    target_column: str = TARGET_COLUMN,
) -> str:
    """Monta o relatório textual completo e determinístico."""
    undefined_count = int(ranking["pearson_r"].isna().sum())
    cycle_ids = data.attrs.get("cycle_ids")
    cycles = ", ".join(str(value) for value in cycle_ids) if cycle_ids else "não disponível"

    lines = [
        "ANÁLISE DE FEATURE IMPORTANCE POR CORRELAÇÃO (CA)",
        "=" * 54,
        "",
        "Metodologia",
        "-----------",
        f"Target: {target_column}",
        "Método: correlação de Pearson entre cada feature e o target.",
        "Importância: valor absoluto da correlação (|r|).",
        "Ordenação: |r| decrescente; empates resolvidos pelo nome da feature.",
        "O sinal de r indica a direção da associação linear.",
        "",
        "Amostra",
        "-------",
        f"Observações: {len(data)}",
        f"Features avaliadas: {len(ranking)}",
        f"Simulações: {', '.join(simulations)}",
        f"Ciclos: {cycles}",
        "Partição: dataset filtrado completo; inclui o ciclo 6 reservado como holdout no fluxo de modelagem.",
        f"Correlações indefinidas: {undefined_count}",
        "",
        "Ranking completo",
        "----------------",
    ]

    for row in ranking.itertuples(index=False):
        lines.append(
            f"{row.rank:02d}. {row.feature} | "
            f"r={_format_coefficient(row.pearson_r)} | "
            f"|r|={_format_coefficient(row.importance_abs).lstrip('+')} | "
            f"direção={row.direction} | status={row.status}"
        )

    lines.extend(
        [
            "",
            "Interpretação e limitações",
            "--------------------------",
            "A correlação mede associação linear marginal, não causalidade.",
            "Uma baixa correlação não exclui relações não lineares ou interações entre features.",
            "Features correlacionadas entre si podem apresentar importâncias marginais semelhantes.",
            "Como todos os ciclos foram usados, este resultado é descritivo e não está isolado do holdout.",
            "",
        ]
    )
    return "\n".join(lines)


def _save_importance_bar(
    ranking: pd.DataFrame, target_column: str, path: Path
) -> None:
    plot_data = ranking.iloc[::-1].copy()
    values = plot_data["importance_abs"].fillna(0.0)
    colors = plot_data["direction"].map(
        {"positiva": "#2E8B57", "negativa": "#D2691E", "neutra": "#808080"}
    ).fillna("#B0B0B0")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.barh(plot_data["feature"], values, color=colors)
    ax.set_title(f"Importância por correlação com {target_column}")
    ax.set_xlabel("Importância absoluta |r de Pearson|")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, max(1.0, float(values.max()) * 1.05))
    ax.legend(
        handles=[
            Patch(color="#2E8B57", label="Correlação positiva"),
            Patch(color="#D2691E", label="Correlação negativa"),
            Patch(color="#B0B0B0", label="Indefinida"),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_target_heatmap(
    ranking: pd.DataFrame, target_column: str, path: Path
) -> None:
    values = ranking["pearson_r"].to_numpy(dtype=float).reshape(-1, 1)
    masked_values = np.ma.masked_invalid(values)
    cmap = plt.colormaps["coolwarm"].with_extremes(bad="#D3D3D3")

    fig, ax = plt.subplots(figsize=(7, 13))
    image = ax.imshow(masked_values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks([0], labels=[target_column])
    ax.set_yticks(np.arange(len(ranking)), labels=ranking["feature"])
    ax.set_title("Correlação de Pearson das features com o target")
    for index, value in enumerate(values[:, 0]):
        label = "N/A" if np.isnan(value) else f"{value:+.3f}"
        ax.text(0, index, label, ha="center", va="center", fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.08, pad=0.08)
    colorbar.set_label("r de Pearson")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_top_scatter(
    ranking: pd.DataFrame,
    data: pd.DataFrame,
    target_column: str,
    path: Path,
    top_n: int = 6,
) -> None:
    top_features = ranking.loc[ranking["pearson_r"].notna(), "feature"].head(top_n).tolist()
    if not top_features:
        raise ValueError("Nenhuma feature possui correlação definida para o gráfico de dispersão")

    columns = 3
    rows = int(np.ceil(len(top_features) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(16, 5 * rows), squeeze=False)
    target = data[target_column].to_numpy(dtype=float)

    for ax, feature in zip(axes.flat, top_features, strict=False):
        values = data[feature].to_numpy(dtype=float)
        ax.scatter(values, target, s=10, alpha=0.22, color="#1F77B4", edgecolors="none")
        if np.ptp(values) > 0:
            slope, intercept = np.polyfit(values, target, 1)
            trend_x = np.linspace(values.min(), values.max(), 100)
            ax.plot(trend_x, slope * trend_x + intercept, color="#B22222", linewidth=2)
        coefficient = ranking.loc[ranking["feature"] == feature, "pearson_r"].iloc[0]
        ax.set_title(f"{feature}\nr={coefficient:+.3f}")
        ax.set_xlabel(feature)
        ax.set_ylabel(target_column)
        ax.grid(alpha=0.2)

    for ax in axes.flat[len(top_features) :]:
        ax.set_visible(False)
    fig.suptitle("Features mais correlacionadas com o target", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_artifacts(
    ranking: pd.DataFrame,
    data: pd.DataFrame,
    simulations: Sequence[str],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    target_column: str = TARGET_COLUMN,
) -> AnalysisArtifacts:
    """Grava ranking, resumo e gráficos no diretório solicitado."""
    destination = Path(output_dir)
    plots_dir = destination / PLOTS_DIR_NAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = destination / CSV_NAME
    summary_path = destination / SUMMARY_NAME
    plot_paths = (
        plots_dir / "feature_importance_bar.png",
        plots_dir / "target_correlation_heatmap.png",
        plots_dir / "top_features_scatter.png",
    )

    ranking.to_csv(csv_path, index=False, encoding="utf-8", float_format="%.10f")
    summary_path.write_text(
        build_summary(ranking, data, simulations, target_column), encoding="utf-8"
    )
    _save_importance_bar(ranking, target_column, plot_paths[0])
    _save_target_heatmap(ranking, target_column, plot_paths[1])
    _save_top_scatter(ranking, data, target_column, plot_paths[2])
    return AnalysisArtifacts(ranking, len(data), summary_path, csv_path, plot_paths)


def run_analysis(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    simulations: Sequence[str] = DEFAULT_CONFIG.included_simulations,
    feature_columns: Sequence[str] = MODELING_FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> AnalysisArtifacts:
    """Executa o pipeline completo de feature importance."""
    simulations = tuple(simulations)
    data = load_analysis_data(dataset_path, simulations, feature_columns, target_column)
    ranking = calculate_correlations(data, feature_columns, target_column)
    return write_artifacts(ranking, data, simulations, output_dir, target_column)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analisa feature importance por correlação de Pearson."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"CSV de entrada (padrão: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Diretório dos resultados (padrão: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--simulations",
        nargs="+",
        default=list(DEFAULT_CONFIG.included_simulations),
        help="Uma ou mais simulações incluídas na análise.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> AnalysisArtifacts:
    args = build_parser().parse_args(argv)
    artifacts = run_analysis(args.dataset, args.output_dir, args.simulations)
    print(f"observações analisadas: {artifacts.sample_size}")
    print(f"features avaliadas: {len(artifacts.ranking)}")
    print(f"ranking: {artifacts.csv_path}")
    print(f"resumo: {artifacts.summary_path}")
    for plot_path in artifacts.plot_paths:
        print(f"gráfico: {plot_path}")
    return artifacts


if __name__ == "__main__":
    main()
