"""Leitura e consolidação diária dos produtos brutos do GFS/GDEX."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:  # Compatibilidade com ``python caminho/arquivo.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.data_io import read_csv_files, write_csv
    from scripts.gfs_data_extractor.gfs_config import (
        FORECAST_24H_KEYS,
        RADIATION_KEYS,
        RAW_OUTPUT_DIR,
        SITES,
        TEMPERATURE_MAX_KEYS,
        TEMPERATURE_MIN_KEYS,
        ProductConfig,
        SiteConfig,
        coordinate_file_pattern,
        get_product,
        get_site,
        processed_forecast_path,
    )
else:
    from ..data_io import read_csv_files, write_csv
    from .gfs_config import (
        FORECAST_24H_KEYS,
        RADIATION_KEYS,
        RAW_OUTPUT_DIR,
        SITES,
        TEMPERATURE_MAX_KEYS,
        TEMPERATURE_MIN_KEYS,
        ProductConfig,
        SiteConfig,
        coordinate_file_pattern,
        get_product,
        get_site,
        processed_forecast_path,
    )


FINAL_FORECAST_COLUMNS = [
    "date",
    "previsao_chuva_24h_mm",
    "previsao_temperatura_maxima_C",
    "previsao_temperatura_minima_C",
    "previsao_temperatura_media_C",
    "previsao_radiacao_solar_MJ_m2_dia",
    "u_grd_10m_m_s",
    "v_grd_10m_m_s",
    "umidade_relativa_prevista_pct",
    "ponto_orvalho_previsto_C",
]


def _find_value_column(df: pd.DataFrame) -> str:
    """Encontra a única coluna meteorológica numérica do CSV GDEX."""
    candidates = [column for column in df.columns if column not in {"Date", "Time"}]
    numeric_candidates = [
        column
        for column in candidates
        if pd.to_numeric(df[column], errors="coerce").notna().any()
    ]
    if len(numeric_candidates) != 1:
        raise ValueError(
            "Não foi possível identificar uma única coluna meteorológica. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )
    return numeric_candidates[0]


def _deduplicate_values(df: pd.DataFrame, product: ProductConfig) -> pd.DataFrame:
    """Remove repetições idênticas e rejeita valores conflitantes."""
    conflicts = (
        df.groupby("datetime", sort=False)["value"]
        .nunique(dropna=False)
        .loc[lambda count: count > 1]
    )
    if not conflicts.empty:
        timestamps = conflicts.index.strftime("%Y-%m-%d %H:%M").tolist()
        raise ValueError(
            f"Valores conflitantes para {product.key} nos horários: {timestamps[:10]}"
        )
    return df.drop_duplicates(["datetime", "value"]).sort_values("datetime").reset_index(drop=True)


def _convert_product_values(values: pd.Series, product: ProductConfig) -> pd.Series:
    """Converte unidades e valida limites próprios de cada produto."""
    if product.family in {"temperature_max", "temperature_min"} or product.key == "dpt_24":
        values = values - 273.15
    if product.key == "rh_24" and ((values < 0) | (values > 100)).any():
        raise ValueError("Umidade relativa fora do intervalo [0, 100].")
    return values


def read_raw_product(
    product: str | ProductConfig,
    site: str | SiteConfig,
    *,
    raw_output_dir: str | Path = RAW_OUTPUT_DIR,
    file_pattern: str | None = None,
) -> pd.DataFrame:
    """Lê um produto bruto e devolve somente ``datetime`` e ``value``."""
    product_config = get_product(product)
    site_config = get_site(site)
    pattern = file_pattern or coordinate_file_pattern(
        site_config.latitude,
        site_config.longitude,
    )
    raw_dir = Path(raw_output_dir) / product_config.raw_directory
    if not raw_dir.exists() or not any(raw_dir.rglob(pattern)):
        raise FileNotFoundError(
            f"Dados brutos de {product_config.key} não encontrados em {raw_dir} "
            f"para {site_config.name} ({pattern})."
        )

    raw = read_csv_files(raw_dir, pattern=pattern)
    missing = {"Date", "Time"} - set(raw.columns)
    if missing:
        raise ValueError(f"CSV GFS sem colunas obrigatórias: {sorted(missing)}")

    value_column = _find_value_column(raw)
    parsed = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                raw["Date"].astype(str) + " " + raw["Time"].astype(str),
                errors="coerce",
            ),
            "value": pd.to_numeric(raw[value_column], errors="coerce"),
        }
    ).dropna()
    if parsed.empty:
        raise ValueError(f"Nenhum registro válido encontrado para {product_config.key}.")
    parsed["value"] = _convert_product_values(parsed["value"], product_config)
    return _deduplicate_values(parsed, product_config)


def _validate_daily_uniqueness(df: pd.DataFrame, product: ProductConfig) -> None:
    duplicated = df.loc[df["date"].duplicated(keep=False), "date"]
    if not duplicated.empty:
        dates = duplicated.dt.strftime("%Y-%m-%d").unique().tolist()
        raise ValueError(f"Mais de um registro diário para {product.key}: {dates[:10]}")


def load_daily_24h_product(
    product_key: str,
    site: str | SiteConfig,
    hour: int = 0,
    *,
    raw_output_dir: str | Path = RAW_OUTPUT_DIR,
    file_pattern: str | None = None,
) -> pd.DataFrame:
    """Seleciona um produto de +24 h e o alinha à inicialização em D."""
    product = get_product(product_key)
    if product.family != "forecast_24h":
        raise ValueError(f"{product.key} não é um produto de horizonte de 24 horas.")
    if hour not in range(24):
        raise ValueError("hour deve estar entre 0 e 23.")

    raw = read_raw_product(
        product,
        site,
        raw_output_dir=raw_output_dir,
        file_pattern=file_pattern,
    )
    selected = raw.loc[
        raw["datetime"].dt.hour.eq(hour) & raw["datetime"].dt.minute.eq(0)
    ].copy()
    if selected.empty:
        raise ValueError(f"Nenhum registro de {hour:02d}:00 encontrado para {product.key}.")
    selected["date"] = selected["datetime"].dt.normalize() - pd.Timedelta(days=1)
    _validate_daily_uniqueness(selected, product)
    return selected[["date", "value"]].rename(columns={"value": product.output_column}).reset_index(drop=True)


def load_six_hour_intervals(
    product_keys: Sequence[str],
    site: str | SiteConfig,
    *,
    raw_output_dir: str | Path = RAW_OUTPUT_DIR,
) -> pd.DataFrame:
    """Monta os intervalos de seis horas na data comum de inicialização."""
    if not product_keys:
        raise ValueError("Informe ao menos um produto de seis horas.")

    frames: list[pd.DataFrame] = []
    columns: list[str] = []
    for product_key in product_keys:
        product = get_product(product_key)
        if product.family == "forecast_24h" or product.interval_end_hour is None:
            raise ValueError(f"{product.key} não é um produto de intervalo de seis horas.")
        raw = read_raw_product(product, site, raw_output_dir=raw_output_dir)
        selected = raw.loc[
            raw["datetime"].dt.hour.eq(product.interval_end_hour)
            & raw["datetime"].dt.minute.eq(0)
        ].copy()
        if selected.empty:
            raise ValueError(
                f"Nenhum final de intervalo às {product.interval_end_hour:02d}:00 "
                f"encontrado para {product.key}."
            )
        selected["date"] = selected["datetime"].dt.normalize()
        if product.interval_end_hour == 0:
            selected["date"] -= pd.Timedelta(days=1)
        _validate_daily_uniqueness(selected, product)
        columns.append(product.output_column)
        frames.append(
            selected[["date", "value"]].rename(columns={"value": product.output_column})
        )

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on="date", how="outer", validate="one_to_one")
    return combined.dropna(subset=columns).sort_values("date").reset_index(drop=True)


def _build_daily_temperature(
    site: str | SiteConfig,
    raw_output_dir: str | Path,
) -> pd.DataFrame:
    keys = TEMPERATURE_MAX_KEYS + TEMPERATURE_MIN_KEYS
    intervals = load_six_hour_intervals(keys, site, raw_output_dir=raw_output_dir)
    max_columns = [get_product(key).output_column for key in TEMPERATURE_MAX_KEYS]
    min_columns = [get_product(key).output_column for key in TEMPERATURE_MIN_KEYS]
    intervals["previsao_temperatura_maxima_C"] = intervals[max_columns].max(axis=1)
    intervals["previsao_temperatura_minima_C"] = intervals[min_columns].min(axis=1)
    intervals["previsao_temperatura_media_C"] = (
        intervals["previsao_temperatura_maxima_C"]
        + intervals["previsao_temperatura_minima_C"]
    ) / 2.0
    return intervals[
        [
            "date",
            "previsao_temperatura_maxima_C",
            "previsao_temperatura_minima_C",
            "previsao_temperatura_media_C",
        ]
    ]


def _build_daily_radiation(
    site: str | SiteConfig,
    raw_output_dir: str | Path,
) -> pd.DataFrame:
    intervals = load_six_hour_intervals(RADIATION_KEYS, site, raw_output_dir=raw_output_dir)
    columns = [get_product(key).output_column for key in RADIATION_KEYS]
    intervals["previsao_radiacao_solar_MJ_m2_dia"] = intervals[columns].sum(axis=1) * 0.0216
    return intervals[["date", "previsao_radiacao_solar_MJ_m2_dia"]]


def generate_site_forecast(
    site: str | SiteConfig,
    output_file: str | Path | None = None,
    *,
    hour: int = 0,
    raw_output_dir: str | Path = RAW_OUTPUT_DIR,
) -> pd.DataFrame:
    """Gera o forecast diário consolidado de uma localidade."""
    site_config = get_site(site)
    frames = [
        _build_daily_temperature(site_config, raw_output_dir),
        _build_daily_radiation(site_config, raw_output_dir),
        *[
            load_daily_24h_product(
                product_key,
                site_config,
                hour=hour,
                raw_output_dir=raw_output_dir,
            )
            for product_key in FORECAST_24H_KEYS
        ],
    ]
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on="date", how="outer", validate="one_to_one")
    combined = combined.sort_values("date").reset_index(drop=True).reindex(columns=FINAL_FORECAST_COLUMNS)

    destination = Path(output_file) if output_file is not None else processed_forecast_path(site_config)
    export = combined.copy()
    export["date"] = export["date"].dt.strftime("%Y-%m-%d")
    write_csv(export, destination)
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolida produtos GFS já baixados.")
    parser.add_argument(
        "--sites",
        nargs="+",
        choices=tuple(SITES),
        default=list(SITES),
        help="Localidades a processar (padrão: todas).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, pd.DataFrame]:
    args = build_parser().parse_args(argv)
    results = {}
    for site in args.sites:
        result = generate_site_forecast(site)
        results[site] = result
        print(f"{site}: {len(result)} dias salvos em {processed_forecast_path(site)}")
    return results


if __name__ == "__main__":
    main()
