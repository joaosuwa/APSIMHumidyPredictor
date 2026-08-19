"""CLI de download dos produtos GFS/GDEX configurados no projeto."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:  # Compatibilidade com ``python caminho/arquivo.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.gfs_data_extractor import gdex_client as rc
    from scripts.gfs_data_extractor.gfs_config import (
        DATASET,
        GFS_PRODUCTS,
        PRODUCT_GROUPS,
        RAW_OUTPUT_DIR,
        SITES,
        ProductConfig,
        SiteConfig,
        get_product,
        get_site,
        resolve_product_keys,
    )
else:
    from . import gdex_client as rc
    from .gfs_config import (
        DATASET,
        GFS_PRODUCTS,
        PRODUCT_GROUPS,
        RAW_OUTPUT_DIR,
        SITES,
        ProductConfig,
        SiteConfig,
        get_product,
        get_site,
        resolve_product_keys,
    )


def format_level(level: str | Mapping[str, object] | None) -> str | None:
    """Converte um nível estruturado para o formato aceito pelo GDEX."""
    if level is None:
        return None
    if isinstance(level, str):
        if not level.strip() or ":" not in level:
            raise ValueError("O nível textual deve estar no formato 'TIPO:VALOR'.")
        level_type, level_value = level.split(":", maxsplit=1)
    elif isinstance(level, Mapping):
        level_type = level.get("type", level.get("level"))
        level_value = level.get("value", level.get("level_value"))
    else:
        raise TypeError("level deve ser string, mapping ou None.")
    if level_type is None or level_value is None:
        raise ValueError("O nível deve informar tipo e valor.")
    level_type = str(level_type).strip()
    level_value = str(level_value).strip()
    if not level_type or not level_value:
        raise ValueError("O nível deve informar tipo e valor não vazios.")
    return f"{level_type}:{level_value}"


def _validate_date(value: str, option_name: str) -> str:
    try:
        datetime.strptime(value, "%Y%m%d%H%M")
    except ValueError as exc:
        raise ValueError(f"{option_name} deve usar o formato YYYYMMDDHHMM: {value}") from exc
    return value


def build_control(
    product: str | ProductConfig,
    site: str | SiteConfig,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Monta o payload GDEX sem executar chamadas de rede."""
    product_config = get_product(product)
    site_config = get_site(site)
    start = _validate_date(start_date or product_config.start_date, "start_date")
    end = _validate_date(end_date or product_config.end_date, "end_date")
    if start > end:
        raise ValueError("start_date deve ser anterior ou igual a end_date.")

    control = {
        "dataset": DATASET,
        "date": f"{start}/to/{end}",
        "datetype": "init",
        "param": product_config.param,
        "product": product_config.product,
        "oformat": "csv",
        "nlat": site_config.latitude,
        "slat": site_config.latitude,
        "wlon": site_config.longitude,
        "elon": site_config.longitude,
    }
    level = format_level(product_config.level)
    if level is not None:
        control["level"] = level
    return control


def submit_request(
    product: str | ProductConfig,
    site: str | SiteConfig,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Submete uma consulta pontual ao GDEX."""
    control = build_control(product, site, start_date=start_date, end_date=end_date)
    print(json.dumps(control, indent=4))
    response = rc.submit_json(control)
    if response.get("http_response") != 200:
        raise RuntimeError("Erro ao submeter request:\n" + json.dumps(response, indent=4))
    request_id = response["data"]["request_id"]
    print(f"Request ID: {request_id}")
    return request_id


def wait_for_request(request_id: str, interval: int = 20) -> dict:
    """Aguarda a conclusão de uma consulta do GDEX."""
    while True:
        status = rc.get_status(request_id)
        request_status = status["data"]["status"]
        print(f"Status: {request_status}")
        if request_status == "Completed":
            return status
        if request_status == "Error":
            raise RuntimeError(f"Request {request_id} terminou com erro.")
        time.sleep(interval)


def download_request(request_id: str, output_dir: str | Path):
    """Baixa os CSVs brutos para o diretório de um produto."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return rc.download(str(request_id), out_dir=str(destination) + os.sep)


def download_product(
    product: str | ProductConfig,
    site: str | SiteConfig,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    """Baixa um produto/localidade e devolve seu diretório RAW."""
    product_config = get_product(product)
    output_dir = RAW_OUTPUT_DIR / product_config.raw_directory
    request_id = submit_request(
        product_config,
        site,
        start_date=start_date,
        end_date=end_date,
    )
    try:
        wait_for_request(request_id)
        download_request(request_id, output_dir)
        return output_dir
    finally:
        rc.purge_request(str(request_id))


def download_products(
    products: Sequence[str],
    sites: Sequence[str],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[Path]:
    """Baixa sequencialmente todas as combinações solicitadas."""
    product_keys = resolve_product_keys(list(products))
    return [
        download_product(
            product_key,
            site,
            start_date=start_date,
            end_date=end_date,
        )
        for site in sites
        for product_key in product_keys
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baixa produtos históricos do GFS/GDEX.")
    parser.add_argument(
        "--sites",
        nargs="+",
        choices=tuple(SITES),
        default=["alegrete"],
        help="Localidades (padrão: alegrete).",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        choices=tuple(GFS_PRODUCTS) + tuple(PRODUCT_GROUPS),
        default=["forecast_24h"],
        help="Produtos ou grupos (padrão: forecast_24h).",
    )
    parser.add_argument("--start-date", help="Início opcional no formato YYYYMMDDHHMM.")
    parser.add_argument("--end-date", help="Fim opcional no formato YYYYMMDDHHMM.")
    return parser


def main(argv: Sequence[str] | None = None) -> list[Path]:
    args = build_parser().parse_args(argv)
    return download_products(
        args.products,
        args.sites,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()
