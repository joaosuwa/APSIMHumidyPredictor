"""Download de produtos do dataset GFS/GDEX.

Este módulo não lê, transforma ou consolida CSVs. O processamento local é
responsabilidade de ``gfs_data_processing.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path

try:
    from ..paths import ensure_data_directories
    from .gfs_config import (
        DATASET,
        END_DATE,
        LATITUDE,
        LONGITUDE,
        ALL_PRODUCTS,
        LEVEL_2M,
        RAW_OUTPUT_DIR,
        START_DATE,
    )
    from . import gdex_client as rc
except ImportError:  # Permite executar este arquivo diretamente.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import ensure_data_directories
    from gfs_config import ALL_PRODUCTS, DATASET, END_DATE, LATITUDE, LEVEL_2M, LONGITUDE, RAW_OUTPUT_DIR, START_DATE
    import gdex_client as rc


ensure_data_directories()


def format_level(level: str | Mapping[str, object] | None) -> str | None:
    """Converte um nível estruturado para o formato aceito pelo GDEX."""
    if level is None:
        return None
    if isinstance(level, str):
        if not level.strip() or ":" not in level:
            raise ValueError("O nível textual deve estar no formato 'TIPO:VALOR'.")
        level_type, level_value = level.split(":", maxsplit=1)
    elif isinstance(level, Mapping):
        level_type = level["type"] if "type" in level else level.get("level")
        level_value = level["value"] if "value" in level else level.get("level_value")
    else:
        raise TypeError("level deve ser string, mapping ou None.")

    if level_type is None or level_value is None:
        raise ValueError("O nível deve informar tipo e valor.")
    level_type = str(level_type).strip()
    level_value = str(level_value).strip()
    if not level_type or not level_value:
        raise ValueError("O nível deve informar tipo e valor não vazios.")
    return f"{level_type}:{level_value}"


def build_control(
    param: str,
    product: str,
    level: str | Mapping[str, object] | None = LEVEL_2M,
) -> dict:
    """Monta o payload de uma consulta sem executar a chamada de rede."""
    control = {
        "dataset": DATASET,
        "date": f"{START_DATE}/to/{END_DATE}",
        "datetype": "init",
        "param": param,
        "product": product,
        "oformat": "csv",
        "nlat": LATITUDE,
        "slat": LATITUDE,
        "wlon": LONGITUDE,
        "elon": LONGITUDE,
    }
    formatted_level = format_level(level)
    if formatted_level is not None:
        control["level"] = formatted_level
    return control


def submit_request(
    param: str,
    product: str,
    level: str | Mapping[str, object] | None = LEVEL_2M,
) -> str:
    """Submete uma consulta pontual ao GDEX."""
    control = build_control(param, product, level)
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


def download_request(request_id: str, output_dir: str | Path = RAW_OUTPUT_DIR):
    """Baixa os CSVs brutos para a pasta de downloads do GFS."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return rc.download(str(request_id), out_dir=str(output_dir) + "/")


def purge_request(request_id: str):
    """Remove a request finalizada do GDEX."""
    return rc.purge_request(str(request_id))


def download_product(product_info: dict) -> Path:
    """Baixa um produto GFS e devolve o diretório dos arquivos brutos."""
    output_dir = RAW_OUTPUT_DIR / product_info["name"]
    request_id = submit_request(
        product_info["param"],
        product_info["product"],
        product_info.get("level", LEVEL_2M),
    )
    try:
        wait_for_request(request_id)
        download_request(request_id, output_dir)
        return output_dir
    finally:
        purge_request(request_id)


def download_all_products(products: list[dict] = ALL_PRODUCTS) -> list[Path]:
    """Baixa todos os produtos configurados, sem processar os CSVs."""
    return [download_product(product) for product in products]


if __name__ == "__main__":
    download_all_products()

