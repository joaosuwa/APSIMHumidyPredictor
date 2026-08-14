"""Download de produtos do dataset GFS/GDEX.

Este módulo não lê, transforma ou consolida CSVs. O processamento local é
responsabilidade de ``gfs_data_processing.py``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    from ..paths import ensure_data_directories
    from .gfs_config import (
        DATASET,
        END_DATE,
        LATITUDE,
        LONGITUDE,
        PRODUCTS,
        RAW_OUTPUT_DIR,
        START_DATE,
    )
    from . import gdex_client as rc
except ImportError:  # Permite executar este arquivo diretamente.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import ensure_data_directories
    from gfs_config import DATASET, END_DATE, LATITUDE, LONGITUDE, PRODUCTS, RAW_OUTPUT_DIR, START_DATE
    import gdex_client as rc


ensure_data_directories()


def submit_request(param: str, product: str) -> str:
    """Submete uma consulta pontual ao GDEX."""
    control = {
        "dataset": DATASET,
        "date": f"{START_DATE}/to/{END_DATE}",
        "datetype": "init",
        "param": param,
        "level": "HTGL:2",
        "product": product,
        "oformat": "csv",
        "nlat": LATITUDE,
        "slat": LATITUDE,
        "wlon": LONGITUDE,
        "elon": LONGITUDE,
    }
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
    request_id = submit_request(product_info["param"], product_info["product"])
    try:
        wait_for_request(request_id)
        download_request(request_id, output_dir)
        return output_dir
    finally:
        purge_request(request_id)


def download_all_products(products: list[dict] = PRODUCTS) -> list[Path]:
    """Baixa todos os produtos configurados, sem processar os CSVs."""
    return [download_product(product) for product in products]


def main() -> list[Path]:
    """Executa somente a etapa de download dos produtos GFS."""
    return download_all_products()


if __name__ == "__main__":
    main()
