"""Caminhos compartilhados do projeto.

Os dados são divididos em duas áreas principais:

* ``data/raw``: arquivos originais ou baixados das fontes externas;
* ``data/processed``: arquivos transformados e prontos para consumo do
  pipeline/modelo.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_APSIM_DIR = RAW_DATA_DIR / "apsim"
RAW_NASA_POWER_DIR = RAW_DATA_DIR / "nasa_power"
RAW_GFS_DIR = RAW_DATA_DIR / "gfs"

PROCESSED_APSIM_DIR = PROCESSED_DATA_DIR / "apsim"
PROCESSED_NASA_POWER_DIR = PROCESSED_DATA_DIR / "nasa_power"
PROCESSED_GFS_DIR = PROCESSED_DATA_DIR / "gfs"
PROCESSED_VALIDATION_DIR = PROCESSED_DATA_DIR / "validation"
MODEL_DATA_DIR = PROCESSED_DATA_DIR / "model"
