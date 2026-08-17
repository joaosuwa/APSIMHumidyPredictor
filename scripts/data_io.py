"""Operações de entrada e saída compartilhadas pelos pipelines de dados."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .paths import MODEL_DATA_DIR
except ImportError:  # Permite executar módulos pelo caminho do arquivo.
    from paths import MODEL_DATA_DIR


def read_csv_files(
    source: str | Path | list[str | Path],
    pattern: str = "*.csv",
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Lê um CSV ou concatena todos os CSVs de um diretório."""
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        files = sorted(source_path.rglob(pattern)) if source_path.is_dir() else [source_path]
    else:
        files = [Path(path) for path in source]

    if not files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {source!s}")

    frames = []
    errors = []
    for csv_file in files:
        try:
            frames.append(pd.read_csv(csv_file, **read_csv_kwargs))
        except Exception as exc:
            errors.append(f"{csv_file}: {exc}")

    if not frames:
        detail = "\n".join(errors)
        raise RuntimeError(f"Não foi possível ler os CSVs:\n{detail}")

    return pd.concat(frames, ignore_index=True)


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Salva um DataFrame em CSV, criando o diretório de destino."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def write_model_dataset(
    df: pd.DataFrame,
    path: str | Path = MODEL_DATA_DIR / "training_dataset.csv",
) -> Path:
    """Salva o dataset central que alimentará o modelo."""
    return write_csv(df, path)
