import json
import time
from pathlib import Path

import pandas as pd

import gdex_client as rc


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DATASET = "d084001"

LATITUDE = -28
LONGITUDE = -53.75

# ============================================================
# PERÍODO COMPLETO
#
# Ambos em UTC.
#
# Início:
# 2019-06-13 00:00 UTC
#
# Fim:
# 2026-08-13 12:00 UTC
# ============================================================

START_DATE = "201906130000"
END_DATE =   "202608011200"

OUTPUT_DIR = Path("gdex_temperature_historical")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PRODUTOS DO GFS
# ============================================================

PRODUCTS = [
    {
        "name": "Tmax_0_6",
        "param": "T MAX",
        "product": "6-hour Maximum (initial+0 to initial+6)",
    },
    {
        "name": "Tmax_6_12",
        "param": "T MAX",
        "product": "6-hour Maximum (initial+6 to initial+12)",
    },
    {
        "name": "Tmax_12_18",
        "param": "T MAX",
        "product": "6-hour Maximum (initial+12 to initial+18)",
    },
    {
        "name": "Tmax_18_24",
        "param": "T MAX",
        "product": "6-hour Maximum (initial+18 to initial+24)",
    },

    {
        "name": "Tmin_0_6",
        "param": "T MIN",
        "product": "6-hour Minimum (initial+0 to initial+6)",
    },
    {
        "name": "Tmin_6_12",
        "param": "T MIN",
        "product": "6-hour Minimum (initial+6 to initial+12)",
    },
    {
        "name": "Tmin_12_18",
        "param": "T MIN",
        "product": "6-hour Minimum (initial+12 to initial+18)",
    },
    {
        "name": "Tmin_18_24",
        "param": "T MIN",
        "product": "6-hour Minimum (initial+18 to initial+24)",
    },
]


# ============================================================
# SUBMIT
# ============================================================

def submit_request(param, product):

    control = {
        "dataset": DATASET,

        # Intervalo das inicializações do GFS
        "date": f"{START_DATE}/to/{END_DATE}",

        # IMPORTANTE:
        # Estamos selecionando as datas de inicialização
        "datetype": "init",

        "param": param,

        # HTGL = altura acima do solo
        # 2 m
        "level": "HTGL:2",

        "product": product,

        "oformat": "csv",

        # Ponto único
        "nlat": LATITUDE,
        "slat": LATITUDE,
        "wlon": LONGITUDE,
        "elon": LONGITUDE,
    }

    print("\n================================================")
    print("SUBMETENDO REQUEST")
    print("================================================")

    print(json.dumps(control, indent=4))

    response = rc.submit_json(control)

    if response.get("http_response") != 200:

        raise RuntimeError(
            "Erro ao submeter request:\n"
            + json.dumps(response, indent=4)
        )

    request_id = response["data"]["request_id"]

    print(f"\nRequest ID: {request_id}")

    return request_id


# ============================================================
# AGUARDAR
# ============================================================

def wait_for_request(request_id, interval=20):

    print(f"\nAguardando request {request_id}...")

    while True:

        status = rc.get_status(request_id)

        request_status = status["data"]["status"]

        print(f"Status: {request_status}")

        if request_status == "Completed":

            print("Request concluído.")

            return status

        if request_status == "Error":

            print("\nSTATUS COMPLETO:")
            print(
                json.dumps(
                    status,
                    indent=4,
                    ensure_ascii=False
                )
            )

            raise RuntimeError(
                f"Request {request_id} terminou com erro."
            )

        time.sleep(interval)


# ============================================================
# DOWNLOAD
# ============================================================

def download_request(request_id, output_dir):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nBaixando request {request_id}..."
    )

    return rc.download(
        str(request_id),
        out_dir=str(output_dir) + "/"
    )


# ============================================================
# PURGE
# ============================================================

def purge_request(request_id):

    print(
        f"\nPurgando request {request_id}..."
    )

    result = rc.purge_request(
        str(request_id)
    )

    print("Request purgada.")

    return result


# ============================================================
# LER TODOS OS CSVs
# ============================================================

def read_downloaded_files(directory):

    directory = Path(directory)

    csv_files = list(
        directory.rglob("*.csv")
    )

    print(
        f"\nEncontrados {len(csv_files)} CSVs "
        f"em {directory}"
    )

    if not csv_files:

        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {directory}"
        )

    frames = []

    for i, csv_file in enumerate(csv_files, start=1):

        try:

            df = pd.read_csv(csv_file)

            frames.append(df)

        except Exception as e:

            print(
                f"Erro lendo {csv_file}: {e}"
            )

        if i % 500 == 0:

            print(
                f"Lidos {i}/{len(csv_files)} arquivos..."
            )

    if not frames:

        raise RuntimeError(
            f"Não foi possível ler nenhum CSV em {directory}"
        )

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================
# IDENTIFICAR COLUNA DE TEMPERATURA
# ============================================================

def find_temperature_column(df):

    candidates = []

    for col in df.columns:

        text = col.lower()

        if (
            "temperature" in text
            or "t max" in text
            or "t min" in text
        ):

            candidates.append(col)

    if not candidates:

        raise ValueError(
            "Coluna de temperatura não encontrada.\n"
            f"Colunas disponíveis: {df.columns.tolist()}"
        )

    return candidates[-1]


# ============================================================
# PROCESSAR RESULTADO
# ============================================================

def process_temperature_directory(
    directory,
    variable_name
):

    df = read_downloaded_files(
        directory
    )

    print("\nColunas encontradas:")
    print(df.columns.tolist())

    temp_col = find_temperature_column(
        df
    )

    print(
        f"\nColuna utilizada: {temp_col}"
    )

    # --------------------------------------------------------
    # Kelvin -> Celsius
    # --------------------------------------------------------

    df[temp_col] = pd.to_numeric(
        df[temp_col],
        errors="coerce"
    )

    df["temperature_C"] = (
        df[temp_col] - 273.15
    )

    # --------------------------------------------------------
    # Criar datetime
    # --------------------------------------------------------

    if (
        "Date" in df.columns
        and "Time" in df.columns
    ):

        df["datetime"] = pd.to_datetime(
            df["Date"].astype(str)
            + " "
            + df["Time"].astype(str),
            errors="coerce"
        )

    # --------------------------------------------------------
    # Selecionar apenas o necessário
    # --------------------------------------------------------

    result = df.copy()

    result["variable"] = variable_name

    return result


# ============================================================
# PROCESSAR UMA REQUEST
# ============================================================

def process_product(product_info):

    name = product_info["name"]

    param = product_info["param"]

    product = product_info["product"]

    output_dir = OUTPUT_DIR / name

    print("\n\n")
    print("################################################")
    print(f"PROCESSANDO {name}")
    print("################################################")

    # --------------------------------------------------------
    # SUBMIT
    # --------------------------------------------------------

    request_id = submit_request(
        param=param,
        product=product
    )

    # --------------------------------------------------------
    # WAIT
    # --------------------------------------------------------

    wait_for_request(
        request_id
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    download_request(
        request_id,
        output_dir
    )

    # --------------------------------------------------------
    # PROCESSAR
    # --------------------------------------------------------

    df = process_temperature_directory(
        output_dir,
        name
    )

    # --------------------------------------------------------
    # Salvar consolidado
    # --------------------------------------------------------

    consolidated_file = (
        OUTPUT_DIR
        / f"{name}.csv"
    )

    df.to_csv(
        consolidated_file,
        index=False
    )

    print(
        f"\nArquivo consolidado:"
        f"\n{consolidated_file}"
    )

    # --------------------------------------------------------
    # PURGE
    # --------------------------------------------------------

    purge_request(
        request_id
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    all_results = {}

    for product in PRODUCTS:

        df = process_product(
            product
        )

        all_results[
            product["name"]
        ] = df

    print("\n")
    print("================================================")
    print("TODAS AS REQUESTS FINALIZADAS")
    print("================================================")

    # ========================================================
    # MONTAR DATAFRAME FINAL
    # ========================================================

    print("\nProcessando resultados...")

    final_df = None

    # --------------------------------------------------------
    # Cada produto possui:
    #
    # datetime
    # temperature_C
    #
    # Vamos juntar todos pelo datetime.
    # --------------------------------------------------------

    for name, df in all_results.items():

        temp = df[
            [
                "datetime",
                "temperature_C"
            ]
        ].copy()

        temp = temp.rename(
            columns={
                "temperature_C": name
            }
        )

        # Remover duplicatas temporais
        temp = temp.drop_duplicates(
            subset=["datetime"]
        )

        if final_df is None:

            final_df = temp

        else:

            final_df = final_df.merge(
                temp,
                on="datetime",
                how="outer"
            )

    # --------------------------------------------------------
    # Ordenar
    # --------------------------------------------------------

    final_df = final_df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    # ========================================================
    # Tmax/Tmin DAS 24 H
    # ========================================================

    max_columns = [
        "Tmax_0_6",
        "Tmax_6_12",
        "Tmax_12_18",
        "Tmax_18_24",
    ]

    min_columns = [
        "Tmin_0_6",
        "Tmin_6_12",
        "Tmin_12_18",
        "Tmin_18_24",
    ]

    # ========================================================
    # IMPORTANTE:
    #
    # Cada linha datetime representa a inicialização do GFS.
    #
    # O Tmax_24h é a maior Tmax entre os quatro intervalos.
    # O Tmin_24h é a menor Tmin entre os quatro intervalos.
    # ========================================================

    final_df["Tmax_24h_C"] = final_df[
        max_columns
    ].max(axis=1)

    final_df["Tmin_24h_C"] = final_df[
        min_columns
    ].min(axis=1)

    final_df["Tmean_24h_C"] = (
        final_df["Tmax_24h_C"]
        + final_df["Tmin_24h_C"]
    ) / 2.0

    # ========================================================
    # SALVAR
    # ========================================================

    output_file = (
        OUTPUT_DIR
        / "gfs_temperature_20190613_20260813.csv"
    )

    final_df.to_csv(
        output_file,
        index=False
    )

    print("\n")
    print("================================================")
    print("ARQUIVO FINAL")
    print("================================================")

    print(output_file)

    print("\nNúmero de registros:")
    print(len(final_df))

    print("\nPrimeiras linhas:")
    print(
        final_df.head(10).to_string(
            index=False
        )
    )

    print("\nÚltimas linhas:")
    print(
        final_df.tail(10).to_string(
            index=False
        )
    )


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    main()