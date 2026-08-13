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

# Cada produto chega do GDEX no horário final do intervalo. Portanto,
# estes são os horários que devem ser usados na consolidação diária.
INTERVAL_END_HOURS = {
    "Tmax_0_6": 6,
    "Tmax_6_12": 12,
    "Tmax_12_18": 18,
    "Tmax_18_24": 0,
    "Tmin_0_6": 6,
    "Tmin_6_12": 12,
    "Tmin_12_18": 18,
    "Tmin_18_24": 0,
}

FINAL_COLUMNS = [
    "datetime",
    "Tmax_0_6",
    "Tmax_6_12",
    "Tmax_12_18",
    "Tmax_18_24",
    "Tmin_0_6",
    "Tmin_6_12",
    "Tmin_12_18",
    "Tmin_18_24",
    "Tmax_24h_C",
    "Tmin_24h_C",
    "Tmean_24h_C",
]

FINAL_OUTPUT_FILE = (
    OUTPUT_DIR
    / "gfs_temperature_20190613_20260813.csv"
)

# Os downloads brutos possuem dois pontos. Use o padrão correspondente ao
# local desejado para não misturar Alegrete e Nova Ramada.
ALEGRETE_FILE_PATTERN = "*29.75S_55.75W.csv"
NOVA_RAMADA_FILE_PATTERN = "*28.0S_53.75W.csv"


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

def read_downloaded_files(directory, file_pattern="*.csv"):

    directory = Path(directory)

    csv_files = list(directory.rglob(file_pattern))

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
    variable_name,
    file_pattern="*.csv"
):

    df = read_downloaded_files(
        directory,
        file_pattern=file_pattern
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

def process_product(product_info, file_pattern="*.csv"):

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
        name,
        file_pattern=file_pattern
    )

    # --------------------------------------------------------
    # PURGE
    # --------------------------------------------------------

    purge_request(
        request_id
    )

    return df


# ============================================================
# REUSE LOCAL DATA
# ============================================================

def load_or_process_product(
    product_info,
    fetch_missing=False,
    file_pattern="*.csv"
):

    name = product_info["name"]
    raw_directory = OUTPUT_DIR / name

    # Sempre lê os downloads brutos em Kelvin. A conversão para Celsius
    # acontece em process_temperature_directory e permanece em memória.
    matching_files = list(raw_directory.rglob(file_pattern))

    if raw_directory.exists() and matching_files:

        print(
            f"\nUsando download bruto existente: {raw_directory} "
            f"({len(matching_files)} arquivo(s))"
        )

        df = process_temperature_directory(
            raw_directory,
            name,
            file_pattern=file_pattern
        )

        return df

    if not fetch_missing:

        raise FileNotFoundError(
            f"Local data not found for {name}. "
            "Use main(fetch_missing=True) to download GFS data."
        )

    return process_product(
        product_info,
        file_pattern=file_pattern
    )


# ============================================================
# DAILY CONSOLIDATION
# ============================================================

def build_daily_temperature_dataframe(all_results):

    daily_df = None

    for product_info in PRODUCTS:

        name = product_info["name"]
        df = all_results[name]

        temp = df[["datetime", "temperature_C"]].copy()
        temp["datetime"] = pd.to_datetime(
            temp["datetime"],
            errors="coerce"
        )
        temp["temperature_C"] = pd.to_numeric(
            temp["temperature_C"],
            errors="coerce"
        )
        temp = temp.dropna(subset=["datetime"])

        # GDEX returns each interval at its ending time:
        # 06:00, 12:00, 18:00 or 00:00.
        expected_hour = INTERVAL_END_HOURS[name]
        temp = temp[temp["datetime"].dt.hour == expected_hour]

        # Keep the first occurrence, as the previous script did when the
        # historical directory contained duplicated timestamps.
        temp = temp.sort_values("datetime")
        temp = temp.drop_duplicates(
            subset=["datetime"],
            keep="first"
        )

        temp["date"] = temp["datetime"].dt.normalize()

        # 00:00 is the end of the previous day's 18:00-24:00 interval.
        if name.endswith("18_24"):
            temp["date"] = temp["date"] - pd.Timedelta(days=1)

        temp = temp.sort_values("date")
        temp = temp.drop_duplicates(
            subset=["date"],
            keep="first"
        )
        temp = temp[["date", "temperature_C"]].rename(
            columns={"temperature_C": name}
        )

        if daily_df is None:
            daily_df = temp
        else:
            daily_df = daily_df.merge(
                temp,
                on="date",
                how="outer"
            )

    if daily_df is None:
        raise RuntimeError("No temperature data was consolidated.")

    daily_df = daily_df.sort_values("date").reset_index(drop=True)

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

    daily_df["Tmax_24h_C"] = daily_df[max_columns].max(axis=1)
    daily_df["Tmin_24h_C"] = daily_df[min_columns].min(axis=1)

    # Requested formula: (Tmax24 - Tmin24) / 2.
    daily_df["Tmean_24h_C"] = (
        daily_df["Tmax_24h_C"]
        - daily_df["Tmin_24h_C"]
    ) / 2.0

    daily_df = daily_df.rename(columns={"date": "datetime"})
    daily_df["datetime"] = daily_df["datetime"].dt.strftime("%Y-%m-%d")

    return daily_df.reindex(columns=FINAL_COLUMNS)


def generate_gfs_temperature_file(
    fetch_missing=False,
    output_file=FINAL_OUTPUT_FILE,
    file_pattern=ALEGRETE_FILE_PATTERN
):

    all_results = {}

    for product in PRODUCTS:

        all_results[product["name"]] = load_or_process_product(
            product,
            fetch_missing=fetch_missing,
            file_pattern=file_pattern
        )

    final_df = build_daily_temperature_dataframe(all_results)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False)

    return final_df


# ============================================================
# MAIN
# ============================================================

def main(fetch_missing=False):

    final_df = generate_gfs_temperature_file(
        fetch_missing=fetch_missing,
        file_pattern=ALEGRETE_FILE_PATTERN
    )

    print("\nArquivo final:")
    print(FINAL_OUTPUT_FILE)
    print(f"Número de registros: {len(final_df)}")

    return final_df


def main_nova_ramada(fetch_missing=False):
    """Gera a série diária usando o ponto GFS de Nova Ramada."""

    output_file = OUTPUT_DIR / "gfs_temperature_20190613_20260813_Nova_Ramada.csv"
    final_df = generate_gfs_temperature_file(
        fetch_missing=fetch_missing,
        output_file=output_file,
        file_pattern=NOVA_RAMADA_FILE_PATTERN
    )

    print("\nArquivo final de Nova Ramada:")
    print(output_file)
    print(f"Número de registros: {len(final_df)}")

    return final_df

if __name__ == "__main__":

    main()
