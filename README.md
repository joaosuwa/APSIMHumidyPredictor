# Previsor de deficit hidrico

## Ideia do projeto:

Construir um modelo de aprendizado de máquina que utilize dados provenientes de clima, simulações no simulador APSINM NG e previsões de chuva históricas para prever o déficit hídrico do solo. O modelo de déficit Hídrico futuro vai ser utilizado para tomar uma decisão de irrigação: Se déficit de hoje cair em relação ao RAW (irrigar até o CC), se não: 50% do Dr.

## Organização do repositório

Os dados ficam separados em duas áreas:

- `data/raw/apsim`: arquivos `Report.csv` originais do APSIM;
- `data/raw/nasa_power`: dados meteorológicos originais da NASA POWER;
- `data/raw/gfs`: previsões e downloads brutos do GFS/GDEX;
- `data/processed/apsim`, `data/processed/nasa_power` e `data/processed/gfs`: dados transformados por fonte;
- `data/processed/validation`: resultados da validação das previsões;
- `data/processed/model`: espaço reservado para o CSV central que será usado no treinamento do modelo.

Os caminhos são definidos em `scripts/paths.py`. As operações genéricas de CSV ficam em `scripts/data_io.py`. O processamento específico foi separado por fonte: APSIM NG em `scripts/apsim/processing.py`, NASA POWER em `scripts/nasa_power/processing.py`, GFS em `scripts/gfs_data_extractor/` (incluindo a agregação de forecasts em `scripts/gfs_data_extractor/forecast.py`) e métricas estatísticas em `scripts/metrics/forecast.py`.

Para processar o Report do APSIM:

```bash
python -m scripts.feature_engineering
```

O módulo `scripts.feature_engineering` continua disponível como ponto de entrada compatível, enquanto as funções podem ser importadas diretamente de `scripts.apsim` e `scripts.nasa_power`.

Para validar as previsões de chuva:

```bash
python -m scripts.forecast_validation
```

O arquivo de validação ficou responsável pela leitura das fontes e pela orquestração. As métricas são calculadas em `scripts.metrics.forecast`.

Para baixar os produtos GFS, execute `python -m scripts.gfs_data_extractor.get_gfs_data`. Esse comando somente faz requisições e grava os downloads brutos em `data/raw/gfs/downloads`.

Para processar os downloads locais e gerar os arquivos consolidados, execute `python -m scripts.gfs_data_extractor.gfs_data_processing`. Esse comando não faz requisições externas e grava os resultados em `data/processed/gfs`.

## Dataset final de treinamento

O comando abaixo gera `data/processed/model/training_dataset.csv`:

```bash
python -m scripts.model_dataset
```

Cada linha contém as variáveis disponíveis no dia `D`. O alvo é
`deficit_agua_proximo_dia_mm`, calculado como `Dr_root` do dia `D+1` no mesmo
cenário e ciclo de cultivo. A última linha de cada ciclo é removida porque não
tem observação do dia seguinte. As previsões de chuva, temperatura e radiação
são as previsões disponíveis em `D` para `D+1`. O detalhamento completo está em
[`dataset_treinamento.md`](dataset_treinamento.md).