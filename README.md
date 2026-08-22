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

Os caminhos são definidos em `scripts/paths.py`. As operações genéricas de CSV ficam em `scripts/data_io.py`. O processamento específico foi separado por fonte: APSIM NG em `scripts/apsim/processing.py`, NASA POWER em `scripts/nasa_power/processing.py`, GFS em `scripts/gfs_data_extractor/gfs_data_processing.py` e métricas estatísticas em `scripts/metrics/forecast.py`.

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

Para baixar os produtos GFS, escolha as localidades e produtos na CLI. O grupo
`forecast_24h` contém A PCP, U/V, RH e DPT:

```bash
python -m scripts.gfs_data_extractor.get_gfs_data --sites alegrete nova_ramada --products forecast_24h
```

É possível baixar somente chuva com `--products apcp_24` e sobrescrever o
período usando `--start-date` e `--end-date` no formato `YYYYMMDDHHMM`. O
comando somente faz requisições e grava os RAW em
`data/raw/gfs/downloads/<produto>`.

Para processar os downloads locais e gerar um arquivo diário por localidade:

```bash
python -m scripts.gfs_data_extractor.gfs_data_processing --sites alegrete nova_ramada
```

Esse comando não faz requisições externas e gera
`gfs_daily_forecast_Alegrete.csv` e `gfs_daily_forecast_Nova_Ramada.csv` em
`data/processed/gfs`.


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

### Preparação e treinamento dos modelos

Instale as dependências e execute o pipeline completo:

```bash
python -m pip install -r requirements-modeling.txt
python modeling/main.py --trials 50 --seed 42
```

A configuração tipada em `modeling/config.py` seleciona as simulações e os
`cycle_id` reservados para teste. O restante é dividido por
`LeaveOneGroupOut`: cada fold valida um ciclo completo nas quatro simulações e
treina nos demais ciclos. Por padrão, o ciclo `6` permanece intocado no teste e
os ciclos `0–5` formam seis folds de cross-validation. As colunas diretas de
irrigação permanecem no CSV para auditoria, mas ficam fora das features desta
configuração não irrigada.

O treinamento compara XGBoost, LightGBM e CatBoost. Os hiperparâmetros são
otimizados com Optuna pelo MAE das previsões out-of-fold dos ciclos `0–5`; o
ciclo `6` só é avaliado depois da escolha do campeão. Modelos, métricas,
previsões e gráficos são gravados em `modeling/artifacts/default`.

A lógica completa, os espaços de busca, as regras contra vazamento e a lista
de artefatos estão documentados em [`modeling/TRAINING.md`](modeling/TRAINING.md).
