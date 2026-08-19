# Previsor de deficit hidrico

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

## Ideia do projeto:

Construir um modelo de aprendizado de máquina que utilize dados provenientes de clima, simulações no simulador APSINM NG e previsões de chuva históricas para prever o déficit hídrico do solo. O modelo de déficit Hídrico futuro vai ser utilizado para tomar uma decisão de irrigação: Se déficit de hoje cair em relação ao RAW (irrigar até o CC), se não: 50% do Dr.

## Variáveis para o modelo:

| Variável | Tipo | Observação | Como obter |
| :--- | :--- | :--- | :--- |
| Umidade do solo | Solo | Umidade do solo atual (Água disponível na zona radicular) | APSIM NG |
| Umidade do solo passada | Série temporal | Umidade de dias anteriores | APSIM NG |
| Precipitação observada | Meteorológica | Principal entrada de água | INMET |
| Irrigação aplicada no dia | Manejo | Fundamental para o modelo entender o efeito da irrigação | APSIM NG |
| Irrigação aplicada no dia posterior | Manejo | Fundamental para o modelo entender o efeito da irrigação | APSIM NG |
| ETo prevista | Meteorológica derivada | ETo de referência para `D+1`, calculada pelo método FAO-56 Penman--Monteith | GFS + PyETo |
| ETreal | Meteorológica derivada | Quanto de água foi consumido solo + planta (Transpiração + evaporação) | APSIM NG |
| Previsão de chuva | Meteorológica | - | [NCAR GDEX](https://gdex.ucar.edu/datasets/d084001/) |
| Previsão de ETo | Meteorológica derivada | Usa temperatura, radiação, vento, umidade e ponto de orvalho previstos para `D+1` | GFS + PyETo |
| Temperatura média | Meteorológica | Influencia evapotranspiração | INMET |
| Temperatura máxima/mínima | Meteorológica | Pode representar melhor extremos térmicos | INMET |
| Umidade relativa | Meteorológica | Importante para a demanda evaporativa | INMET |
| Velocidade do vento | Meteorológica | Entra no cálculo de ET0 | INMET |
| Radiação solar | Meteorológica | Entra no cálculo de ET0 e representa energia disponível | INMET |
| Depleção de água no solo (Dr) | Balanço hídrico | Excelente variável de estado | Simulador (DUL - SW) até zona radicular |
| Profundidade radicular Zr | Estado/parâmetro | Pode variar durante o ciclo | APSIM NG oferece / GDD acumulado |
| GDD Acumulado | Meteorológica | Ajuda a calcular o Kc do cultivar | Cálculo próprio |
| ETr Acumulado | Meteorológica | Outra variável importante para constatar o estágio da planta | Cálculo próprio |
| Dia do ano | Meteorológica | - | Cálculo próprio |
| Mês do ano | Meteorológica | - | Cálculo próprio |
| Escoamento superficial de água | Solo | - | APSIM NG |
| Drenagem | Solo | - | APSINM NG |
| Água Total Disponível no Solo (TAW) | Solo | Capacidade TOTAL de água disponível no solo | APSINM NG | 
| Dias desde semeadura | Planta | - | APSINM NG |

Para o milho, o GDD usa `Tbase = 8 °C` e `Tupper = 30 °C`. As temperaturas
mínima e máxima são limitadas a esses valores antes da média diária (método 2
do AquaCrop/FAO). A referência é o manual oficial da FAO/AquaCrop:
<https://www.fao.org/fileadmin/user_upload/faowater/docs/Annexes.pdf>.


## Como serão obtidos os dados:

### Variáveis report no APSINM NG:

Irrigação aplicada, Irrigação aplicada no dia posterior, Profundidade radicular Zr, Escoamento superficial de água e Drenagem

### Cálculos feitos usando dados do APSINM NG:

Umidade do solo, Umidade do solo histórica = SW até zona radicular.

ETreal = [Soil].SoilWater.Es (Evaporação do solo) + [Plant].Leaf.Transpiration (Transpiração da planta)

Depleção de água no solo (Dr) = DUL - SW de todas as camadas até zona radicular (Cálculo já feito no simulador)

ETo prevista = método FAO-56 Penman--Monteith usando PyETo. A temperatura
máxima/mínima e a radiação são obtidas dos produtos GFS já consolidados. U GRD
e V GRD são combinados para obter a velocidade do vento e convertidos de 10 m
para 2 m. O DPT é convertido de Kelvin para Celsius e usado preferencialmente
para obter a pressão real de vapor; a umidade relativa é usada como fallback.
Na escala diária, o fluxo de calor do solo é considerado zero. A pressão
atmosférica é estimada pela altitude configurada para cada local.

Dia do ano =
$$DOY_{\sin} = \sin\left(2\pi \frac{DOY}{365}\right)$$
$$DOY_{\cos} = \cos\left(2\pi \frac{DOY}{365}\right)$$

Mês do ano =
$$Month_{\sin} = \sin\left(2\pi \frac{Month}{12}\right)$$
$$Month_{\cos} = \cos\left(2\pi \frac{Month}{12}\right)$$

Água Total Disponível no Solo (TAW) = (DUL - LL)

### NASA-POWER | LocalisAgro:

Temperatura média, Temperatura máxima/mínima, Umidade relativa, Velocidade do vento, Radiação solar

OBS: INMET contém dados nulos (ausentes). NASA-POWER contém todos os dados no período analisado (2019-2026)

### GFS Archive (Dados disponíveis desde 13/06/2019):

Previsão de chuva, temperatura máxima e mínima previstas, radiação prevista,
componentes U/V do vento, umidade relativa e ponto de orvalho. A temperatura e
a radiação são montadas a partir dos intervalos `0–6`, `6–12`, `12–18` e
`18–24` horas da mesma inicialização. O intervalo terminado às `00:00` é
associado ao dia da inicialização anterior, sem outro deslocamento. A PCP,
U/V, RH e DPT são produtos de horizonte de 24 horas: o registro de `00:00` é
deslocado um dia para trás para recuperar a inicialização em `D`. A ETo
prevista é calculada com FAO-56/PyETo.

## Fonte dos dados coletados:

### GFS Archive:
Chuva acumulada futura (24h), temperatura futura (0h-6h, 6h-12h, 12h-18h,
18h-24h), radiação futura (0h-6h, 6h-12h, 12h-18h, 18h-24h), U/V do vento a
10 m, umidade relativa e ponto de orvalho a 2 m.

### NASA POWER:
Dados históricos de clima

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
