# Previsor de deficit hidrico

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
| ET0 | Meteorológica derivada | Será calculada em etapa posterior; não está no CSV atual | Etapa posterior |
| ETreal | Meteorológica derivada | Quanto de água foi consumido solo + planta (Transpiração + evaporação) | APSIM NG |
| Previsão de chuva | Meteorológica | - | [NCAR GDEX](https://gdex.ucar.edu/datasets/d084001/) |
| Previsão de ET0 | Meteorológica derivada | Será calculada em etapa posterior; não está no CSV atual | Etapa posterior |
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

ET0 = Fórmula Priestley-Taylor (cálculo reservado para etapa posterior; não
entra no dataset atual)

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

Previsão de chuva, temperatura máxima e mínima previstas e radiação prevista.
O cálculo de ETo foi deixado para uma etapa posterior.

## Fonte dos dados coletados:

### GFS Archive:
Chuva acumulada futura (24h), temperatura futura (0h-6h,6h-12h,12h-18h,18h,24h) e radiação futura (0h-6h,6h-12h,12h-18h,18h,24h).

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