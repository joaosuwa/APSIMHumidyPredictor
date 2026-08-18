# Dataset de treinamento

Arquivo gerado: `data/processed/model/training_dataset.csv`.

Cada linha representa o dia `D` de um cenário e ciclo de cultivo de milho. O
target usado na modelagem é a variação do déficit no dia seguinte (`D+1`), no
mesmo cenário e ciclo. O déficit absoluto de `D+1` também é mantido para
reconstrução e avaliação das previsões.
A última linha de cada ciclo não entra no arquivo porque não possui observação
de `D+1`. Linhas com alguma variável necessária ausente também são removidas.

O cálculo de ETo foi deliberadamente deixado fora desta versão. Portanto, o
dataset não possui coluna de ETo histórico nem de ETo previsto.

## Variáveis

| Coluna | Fonte | Como foi obtida ou calculada |
| :--- | :--- | :--- |
| `umidade_solo_mm` | APSIM NG | Água no solo integrada às camadas dentro da profundidade radicular (`SW` ponderado pela fração de cada camada na zona radicular). |
| `umidade_solo_passada_1d_mm` | APSIM NG | `umidade_solo_mm` de `D-1`, dentro do mesmo cenário e ciclo. |
| `umidade_solo_passada_2d_mm` | APSIM NG | `umidade_solo_mm` de `D-2`, dentro do mesmo cenário e ciclo. |
| `umidade_solo_passada_3d_mm` | APSIM NG | `umidade_solo_mm` de `D-3`, dentro do mesmo cenário e ciclo. |
| `precipitacao_observada_mm` | NASA POWER | `PRECTOTCORR` diário do local e da data. |
| `irrigacao_aplicada_mm` | APSIM NG | `Irrigation.IrrigationApplied` do dia `D`. |
| `irrigacao_aplicada_dia_posterior_mm` | APSIM NG | Irrigação aplicada em `D+1`, obtida por deslocamento dentro do mesmo cenário e ciclo. É uma variável de manejo futuro e só deve ser usada se o cronograma de `D+1` for conhecido no momento da previsão. |
| `etreal_mm_dia` | APSIM NG | Evaporação do solo + transpiração da planta no dia `D`. |
| `previsao_chuva_24h_mm` | GFS/GDEX | Previsão de chuva acumulada em 24 horas disponível em `D` para o dia seguinte. Foi usada a previsão das 00:00 UTC. |
| `previsao_temperatura_maxima_C` | GFS/GDEX | Máxima diária prevista para `D+1`, consolidada dos quatro intervalos de seis horas. |
| `previsao_temperatura_minima_C` | GFS/GDEX | Mínima diária prevista para `D+1`, consolidada dos quatro intervalos de seis horas. |
| `previsao_radiacao_solar_MJ_m2_dia` | GFS/GDEX | Radiação DSWRF prevista para `D+1`, agregada dos quatro intervalos de seis horas e convertida de W/m² para MJ/m²/dia. |
| `temperatura_media_C` | NASA POWER | `T2M` diário. |
| `temperatura_maxima_C` | NASA POWER | `T2M_MAX` diário. |
| `temperatura_minima_C` | NASA POWER | `T2M_MIN` diário. |
| `umidade_relativa_pct` | NASA POWER | `RH2M` diário. |
| `velocidade_vento_m_s` | NASA POWER | `WS2M` diário. |
| `radiacao_solar_MJ_m2_dia` | NASA POWER | `ALLSKY_SFC_SW_DWN` diário. |
| `dr_mm` | APSIM NG | Depleção de água na zona radicular no dia `D`, calculada como `DUL - SW` nas camadas radiculares. |
| `profundidade_radicular_mm` | APSIM NG | `Maize.Root.Depth` no dia `D`. |
| `gdd_acumulado_C_dia` | APSIM NG + cálculo próprio | Soma desde a semeadura do GDD diário do milho. Usa `Tbase = 8 °C` e `Tupper = 30 °C`; Tmin e Tmax são limitadas ao intervalo antes da média (método 2 do AquaCrop/FAO). |
| `etr_acumulado_mm` | APSIM NG + cálculo próprio | Soma acumulada de `etreal_mm_dia` desde o início do ciclo. |
| `doy_sin` | Cálculo próprio | `sin(2π × DOY / 365)`. |
| `doy_cos` | Cálculo próprio | `cos(2π × DOY / 365)`. |
| `month_sin` | Cálculo próprio | `sin(2π × mês / 12)`. |
| `month_cos` | Cálculo próprio | `cos(2π × mês / 12)`. |
| `escoamento_superficial_mm` | APSIM NG | `Soil.SoilWater.Runoff` do dia `D`. |
| `drenagem_mm` | APSIM NG | `Soil.SoilWater.Drainage` do dia `D`. |
| `taw_mm` | APSIM NG + cálculo próprio | Água total disponível na zona radicular: soma de `DUL - LL` nas camadas radiculares. |
| `dias_desde_semeadura` | APSIM NG | `Maize.DaysAfterSowing`. |
| `deficit_agua_proximo_dia_mm` | APSIM NG | **Target**: `dr_mm` de `D+1`, obtido com `shift(-1)` agrupado por cenário e ciclo de cultivo. |
| `variacao_deficit_proximo_dia_mm` | APSIM NG + cálculo próprio | **Target do modelo**: `deficit_agua_proximo_dia_mm - dr_mm`. |

## Metadados para divisão e auditoria

Os campos abaixo são mantidos no CSV, mas não são entregues aos modelos como
features:

| Coluna | Uso |
| :--- | :--- |
| `data` | Data das features no dia `D`. |
| `data_alvo` | Data do target em `D+1`; deve ser exatamente um dia após `data`. |
| `simulation_name` | Identifica o cenário APSIM. |
| `cycle_id` | Identifica o ciclo dentro de cada cenário. |
| `ano_semeadura` | Define os cortes temporais de treino, validação e teste. |
| `local` | Localidade associada ao cenário (`Alegrete` ou `NovaRamada`). |
| `cenario_irrigado` | Indica se a simulação contém manejo de irrigação programada. |

## Locais e alinhamento temporal

As simulações APSIM com nome iniciado por `Alegrete` usam os arquivos de clima
de Alegrete. As simulações `Simulation*` usam Nova Ramada. Os dados NASA POWER,
GFS e APSIM são unidos pela localidade e pela data. As variáveis meteorológicas
do GFS são alinhadas como previsão disponível em `D` para `D+1`.

## Referência do GDD

Os parâmetros de milho (`Tbase = 8 °C` e `Tupper = 30 °C`) e o método de
limitação de Tmin/Tmax seguem a parametrização de milho do manual AquaCrop da
FAO: <https://www.fao.org/fileadmin/user_upload/faowater/docs/Annexes.pdf>.
