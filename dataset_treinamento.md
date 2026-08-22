# Dataset de treinamento

Arquivo gerado: `data/processed/model/training_dataset.csv`.

Cada linha representa o dia `D` de um cenário e ciclo de cultivo de milho. O
target de treinamento é a variação do déficit de água até o dia seguinte
(`D+1`), no mesmo cenário e ciclo. O déficit absoluto de `D+1` também é
preservado para auditoria e reconstrução das previsões.
A última linha de cada ciclo não entra no arquivo porque não possui observação
de `D+1`. Linhas com alguma variável necessária ausente também são removidas.

O dataset possui a ETo de referência prevista para `D+1`, calculada a partir
dos forecasts GFS pelo método FAO-56 Penman--Monteith usando PyETo. Umidade
relativa, ponto de orvalho e velocidade do vento prevista a 2 m também são
preservados como features independentes.

As colunas `SimulationName`, `Clock_today`, `sowing_date` e `cycle_id` são metadados mantidos
para auditoria e cortes temporais. Elas não fazem parte das features usadas
para treinar o modelo.

As colunas `irrigacao_aplicada_mm` e
`irrigacao_aplicada_dia_posterior_mm` continuam no CSV central para auditoria e
experimentos futuros, mas não fazem parte das features da configuração atual em
`modeling`. Essa configuração usa somente cenários não irrigados.

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
| `chuva_irrigacao_passada_1d_mm` | APSIM NG | Soma de chuva e irrigação aplicada em `D-1`, deslocada dentro do mesmo cenário e ciclo. |
| `chuva_irrigacao_passada_2d_mm` | APSIM NG | Soma de chuva e irrigação aplicada em `D-2`, deslocada dentro do mesmo cenário e ciclo. |
| `chuva_irrigacao_passada_3d_mm` | APSIM NG | Soma de chuva e irrigação aplicada em `D-3`, deslocada dentro do mesmo cenário e ciclo. |
| `etreal_mm_dia` | APSIM NG | Evaporação do solo + transpiração da planta no dia `D`. |
| `previsao_chuva_24h_mm` | GFS/GDEX | Previsão de chuva acumulada em 24 horas disponível em `D` para o dia seguinte. Foi usada a previsão das 00:00 UTC. |
| `previsao_temperatura_maxima_C` | GFS/GDEX | Máxima diária prevista para `D+1`, consolidada dos quatro intervalos de seis horas. |
| `previsao_temperatura_minima_C` | GFS/GDEX | Mínima diária prevista para `D+1`, consolidada dos quatro intervalos de seis horas. |
| `previsao_radiacao_solar_MJ_m2_dia` | GFS/GDEX | Radiação DSWRF prevista para `D+1`, agregada dos quatro intervalos de seis horas e convertida de W/m² para MJ/m²/dia. |
| `umidade_relativa_prevista_pct` | GFS/GDEX | Umidade relativa prevista em `D` para `D+1`, limitada fisicamente ao intervalo de 0% a 100%. |
| `ponto_orvalho_previsto_C` | GFS/GDEX | Ponto de orvalho previsto em `D` para `D+1`, convertido de Kelvin para graus Celsius durante a consolidação. |
| `velocidade_vento_prevista_2m_m_s` | GFS/GDEX + cálculo FAO-56 | Módulo do vento previsto a 10 m, `sqrt(U² + V²)`, convertido para 2 m pela equação 47 da FAO-56. |
| `previsao_eto_mm_dia` | GFS/GDEX + PyETo | ETo diária FAO-56 Penman--Monteith prevista para `D+1`, calculada com temperatura, radiação, vento, umidade e ponto de orvalho previstos pelo GFS. |
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
| `precipitacao_observada_dia_posterior_mm` | NASA POWER + alinhamento temporal | Chuva observada em `D+1`, obtida por `shift(-1)` dentro do mesmo cenário e ciclo. É somente auditoria pós-seleção e nunca entra nas features. |
| `deficit_agua_proximo_dia_mm` | APSIM NG | Déficit absoluto em `D+1`: `dr_mm` deslocado com `shift(-1)` dentro do mesmo cenário e ciclo. É preservado para auditoria e avaliação reconstruída. |
| `variacao_deficit_proximo_dia_mm` | APSIM NG + cálculo próprio | **Target de treinamento**: `deficit_agua_proximo_dia_mm - dr_mm`, isto é, a mudança do déficit entre `D` e `D+1`. |

## Metadados preservados

| Coluna | Fonte | Como foi obtida |
| :--- | :--- | :--- |
| `SimulationName` | APSIM NG | Nome original da simulação APSIM. |
| `Clock_today` | APSIM NG | `Clock.Today` convertido para a data ISO `YYYY-MM-DD`. |
| `sowing_date` | APSIM NG | Data de semeadura do ciclo, convertida para a data ISO `YYYY-MM-DD`. |
| `cycle_id` | APSIM NG + cálculo próprio | Identificador do ciclo, criado a partir das mudanças de `Maize.SowingDate` dentro de cada simulação. |

## Locais e alinhamento temporal

As simulações APSIM com nome iniciado por `Alegrete` usam os arquivos de clima
de Alegrete. As simulações `Simulation*` usam Nova Ramada. Os dados NASA POWER,
GFS e APSIM são unidos pela localidade e pela data. As variáveis meteorológicas
do GFS são alinhadas como previsão disponível em `D` para `D+1`.

Os produtos GFS de horizonte de 24 horas (`A PCP`, `U GRD`, `V GRD`, `R H` e
`DPT`) mantêm somente o registro de `00:00`. Como a data registrada no CSV
representa o final do horizonte, a data de inicialização é calculada como
`Date - 1 dia`. Assim, `6/14/2019,0:00` representa a inicialização de
`2019-06-13`.

Tmax, Tmin e DSWRF são diferentes: cada dia é formado pelos quatro produtos de
seis horas `0–6`, `6–12`, `12–18` e `18–24`. Os três primeiros terminam às
`06:00`, `12:00` e `18:00`; o último termina às `00:00` do dia seguinte e é
associado ao mesmo dia de inicialização. Não há um segundo deslocamento depois
dessa consolidação. Dias sem os quatro intervalos são descartados.

O pipeline grava todas essas variáveis em um único arquivo por localidade:
`gfs_daily_forecast_Alegrete.csv` ou `gfs_daily_forecast_Nova_Ramada.csv`.

Para a ETo, U/V são combinados em velocidade do vento a 10 m e convertidos
para 2 m. O DPT em Kelvin é convertido para Celsius e usado preferencialmente
para calcular a pressão real de vapor; RH é usado como fallback. A pressão
atmosférica é estimada pela altitude configurada para cada localidade:

- Alegrete: 102 m;
- Nova Ramada: 511 m.

## Referência do GDD

Os parâmetros de milho (`Tbase = 8 °C` e `Tupper = 30 °C`) e o método de
limitação de Tmin/Tmax seguem a parametrização de milho do manual AquaCrop da
FAO: <https://www.fao.org/fileadmin/user_upload/faowater/docs/Annexes.pdf>.
