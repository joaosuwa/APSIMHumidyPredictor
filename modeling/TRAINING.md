# Treinamento pela variação do déficit hídrico

Este documento descreve integralmente o pipeline executado por
`modeling/main.py`: target, dados, validação, Optuna, persistência, métricas e
limitações. A implementação anterior, que aprendia diretamente o déficit
absoluto de `D+1`, foi preservada apenas nos artefatos históricos em
`modeling/artifacts/default`.

## Problema e targets

Cada linha contém informações disponíveis no dia `D`. O pipeline aprende a
mudança do déficit até o dia seguinte:

```text
variacao_deficit_proximo_dia_mm = deficit_agua_proximo_dia_mm - dr_mm
```

As interfaces oficiais são:

```text
NEXT_DEFICIT_COLUMN = deficit_agua_proximo_dia_mm
VARIATION_TARGET_COLUMN = variacao_deficit_proximo_dia_mm
TARGET_COLUMN = VARIATION_TARGET_COLUMN
```

O CSV mantém os dois targets. A preparação valida a identidade acima com
tolerância numérica de `1e-9`; qualquer divergência interrompe o pipeline. Os
dois targets, metadados e colunas diretas de irrigação ficam fora das features.

O conjunto oficial possui 35 features. Além das 32 variáveis anteriores, o
Plano 7 inclui três previsões GFS disponíveis em `D` para `D+1`:

- `umidade_relativa_prevista_pct`;
- `ponto_orvalho_previsto_C`;
- `velocidade_vento_prevista_2m_m_s`.

O vento é calculado combinando as componentes U/V previstas a 10 m por
`sqrt(U² + V²)` e convertendo o módulo para 2 m pela equação 47 da FAO-56. A
preparação rejeita umidade fora de `[0, 100]`, vento negativo ou não finito e
ponto de orvalho não finito. Não há imputação silenciosa.

O CSV também guarda `precipitacao_observada_dia_posterior_mm`, calculada por
`shift(-1)` dentro do mesmo `SimulationName` e `cycle_id`. Ela serve somente
para auditoria e diagnóstico de erro após a seleção do campeão. A coluna é
explicitamente excluída de `MODELING_FEATURE_COLUMNS`, portanto nem o Optuna,
nem o early stopping, nem o treino final têm acesso à chuva realizada em D+1.

`dr_mm` permanece como feature. Ao contrário do target absoluto, sua correlação
linear com a variação é baixa. Ainda assim, representa o estado hídrico
disponível em `D` e pode interagir com chuva, ETo, TAW e profundidade radicular.
Não há seleção de features, normalização, clipping ou imputação.

Depois de prever a variação, o déficit operacional é reconstruído por:

```text
deficit_previsto_D+1 = dr_mm_D + variacao_prevista_D+1
```

Os erros de MAE, RMSE e bias são numericamente iguais nas escalas residual e
absoluta, pois a mesma constante `dr_mm` é somada ao observado e ao previsto.
O R² é diferente porque a variância dos dois targets é diferente.

## Dados e isolamento do teste

São usadas somente as simulações não irrigadas `Alegrete`, `AlegreteClay`,
`Simulation` e `SimulationClay`:

| Parte | cycle_id | Linhas | Uso |
| --- | --- | ---: | --- |
| Desenvolvimento | `0–5` | 3.422 | tuning, seleção e treino final |
| Teste final | `6` | 614 | avaliação pós-seleção |

O desenvolvimento usa `LeaveOneGroupOut` por `cycle_id`. Cada um dos seis folds
valida o mesmo identificador completo nas quatro simulações e treina nos outros
cinco. Cada linha recebe exatamente uma previsão out-of-fold (OOF).

As funções de otimização recebem somente o DataFrame de desenvolvimento. O
campeão, seus parâmetros e os três modelos finais são definidos antes do
primeiro acesso ao teste. Métricas históricas também só são lidas depois da
nova avaliação, portanto não influenciam a seleção.

## Loss, objetivo do Optuna e critério de seleção

Para cada trial, os seis folds produzem previsões da variação. O valor
minimizado é o RMSE global OOF, ponderado naturalmente pelo número de linhas:

```text
RMSE_OOF = sqrt(soma((variacao_real - variacao_prevista)²) / 3422)
```

Após cada fold, o RMSE acumulado por linha é reportado ao `MedianPruner`. Não é
usada uma média simples dos RMSEs de fold, que daria o mesmo peso a folds de
tamanhos diferentes. O sampler é
`TPESampler(seed=42, n_startup_trials=10)` e o pruner começa depois dos dez
trials iniciais completos, com dois folds de aquecimento. Trials completos,
podados ou com falha contam para o orçamento total.

O padrão é 20 trials por algoritmo. É uma busca mais econômica que os 50 trials
da abordagem histórica e pode encontrar um ótimo local inferior. A execução é
sequencial por trial (`n_jobs=1`); cada biblioteca pode usar threads internos.

## Modelos e espaços de busca

Os espaços são mantidos iguais aos da abordagem absoluta para isolar o efeito
da troca de target. Configurações fixas:

| Modelo | Loss/objective | Máximo | Early stopping |
| --- | --- | ---: | ---: |
| XGBoost | `reg:squarederror`, avaliação `rmse` | 3.000 árvores | 100 rounds |
| LightGBM | regressão L2, avaliação `rmse` | 3.000 árvores | 100 rounds |
| CatBoost | `RMSE` | 3.000 árvores | 100 rounds |

Assim, erros grandes influenciam a construção das árvores, o early stopping,
a escolha de hiperparâmetros e a escolha do algoritmo. Usar RMSE somente no
ranking final não produziria o mesmo aprendizado. A CLI aceita
`--metric {mae,rmse}` para experimentos controlados, mas o padrão deste plano é
`rmse`; a métrica entra no fingerprint e gera um estudo Optuna incompatível com
o estudo MAE anterior.

Parâmetro comum: `learning_rate` entre `0.01` e `0.20`, em escala logarítmica.

XGBoost busca `max_depth` (`3–10`), `min_child_weight` (`1–20`, log),
`subsample` e `colsample_bytree` (`0.60–1.00`), `gamma` (`1e-8–10`, log),
`reg_alpha` (`1e-8–10`, log) e `reg_lambda` (`1e-3–100`, log).

LightGBM busca `num_leaves` (`15–255`), `max_depth` (`3–12`),
`min_child_samples` (`10–100`), `subsample` e `colsample_bytree`
(`0.60–1.00`), `reg_alpha` (`1e-8–10`, log) e `reg_lambda`
(`1e-3–100`, log).

CatBoost usa bootstrap bayesiano e busca `depth` (`4–10`), `l2_leaf_reg`
(`1–30`, log), `random_strength` (`1e-3–10`, log) e
`bagging_temperature` (`0–10`).

## Early stopping e treino final

Em cada fold, somente a validação daquele fold controla early stopping. O
melhor trial é repetido nos seis folds para gerar as previsões OOF definitivas.
O número final de árvores é a mediana arredondada das seis melhores iterações.
Depois, cada modelo é treinado em todas as 3.422 linhas sem early stopping.

O campeão é o algoritmo com menor RMSE OOF da variação. Todos os três modelos
são persistidos e avaliados; resultados de teste não podem trocar o campeão.

## Baseline e métricas

A baseline prevê variação zero:

```text
variacao_prevista = 0
deficit_previsto_D+1 = dr_mm_D
```

Para variação e déficit reconstruído são gravados RMSE, MAE, R² e bias. Também
são calculados:

- `direction_accuracy`: acerto entre redução (`variação < 0`) e não redução;
- `reduction_precision`: precisão ao prever redução;
- `reduction_recall`: cobertura das reduções reais.

`cv_event_metrics.csv` e `test_event_metrics.csv` detalham MAE, RMSE, R², bias e
acerto direcional em três regimes:

- `increase_or_stable`: variação maior ou igual a zero;
- `reduction`: variação negativa;
- `large_reduction`: variação igual ou inferior a `-5 mm`.

Esses recortes são apenas diagnósticos. O campeão continua sendo escolhido
exclusivamente pelo RMSE OOF global.

### Erro condicionado à chuva observada

Para entender o comportamento justamente nos dias em que o déficit tende a
cair, `rain_error_metrics.csv` divide desenvolvimento OOF e teste em:

| Faixa | Precipitação observada em D+1 |
| --- | --- |
| `sem_chuva` | `≤ 0,1 mm` |
| `fraca` | `(0,1, 5] mm` |
| `moderada` | `(5, 20] mm` |
| `forte` | `> 20 mm` |

Cada modelo e a baseline recebem contagem, MAE, RMSE e bias em cada faixa. O
resíduo usado nos gráficos é `variação prevista - variação real`. Durante a
chuva, um resíduo positivo normalmente significa que o modelo previu uma queda
menos intensa e, portanto, subestimou a redução do déficit.

## Estrutura do código

```text
modeling/
  config.py          configurações públicas
  data.py            carregamento, validação, holdout e folds
  main.py            orquestrador e CLI
  pipeline/
    artifacts.py     parâmetros e JSONs
    evaluation.py    métricas e eventos
    models.py        factories, ajuste e modelos nativos
    optimization.py  Optuna, OOF e treino final
    plots.py         visualizações, chuva e permutation importance
  tests/
  TRAINING.md
```

A pasta antiga de caches compilados foi removida.
As interfaces públicas continuam disponíveis pelo pacote `modeling`.

## Artefatos

A nova abordagem usa `modeling/artifacts/variation_target`; o diretório
histórico `modeling/artifacts/default` não é alterado.

- `models/`: `xgboost.ubj`, `lightgbm.txt` e `catboost.cbm`;
- `parameters/`: um JSON por modelo e `champion.json`;
- `studies/optuna.sqlite3`: estudos retomáveis;
- `results/`: métricas, previsões OOF/teste, eventos, chuva, importâncias e manifesto;
- `plots/`: comparação, folds, tuning, resíduos, chuva, importâncias e observado versus previsto.

Cada arquivo de parâmetros registra target, estudo, hiperparâmetros, métricas
OOF, melhores iterações por fold e iteração final. `champion.json` referencia os
três arquivos. O manifesto inclui hash do dataset, configuração, versões,
caminhos de modelos/parâmetros e a regra de reconstrução.

As tabelas de previsões contêm, para cada modelo e baseline:

```text
predicted_variation_<modelo>
predicted_next_deficit_<modelo>
```

`historical_comparison.csv` compara o teste novo com os artefatos absolutos
somente depois da nova seleção. Essa tabela não participa do aprendizado.

`rain_error_metrics.csv`, `test_rmse_by_observed_rain.png` e
`test_residuals_by_observed_rain.png` registram o desvio condicionado à chuva
real. Há um CSV e um gráfico de permutation importance para cada um dos três
modelos, além de `permutation_importance_comparison.png`. A importância é
calculada no teste como o aumento do RMSE após dez permutações da feature. Ela
é diagnóstica: não altera modelo, parâmetros ou campeão.

## Fingerprint e retomada

O nome de cada estudo deriva do SHA-256 do CSV, features, target, simulações,
ciclos de teste, seed, early stopping, máximo de árvores, métrica do objetivo e
versão do espaço de busca. Alterar features, métrica ou regenerar o CSV cria
estudos incompatíveis separados.
Reexecutar a mesma configuração agenda apenas trials suficientes para atingir o
total solicitado.

## Execução

```bash
python -m pip install -r requirements-modeling.txt
python -m scripts.model_dataset
python -m pytest modeling/tests -q
python modeling/main.py --trials 20 --seed 42 --metric rmse
```

Para um smoke test ou destino alternativo:

```bash
python modeling/main.py --trials 1 --metric rmse --output-dir modeling/artifacts/smoke
```

## Limitações e próximos experimentos

A troca de target remove o atalho de reconstruir diretamente um valor altamente
correlacionado com `dr_mm`. As novas previsões adicionam informação
meteorológica, mas não corrigem um forecast de chuva pouco informativo. No
desenvolvimento, a variação tem correlação `-0,835` com a chuva observada em
`D+1`; a previsão de chuva tem correlação de apenas `0,136` com a chuva
realizada e `-0,047` com a variação. O ponto de orvalho previsto apresenta
correlação marginal aproximada de `-0,220` com a variação no desenvolvimento;
umidade e vento têm relações marginais menores. Por isso, o principal limite
provável continua sendo a qualidade ou o alinhamento do forecast de chuva.

O RMSE pressiona o ajuste sobre eventos com erro alto, incluindo quedas grandes
durante chuva moderada ou forte, mas pode piorar o desempenho em dias secos. Ele
também não força artificialmente `previsao_chuva_24h_mm` a ganhar importância:
uma árvore só usará essa feature se ela reduzir o erro nos folds históricos.

Próximos experimentos, em ordem:

1. auditar horizonte, data de inicialização e qualidade do GFS;
2. criar features de balanço hídrico, `dr/taw`, água disponível, variação de
   umidade e chuva recente acumulada;
3. testar classificador de redução seguido de regressão de magnitude;
4. testar pesos ou loss assimétrica para reduções intensas, se o custo
   operacional justificar;
5. executar ablação com e sem `dr_mm` usando os mesmos folds.

Esses experimentos não são combinados nesta versão para que o efeito do novo
target possa ser medido isoladamente.
