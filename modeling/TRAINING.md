# Treinamento dos modelos de déficit hídrico

Este documento é a especificação técnica do pipeline executado por
`modeling/main.py`. Ele descreve todas as decisões que transformam o dataset
preparado em modelos persistidos e resultados de avaliação reproduzíveis.

## Objetivo da modelagem

O problema é uma regressão supervisionada. Para cada linha do dia `D`, o modelo
prevê o déficit hídrico absoluto da zona radicular no dia seguinte:

```text
target = deficit_agua_proximo_dia_mm = dr_mm em D+1
```

O target não é a variação diária. `dr_mm`, o déficit observado no próprio dia
`D`, é uma feature válida e também sustenta a baseline de persistência. Valores
negativos são preservados porque existem no resultado do APSIM e não há regra
física aprovada no projeto para truncá-los. As previsões também não são
limitadas artificialmente.

São usadas exatamente as 32 colunas declaradas em
`modeling.data.MODELING_FEATURE_COLUMNS`. Não há seleção pela análise de
correlação, normalização, padronização ou imputação. Os três algoritmos são
baseados em árvores e não exigem escala comum. O carregamento falha quando uma
feature obrigatória está ausente; no dataset atual não existem `NaN` ou valores
infinitos nas features selecionadas.

Metadados, target e colunas diretas de irrigação nunca entram em `X`:

- `SimulationName`, `Clock_today`, `sowing_date` e `cycle_id` são metadados;
- `deficit_agua_proximo_dia_mm` é o target;
- `irrigacao_aplicada_mm` e
  `irrigacao_aplicada_dia_posterior_mm` ficam apenas para auditoria.

## Dados e fronteira contra vazamento

A configuração padrão usa somente `Alegrete`, `AlegreteClay`, `Simulation` e
`SimulationClay`. Os mesmos identificadores de ciclo têm significado alinhado
nas quatro simulações.

Depois da filtragem há 4.036 linhas:

| Parte | Ciclos | Linhas | Uso |
| --- | --- | ---: | --- |
| Desenvolvimento | `0–5` | 3.422 | Optuna, escolha do campeão e treino final |
| Teste final | `6` | 614 | Uma avaliação após a escolha do campeão |

O código impõe a fronteira de duas formas:

1. `tune_model` e `cross_validate_parameters` recebem somente o DataFrame de
   desenvolvimento, os folds, as features e o nome do target. Essas funções
   não recebem `PreparedData` nem o DataFrame de teste.
2. `main.py` conclui os três estudos e escolhe o campeão pelo MAE OOF antes de
   acessar `prepared.test` para gerar previsões finais.

As métricas de teste não mudam hiperparâmetros, número de árvores ou campeão.
Todos os modelos são avaliados no teste para comparação transparente, mas o
campo `champion_selected_by_oof_mae` do manifesto continua refletindo apenas a
cross-validation.

## Cross-validation agrupada por ciclo

O desenvolvimento é dividido por `LeaveOneGroupOut`, usando `cycle_id` como
grupo. Há seis folds. Em cada fold, um identificador completo aparece na
validação e os outros cinco aparecem no treino. As quatro simulações daquele
identificador ficam juntas na validação.

Cada linha de desenvolvimento recebe exatamente uma previsão out-of-fold
(OOF). O objetivo de um trial é o MAE calculado sobre a concatenação das 3.422
previsões OOF, e não a média simples dos seis MAEs:

```text
MAE_OOF = soma(|y_i - predição_i|) / 3422
```

Isso pondera os ciclos pelo número real de observações. O pipeline também grava
MAE, RMSE, R², bias e quantidade de árvores de cada fold para diagnosticar
variação entre ciclos.

## Modelos e hiperparâmetros fixos

Os algoritmos candidatos são XGBoost, LightGBM e CatBoost. Todos usam CPU,
seed configurável e a implementação nativa de MAE como loss ou objective.
Optuna executa trials sequencialmente (`n_jobs=1`) para evitar que múltiplos
trials disputem todos os núcleos. A biblioteca do modelo pode usar seus threads
internos durante um ajuste.

Configurações fixas:

| Modelo | Objective/loss | Máximo de árvores | Outros parâmetros |
| --- | --- | ---: | --- |
| XGBoost | `reg:absoluteerror` | 3.000 | `eval_metric=mae`, `tree_method=hist` |
| LightGBM | `l1` | 3.000 | `subsample_freq=1`, `eval_metric=mae` |
| CatBoost | `MAE` | 3.000 | bootstrap bayesiano, sem arquivos auxiliares |

O máximo e o número de rounds de early stopping vêm de `TrainingConfig`. Os
valores padrão são 3.000 e 100, respectivamente.

## Espaços de busca Optuna

O sampler padrão é `TPESampler(seed=42, n_startup_trials=10)`. Os dez primeiros
trials formam a amostra inicial; os seguintes usam TPE. `MedianPruner` começa a
podar depois desses dez trials e só considera pruning a partir do terceiro fold
(`n_warmup_steps=2`). Após cada fold, o trial reporta o MAE acumulado das linhas
OOF já avaliadas.

Parâmetro comum:

| Parâmetro | Intervalo | Escala |
| --- | --- | --- |
| `learning_rate` | `0.01–0.20` | logarítmica |

XGBoost:

| Parâmetro | Intervalo | Escala |
| --- | --- | --- |
| `max_depth` | `3–10` | inteira |
| `min_child_weight` | `1–20` | logarítmica |
| `subsample` | `0.60–1.00` | linear |
| `colsample_bytree` | `0.60–1.00` | linear |
| `gamma` | `1e-8–10` | logarítmica |
| `reg_alpha` | `1e-8–10` | logarítmica |
| `reg_lambda` | `1e-3–100` | logarítmica |

LightGBM:

| Parâmetro | Intervalo | Escala |
| --- | --- | --- |
| `num_leaves` | `15–255` | inteira |
| `max_depth` | `3–12` | inteira |
| `min_child_samples` | `10–100` | inteira |
| `subsample` | `0.60–1.00` | linear |
| `colsample_bytree` | `0.60–1.00` | linear |
| `reg_alpha` | `1e-8–10` | logarítmica |
| `reg_lambda` | `1e-3–100` | logarítmica |

CatBoost:

| Parâmetro | Intervalo | Escala |
| --- | --- | --- |
| `depth` | `4–10` | inteira |
| `l2_leaf_reg` | `1–30` | logarítmica |
| `random_strength` | `1e-3–10` | logarítmica |
| `bagging_temperature` | `0–10` | linear |

O orçamento padrão é de 50 trials por modelo. Trials `COMPLETE`, `PRUNED` ou
`FAIL` contam no total solicitado. Trials podados economizam ajustes dos folds
restantes, mas não podem vencer o estudo. Como os primeiros dez não são
podados, cada estudo terá trials completos para comparação.

## Early stopping e número final de árvores

Cada ajuste de fold pode criar até 3.000 árvores. A validação daquele fold é
passada somente para early stopping e cálculo da previsão OOF. Se o MAE não
melhorar por 100 rounds, o algoritmo encerra o ajuste e registra sua melhor
iteração.

Depois do estudo, o melhor conjunto de hiperparâmetros é executado novamente
nos seis folds com a mesma seed. Essa execução produz os CSVs OOF definitivos e
seis contagens de melhores iterações. O número usado no treino final é:

```text
iterações_finais = arredondar(mediana(melhores_iterações_dos_6_folds))
```

A mediana reduz a influência de um ciclo excepcionalmente curto ou difícil. O
modelo final é ajustado em todas as 3.422 linhas, com esse número fixo de
árvores e sem early stopping, pois não existe validação separada depois que todo
o desenvolvimento é reunido.

## Seleção, baseline e métricas

O campeão é o algoritmo com menor `MAE_OOF`. A baseline de persistência não é
um modelo candidato e não muda essa seleção; ela serve para verificar se o
aprendizado de máquina supera a regra simples:

```text
predição_persistência(D+1) = dr_mm(D)
```

No desenvolvimento, a baseline é avaliada sobre as mesmas 3.422 linhas. No
teste, ela é avaliada nas mesmas 614 linhas dos três modelos. Se a baseline
superar o campeão no teste, o pipeline registra isso honestamente em
`baseline_outperformed_champion_on_test`; não troca o campeão retroativamente.

As métricas são:

- MAE: erro absoluto médio em milímetros e critério principal;
- RMSE: raiz do erro quadrático médio, mais sensível a erros grandes;
- R²: fração da variância explicada, podendo ser negativa;
- bias: média de `previsto - observado`; positivo indica superestimação.

## Persistência e carregamento

Todos os modelos finais são mantidos, não apenas o campeão:

- XGBoost: `models/xgboost.ubj`;
- LightGBM: `models/lightgbm.txt`;
- CatBoost: `models/catboost.cbm`.

São formatos nativos para evitar depender da serialização interna de objetos
Python. O carregamento é feito por `modeling.load_trained_model`. O consumidor
deve fornecer um DataFrame com as features na ordem salva no campo `features`
do manifesto.

## Estudos, fingerprint e retomada

Os três estudos ficam no mesmo banco `studies/optuna.sqlite3`. O nome contém um
fingerprint derivado de:

- SHA-256 do CSV;
- lista e ordem das features;
- target;
- simulações e ciclos de teste;
- seed;
- máximo de iterações e early stopping;
- versão explícita do espaço de busca.

Se esses itens não mudarem, uma nova execução retoma o estudo e agenda apenas o
número necessário para alcançar `--trials`. Aumentar de 20 para 50 continua o
mesmo estudo. Alterar dados ou uma decisão de treinamento cria automaticamente
outro nome de estudo e impede a mistura de trials incompatíveis. O número de
trials não participa do fingerprint porque representa apenas o orçamento.

## Artefatos gerados

O diretório padrão é `modeling/artifacts/default` e não é versionado no Git.

`results/` contém:

- `optuna_trials_<modelo>.csv`: histórico completo de cada estudo;
- `best_params.json`: parâmetros, MAE OOF e iterações escolhidas;
- `cv_fold_metrics.csv`: métricas e árvores por modelo e fold;
- `cv_metrics.csv`: métricas OOF agregadas e baseline;
- `oof_predictions.csv`: metadados, observado e todas as previsões OOF;
- `test_metrics.csv`: métricas finais dos três modelos e baseline;
- `test_predictions.csv`: 614 observações e previsões finais;
- `permutation_importance.csv`: importância do campeão no teste;
- `manifest.json`: configuração, hashes, versões, campeão e caminhos.

`plots/` contém:

- comparação OOF e MAE por fold;
- comparação no teste;
- observado versus previsto e resíduos do campeão;
- histórico de otimização Optuna;
- permutation importance das 20 principais features do campeão.

A permutation importance usa o teste somente depois da avaliação final. É um
diagnóstico pós-seleção e não alimenta tuning ou escolha de features.

## Execução

Na raiz do projeto:

```bash
python -m pip install -r requirements-modeling.txt
python -m pytest modeling/tests -q
python modeling/main.py --trials 50 --seed 42
```

Para outro diretório ou orçamento:

```bash
python modeling/main.py \
  --trials 20 \
  --seed 42 \
  --output-dir modeling/artifacts/experimento-20
```

`--trials` significa total desejado por algoritmo, não trials adicionais. A
execução direta e `python -m modeling.main` são suportadas.

## Reprodutibilidade e limitações

A seed controla Optuna e as três bibliotecas. Os trials são sequenciais. Ainda
podem existir pequenas diferenças de ponto flutuante entre arquiteturas,
versões de compilador ou quantidades de threads; o manifesto registra as
versões instaladas para auditoria.

O conjunto possui apenas seis grupos de desenvolvimento. `cycle_id` alinha as
quatro simulações, mas a validação não é temporalmente expansiva e pode treinar
com identificadores posteriores ao validado. Essa decisão vem do plano 4; a
generalização para o ciclo posterior é medida exclusivamente pelo ciclo `6`.

A análise de Pearson em `feature_importance` usa inclusive o ciclo `6`, é
descritiva e marginal. Por isso ela não pode orientar seleção de features ou
hiperparâmetros neste pipeline. Também não se deve interpretar permutation
importance como causalidade.
