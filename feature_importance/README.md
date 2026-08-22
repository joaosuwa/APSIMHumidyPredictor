# Feature importance por correlação

Esta pasta contém o código e os resultados da análise de correlação de Pearson
entre as features oficiais do modelo e `deficit_agua_proximo_dia_mm`.

## Executar

A partir da raiz do projeto:

```bash
python -m pip install -r requirements-modeling.txt
python -m feature_importance.analysis
```

Por padrão, são analisadas as quatro simulações não irrigadas configuradas em
`modeling/config.py`. Para selecionar outras simulações ou caminhos:

```bash
python -m feature_importance.analysis \
  --dataset data/processed/model/training_dataset.csv \
  --output-dir feature_importance \
  --simulations Alegrete AlegreteClay Simulation SimulationClay
```

O comando sobrescreve deterministicamente `summary.txt`,
`feature_importance.csv` e os três arquivos PNG em `plots/`.

## Testar

```bash
python -m pytest feature_importance/test_analysis.py
```

O ranking usa a magnitude `|r|`; o campo `pearson_r` mantém o sinal da
associação. Correlação é uma medida marginal de associação linear e não implica
causalidade.
