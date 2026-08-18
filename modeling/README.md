# Modelagem do déficit hídrico

Esta pasta isola o código e os artefatos dos experimentos de regressão.

## Execução

Na raiz do repositório:

```bash
python -m pip install -r modeling/requirements.txt
python -m scripts.model_dataset
python -m unittest discover -s modeling/tests
python -m modeling.src.train --trials 50 --seed 42
```

O tuning usa os ciclos semeados entre 2019 e 2023 para treino e os ciclos de
2024 para validação. O treino final usa 2019–2024; os ciclos semeados em 2025
formam o teste temporal intocado.

O target é `variacao_deficit_proximo_dia_mm`. A previsão absoluta é
reconstruída somando a variação prevista ao `dr_mm` do dia atual.

## Saídas

- `results/`: métricas, previsões, hiperparâmetros e estudos Optuna;
- `models/`: modelos finais serializados;
- `plots/`: comparação, resíduos, permutation importance e SHAP.
