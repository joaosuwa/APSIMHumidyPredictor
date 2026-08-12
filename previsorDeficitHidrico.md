# Previsor de deficit hidrico

## Ideia do projeto:

Construir um modelo de aprendizado de máquina que utilize dados provenientes de clima, simulações no simulador APSINM NG e previsões de chuva históricas para prever o déficit hídrico do solo.

## Variáveis para o modelo:

| Variável | Tipo | Observação | Como obter |
| :--- | :--- | :--- | :--- |
| Umidade do solo | Solo | Umidade do solo atual (Água disponível na zona radicular) | APSIM NG |
| Umidade do solo passada | Série temporal | Umidade de dias anteriores | APSIM NG |
| Precipitação observada | Meteorológica | Principal entrada de água | INMET |
| Irrigação aplicada no dia | Manejo | Fundamental para o modelo entender o efeito da irrigação | APSIM NG |
| Irrigação aplicada no dia posterior | Manejo | Fundamental para o modelo entender o efeito da irrigação | APSIM NG |
| ET0 | Meteorológica derivada | Representa a demanda atmosférica | APSIM NG |
| ETreal | Meteorológica derivada | Quanto de água foi consumido solo + planta (Transpiração + evaporação) | APSIM NG |
| Previsão de chuva | Meteorológica | - | [NCAR GDEX](https://gdex.ucar.edu/datasets/d084001/) |
| Previsão de ET0 | Meteorológica derivada | Informação complicada de se obter | [NCAR GDEX](https://gdex.ucar.edu/datasets/d084001/) |
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


## Como serão obtidos os dados:

### Variáveis report no APSINM NG:

Irrigação aplicada, Irrigação aplicada no dia posterior, Profundidade radicular Zr, Escoamento superficial de água e Drenagem

### Cálculos feitos usando dados do APSINM NG:

Umidade do solo, Umidade do solo histórica = SW até zona radicular.

ETreal = [Soil].SoilWater.Es (Evaporação do solo) + [Plant].Leaf.Transpiration (Transpiração da planta)

Depleção de água no solo (Dr) = DUL - SW de todas as camadas até zona radicular (Cálculo já feito no simulador)

ET0 = Fórmula Priestley-Taylor

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

Previsão de chuva, Previsão de ET0