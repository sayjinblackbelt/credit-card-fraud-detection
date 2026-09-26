# Credit Card Fraud Detection

Projeto de Machine Learning para detecção de transações fraudulentas em cartões de crédito, desenvolvido em Python e executado em Google Colab.

O projeto foi desenvolvido a partir do desafio de detecção de anomalias em transações apresentado no curso, com experimentos adicionais de comparação de modelos, ajuste de limiar e explicabilidade com SHAP.

## Objetivo

Desenvolver e avaliar modelos capazes de identificar transações fraudulentas em uma base altamente desbalanceada.

O dataset possui 284.807 transações, sendo:

- 284.315 transações legítimas
- 492 transações fraudulentas
- 99,827% de transações legítimas
- 0,173% de transações fraudulentas

Nesse cenário, a acurácia isoladamente é uma métrica inadequada. Um modelo que classificasse praticamente todas as transações como legítimas poderia apresentar uma acurácia muito alta e, ao mesmo tempo, não detectar as fraudes.

Por isso, o projeto concentra a análise principalmente em:

- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Curva Precision-Recall

## Dataset

A base contém as variáveis:

- `Time`
- `V1` a `V28`
- `Amount`
- `Class`

As variáveis `V1` a `V28` são componentes transformadas por PCA.

A variável `Class` representa o alvo:

- `0` = transação legítima
- `1` = fraude

O dataset é carregado diretamente por URL no notebook e não é armazenado neste repositório.

## Pipeline

O projeto segue as seguintes etapas:

1. Carregamento dos dados
2. Exploração da estrutura da base
3. Análise do desbalanceamento
4. Criação da variável `LogAmount`
5. Separação entre características e variável-alvo
6. Divisão entre treino e teste utilizando `stratify`
7. Padronização das variáveis com `StandardScaler`
8. Treinamento dos modelos
9. Avaliação por métricas adequadas ao problema
10. Ajuste do limiar de decisão
11. Comparação dos modelos
12. Análise das curvas ROC e Precision-Recall
13. Explicabilidade utilizando SHAP

## Preparação dos dados

Foi criada a variável:

```python
LogAmount = log1p(Amount)
```

A transformação reduz a influência da grande variação existente nos valores das transações.

Os dados foram divididos em treino e teste utilizando `stratify=y`.

Resultado da divisão:

- Treino: 227.845 transações
- Teste: 56.962 transações
- Fraudes no treino: 394
- Fraudes no teste: 98

As variáveis foram padronizadas utilizando `StandardScaler`, ajustado somente no conjunto de treinamento.

## Modelos

Foram avaliados três modelos:

### Regressão Logística

Utilizada como modelo baseline.

| Métrica | Resultado |
|---|---:|
| Precision | 0,0612 |
| Recall | 0,9184 |
| F1-score | 0,1147 |
| ROC-AUC | 0,9719 |
| Average Precision | 0,7194 |

O modelo apresentou alto recall, identificando grande parte das fraudes, mas com baixa precisão e muitos falsos positivos.
### Random Forest

O Random Forest foi treinado utilizando `class_weight="balanced"`.

No limiar de 0,30:

| Métrica | Resultado |
|---|---:|
| Precision | 0,9419 |
| Recall | 0,8265 |
| F1-score | 0,8804 |
| ROC-AUC | 0,9523 |
| Average Precision | 0,8666 |

O ajuste do limiar permitiu observar diferentes relações entre precisão e recall.

### XGBoost

O XGBoost foi treinado considerando o desbalanceamento das classes por meio de peso para a classe fraude.

No limiar de 0,50:

| Métrica | Resultado |
|---|---:|
| Precision | 0,1833 |
| Recall | 0,8980 |
| F1-score | 0,3045 |
| ROC-AUC | 0,9800 |
| Average Precision | 0,6858 |

O modelo apresentou ROC-AUC de aproximadamente 0,98.

No limiar de 0,90:

| Métrica | Resultado |
|---|---:|
| Precision | 0,6777 |
| Recall | 0,8367 |
| F1-score | 0,7489 |
## Comparação dos modelos

Os resultados mostram que diferentes modelos e limiares produzem diferentes relações entre precisão e recall.

O Random Forest apresentou F1-score de 0,8804 no limiar de 0,30.

O XGBoost apresentou ROC-AUC de aproximadamente 0,98.

Esses resultados foram analisados em conjunto com as curvas ROC e Precision-Recall, evitando utilizar a acurácia como principal critério de avaliação.

## Ajuste de limiar

Foram testados diferentes limiares de decisão para observar o comportamento do XGBoost.

| Limiar | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0,1 | 0,0237 | 0,9388 | 0,0462 |
| 0,2 | 0,0500 | 0,9184 | 0,0949 |
| 0,3 | 0,0843 | 0,9184 | 0,1545 |
| 0,4 | 0,1294 | 0,9082 | 0,2265 |
| 0,5 | 0,1833 | 0,8980 | 0,3045 |
| 0,6 | 0,2507 | 0,8878 | 0,3910 |
| 0,7 | 0,3414 | 0,8673 | 0,4899 |
| 0,8 | 0,4450 | 0,8673 | 0,5882 |
| 0,9 | 0,6777 | 0,8367 | 0,7489 |

Neste experimento, o aumento do limiar reduziu as previsões de fraude, elevando a precisão e reduzindo o recall.

Esse comportamento demonstra o trade-off existente entre detectar mais fraudes e reduzir falsos positivos.
## Explicabilidade com SHAP

O SHAP foi utilizado para analisar a contribuição das variáveis nas decisões do modelo.

A análise global identificou as seguintes variáveis entre as de maior influência:

1. `V14`
2. `V4`
3. `V10`
4. `V12`
5. `V3`
6. `V11`
7. `V8`
8. `V7`
9. `V26`
10. `Amount`

Como `V1` a `V28` são componentes transformadas por PCA, sua importância pode ser utilizada para interpretar o comportamento do modelo, mas não permite atribuir diretamente um significado de negócio à variável original.

Também foi realizada uma análise SHAP individual de uma transação classificada como fraude.

As contribuições SHAP representam a influência das variáveis na decisão do modelo e não devem ser interpretadas como relações causais.
## O que foi acrescentado ao projeto

Além do pipeline apresentado no curso, foram realizados experimentos adicionais:

- comparação entre Regressão Logística, Random Forest e XGBoost;
- utilização de pesos para lidar com o desbalanceamento;
- teste de diferentes limiares de decisão;
- comparação de Precision, Recall e F1;
- análise das curvas ROC;
- análise das curvas Precision-Recall;
- cálculo de Average Precision;
- análise global de importância com SHAP;
- explicação individual de uma transação fraudulenta;
- documentação dos resultados obtidos nos experimentos.

O objetivo foi não apenas treinar modelos, mas compreender como as decisões de preparação, algoritmo e limiar alteram o comportamento do sistema de detecção.

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Google Colab

## Estrutura do projeto

```text
credit-card-fraud-detection/
│
├── .gitignore
├── README.md
│
└── notebooks/
    └── credit_card_fraud_detection.ipynb
```

## Como executar

O notebook foi desenvolvido para execução no Google Colab.

O dataset é carregado diretamente pela URL utilizada no notebook:

```python
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)
```

Não é necessário armazenar o dataset no repositório.

Para reproduzir o projeto:

1. Abra o notebook localizado em `notebooks/credit_card_fraud_detection.ipynb`.
2. Abra-o no Google Colab.
3. Execute as células em sequência.
4. Aguarde o treinamento dos modelos e a geração das análises.

## Conclusão

O projeto demonstra que a detecção de fraude é um problema em que a escolha das métricas e do limiar de decisão é fundamental.

A elevada acurácia observada em problemas altamente desbalanceados pode esconder uma capacidade insuficiente de detectar a classe minoritária.

A comparação entre diferentes modelos, o ajuste de limiar, a análise das curvas ROC e Precision-Recall e a utilização de SHAP permitiram analisar tanto o desempenho quanto o comportamento das decisões dos modelos.

O principal aprendizado foi compreender a relação entre recall, precisão, F1-score e o custo dos falsos positivos em um problema de classificação altamente desbalanceado.
