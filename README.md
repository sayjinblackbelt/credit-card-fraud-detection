# Credit Card Fraud Detection

Projeto de Machine Learning para detecção de transações fraudulentas em cartões de crédito, desenvolvido em Python e executado em Google Colab.

O projeto foi desenvolvido a partir do desafio de detecção de anomalias em transações apresentado no curso, mas com experimentos adicionais de comparação de modelos, ajuste de limiar e explicabilidade com SHAP.

## Objetivo

Desenvolver e avaliar modelos capazes de identificar transações fraudulentas em uma base altamente desbalanceada.

O dataset possui 284.807 transações, sendo:

- 284.315 transações legítimas
- 492 transações fraudulentas
- 99,827% de transações legítimas
- 0,173% de transações fraudulentas

Nesse cenário, a acurácia isoladamente é uma métrica inadequada. Um modelo que classificasse praticamente todas as transações como legítimas poderia apresentar uma acurácia muito alta e, ao mesmo tempo, não detectar as fraudes.

Por isso, o projeto concentra a análise principalmente em:

- Precisão (Precision)
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

As variáveis `V1` a `V28` são componentes transformadas por PCA para preservar a privacidade das informações originais.

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

`LogAmount = log1p(Amount)`

A transformação reduz a influência da grande variação existente nos valores das transações.

Os dados foram divididos em treino e teste utilizando `stratify=y`, preservando a proporção de fraudes nos dois conjuntos.

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

Resultado no ponto de decisão analisado:

| Métrica | Resultado |
|---|---:|
| Precision | 0,0612 |
| Recall | 0,9184 |
| F1-score | 0,1147 |
| ROC-AUC | 0,9719 |
| Average Precision | 0,7194 |

O modelo apresentou alto recall, identificando a maior parte das fraudes, mas com baixa precisão e muitos falsos positivos.

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

No limiar padrão de 0,50:

| Métrica | Resultado |
|---|---:|
| Precision | 0,1833 |
| Recall | 0,8980 |
| F1-score | 0,3045 |
| ROC-AUC | 0,9800 |
| Average Precision | 0,6858 |

O modelo apresentou ROC-AUC de aproximadamente 0,98.

O ajuste do limiar mostrou uma melhora significativa na precisão:

No limiar de 0,90:

| Métrica | Resultado |
|---|---:|
| Precision | 0,6777 |
| Recall | 0,8367 |
| F1-score | 0,7489 |

## Comparação

Os resultados mostram que não existe uma única métrica suficiente para avaliar o problema.

O limiar modifica diretamente o equilíbrio entre:

- detectar mais fraudes;
- gerar menos falsos positivos.

O experimento com diferentes limiares permitiu observar esse trade-off de forma prática.

O Random Forest apresentou F1-score de 0,8804 no limiar de 0,30, enquanto o XGBoost apresentou ROC-AUC de aproximadamente 0,98.

Esses resultados foram analisados em conjunto com as curvas ROC e Precision-Recall, evitando utilizar a acurácia como principal critério de escolha.

## Ajuste de limiar

Foram testados diferentes limiares de decisão para observar o comportamento dos modelos.

No XGBoost, por exemplo:

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

O experimento demonstra que aumentar o limiar reduz o número de transações classificadas como fraude, aumentando a precisão, mas reduzindo o recall.

## Explicabilidade com SHAP

O SHAP foi utilizado para compreender como o XGBoost toma suas decisões.

Na análise global, as variáveis que apresentaram maior influência média foram:

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

### Explicação individual

Também foi analisada uma transação específica:

- Índice: `3287`
- Classe real: `1` (fraude)
- Probabilidade estimada: `0.9964`
- Limiar analisado: `0.90`
- Previsão: `1` (fraude)

A análise SHAP mostrou quais variáveis contribuíram positivamente ou negativamente para a saída do modelo nessa decisão.

As contribuições observadas nessa transação incluem variáveis como `V14`, `V19`, `V10`, `V12` e `V11`.

As contribuições SHAP representam a influência das variáveis na decisão do modelo e não devem ser interpretadas como relações causais.

## O que foi acrescentado em relação ao caminho apresentado nas aulas

Além do pipeline apresentado no projeto, foram realizados experimentos adicionais:

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

## Estrutura do projeto

```text
credit-card-fraud-detection/
│
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
│
└── README.md
