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
