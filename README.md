# Detecção de Fraudes em Cartões de Crédito usando Machine Learning

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.0%2B-green)
![NumPy](https://img.shields.io/badge/NumPy-1.18%2B-yellow)

#
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-j836FHavVI-DtgGpnlPAjf4uag3KBww?usp=sharing)
#

## 📊 Visão Geral do Projeto

Este projeto implementa um sistema avançado de detecção de fraudes em cartões de crédito utilizando técnicas de machine learning. O sistema analisa transações e identifica padrões suspeitos que possam indicar atividades fraudulentas.

## 👥 Equipe

- Gustavo Rossi
- Bruno Trevizan
- Yuji Kiyota

## 🛠️ Bibliotecas Utilizadas

- **pandas**: Manipulação e análise de dados
- **numpy**: Operações numéricas
- **scikit-learn**: Algoritmos de ML e pré-processamento
- **imbalanced-learn**: Tratamento de dados desbalanceados
- **colorama**: Saída colorida no console
- **tabulate**: Formatação de tabelas no console

## 🔧 Instalação

pip install pandas numpy scikit-learn imbalanced-learn colorama tabulate

Funcionalidades Principais

    Carregamento e Pré-processamento de Dados
        Carrega dados do arquivo CSV
        Cria novas features
        Trata valores ausentes

    Engenharia de Features
        Seleção das melhores features usando SelectKBest
        Balanceamento de dados com SMOTETomek

    Treinamento de Modelos
        Random Forest
        Regressão Logística
        Gradient Boosting
        SVM
        Ensemble (Voting Classifier)

    Otimização de Hiperparâmetros
        Utiliza GridSearchCV para cada modelo

    Avaliação de Modelos
        Métricas: Acurácia, Matriz de Confusão, AUC-ROC, Relatório de Classificação

    Ajuste de Limiar
        Otimiza o limiar de classificação para melhorar o desempenho

    Análise de Erros
        Examina as características das previsões incorretas

💻 Como Usar

    Clone o repositório:

    git clone https://github.com/seu-usuario/deteccao-fraude-cartao.git

📊 Resultados

O script exibirá resultados detalhados para cada modelo, incluindo:

    Acurácia
    Matriz de Confusão
    AUC-ROC
    Relatório de Classificação

🔍 Análise de Erros

O sistema realiza uma análise detalhada dos erros de classificação, fornecendo insights sobre as características das transações mal classificadas.


    

