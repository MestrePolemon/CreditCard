import pandas as pd
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from colorama import Fore, Style
from tabulate import tabulate
from sklearn.metrics import roc_auc_score

def carregar_dados(caminho):
    return pd.read_csv(caminho)


def balancear_dados(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)


def dividir_dados(X, y, test_size=0.4, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def padronizar_dados(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def otimizar_modelo(modelo, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
    grid_search = GridSearchCV(modelo, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def exibir_informacoes(y_test, y_prev):
    acs = accuracy_score(y_test, y_prev) * 100
    cfm = confusion_matrix(y_test, y_prev)
    auc_roc = roc_auc_score(y_test, y_prev)
    matriz_confusao = tabulate(cfm, headers=['Fraude', 'Não Fraude'], tablefmt='fancy_grid', showindex=['Fraude', 'Não Fraude'])
    print(Fore.GREEN + 'Informações sobre o modelo:' + Style.RESET_ALL)
    print(f'Acurácia(%): {acs:.2f}%')
    print(matriz_confusao)
    print(f'Classes divididas com eficacia: {auc_roc:.4f}')
    print(classification_report(y_test, y_prev))


def treinar_modelos(X_train, y_train, X_test, y_test):

    # Definir os hiperparâmetros para otimização dos modelos
    param_grid_rf = {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 30],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }

    param_grid_lr = {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear'],
        'penalty': ['l2'],
        'max_iter': [10000],
        'class_weight': ['balanced'],
    }

    param_grid_gb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'subsample': [0.8, 1]
    }

    print("\nTREINAMENTO COM RANDOM FOREST\n")
    best_rf = otimizar_modelo(RandomForestClassifier(random_state=42), param_grid_rf, X_train, y_train)
    y_prev = best_rf.predict(X_test)
    exibir_informacoes(y_test, y_prev)

    print("\nTREINAMENTO COM REGRESSÃO LOGÍSTICA\n")
    X_train_scaled, X_test_scaled = padronizar_dados(X_train, X_test)
    best_lr = otimizar_modelo(LogisticRegression(random_state=42), param_grid_lr, X_train_scaled, y_train)
    y_prev = best_lr.predict(X_test_scaled)
    exibir_informacoes(y_test, y_prev)

    print("\nTREINAMENTO COM GRADIENT BOOSTING\n")
    best_gb = otimizar_modelo(GradientBoostingClassifier(random_state=42), param_grid_gb, X_train, y_train)
    y_prev = best_gb.predict(X_test)
    exibir_informacoes(y_test, y_prev)

    print("\nTREINAMENTO COM VOTING CLASSIFIER\n")
    voting_class = VotingClassifier(estimators=[('rf', best_rf), ('lr', best_lr), ('gb', best_gb)],
                                    voting='soft',
                                    weights=[2, 1, 1])
    voting_class.fit(X_train, y_train)
    y_pred = voting_class.predict(X_test)
    exibir_informacoes(y_test, y_pred)


def main():
    os.environ['LOKY_MAX_CPU_COUNT'] = '6'
    df = carregar_dados('creditcard.csv')
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = dividir_dados(X, y)
    X_train_res, y_train_res = balancear_dados(X_train, y_train)
    treinar_modelos(X_train_res, y_train_res, X_test, y_test)


if __name__ == "__main__":
    main()

