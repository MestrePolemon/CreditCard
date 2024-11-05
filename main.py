import pandas as pd
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from colorama import Fore, Style
from tabulate import tabulate

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
    print(f"Melhores parâmetros para {modelo.__class__.__name__}: {grid_search.best_params_}")
    return grid_search.best_estimator_

def exibir_informacoes(y_test, y_pred):
    acs = accuracy_score(y_test, y_pred) * 100
    cfm = confusion_matrix(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    matriz_confusao = tabulate(cfm, headers=['Fraude', 'Não Fraude'], tablefmt='fancy_grid', showindex=['Fraude', 'Não Fraude'])
    print(Fore.GREEN + 'Informações sobre o modelo:' + Style.RESET_ALL)
    print(f'Acurácia(%): {acs:.2f}%')
    print(matriz_confusao)
    print(f'AUC-ROC: {auc_roc:.4f}')
    print(classification_report(y_test, y_pred))

def treinar_modelos(X_train, y_train, X_test, y_test):
    # Definir os hiperparâmetros com intervalos mais amplos

    # Melhores parâmetros para RandomForestClassifier: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 150}
    param_grid_rf = {
        'n_estimators': [150],
        'max_depth': [20],
        'min_samples_split': [5],
        'min_samples_leaf': [1],
        'max_features': ['sqrt']
    }

    # Melhores parâmetros para LogisticRegression: {'C': 1, 'class_weight': 'balanced', 'max_iter': 1000, 'penalty': 'l2', 'solver': 'liblinear'}
    param_grid_lr = {
        'C': [1],
        'solver': ['liblinear'],
        'penalty': ['l2'],
        'class_weight': ['balanced'],
        'max_iter': [1000]
    }

    # Melhores parâmetros para GradientBoostingClassifier: {'learning_rate': 0.1, 'max_depth': 7, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200, 'subsample': 1}
    param_grid_gb = {
        'n_estimators': [150],
        'learning_rate': [0.1],
        'max_depth': [7],
        'min_samples_split': [5],
        'min_samples_leaf': [1],
        'subsample': [1]
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

    # Otimização do VotingClassifier com pesos ajustados
    print("\nTREINAMENTO COM VOTING CLASSIFIER\n")
    voting_class = VotingClassifier(
        estimators=[('rf', best_rf), ('lr', best_lr), ('gb', best_gb)],
        voting='hard'
    )
    voting_class.fit(X_train, y_train)
    y_pred = voting_class.predict(X_test)
    exibir_informacoes(y_test, y_pred)

def main():
    os.environ['LOKY_MAX_CPU_COUNT'] = '4'
    df = carregar_dados('creditcard.csv')
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = dividir_dados(X, y)
    X_train_res, y_train_res = balancear_dados(X_train, y_train)
    treinar_modelos(X_train_res, y_train_res, X_test, y_test)

if __name__ == "__main__":
    main()
