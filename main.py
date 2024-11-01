# Projeto de detecção de fraude em cartão de crédito
# Integrantes Bruno Trevizan, Gustavo Rossi e Yuji Kiyota

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from colorama import Fore, Style
import pandas as pd

def exibir_informacoes(y_test, y_prev):
    acs = accuracy_score(y_test, y_prev, normalize=True) * 100
    cfm = confusion_matrix(y_test, y_prev)
    matriz_confusao = tabulate(cfm, headers=['Fraude', 'Não Fraude'], tablefmt='fancy_grid', showindex=['Fraude', 'Não Fraude'])
    print(Fore.GREEN + 'Informações sobre o modelo:' + Style.RESET_ALL)
    print(f'Acurácia(%): {acs:.2f}%')
    print(matriz_confusao)
    print(classification_report(y_test, y_prev))

df = pd.read_csv('creditcard.csv')

X = df.drop(columns=['Class'])
y = df['Class']

# Split the data once
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Define parameter grids for each model
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 30],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

param_grid_lr = {
    'C': [0.1, 1, 10],
    'class_weight': ['balanced']
}


param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.02, 0.2, 0.3]
}

# Perform Grid Search for Random Forest
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='accuracy')
grid_search_rf.fit(X_train_res, y_train_res)
best_rf = grid_search_rf.best_estimator_

# Perform Grid Search for Logistic Regression
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

grid_search_lr = GridSearchCV(LogisticRegression(class_weight="balanced", random_state=42, max_iter=10000), param_grid_lr, cv=3, scoring='accuracy')
grid_search_lr.fit(X_train_res_scaled, y_train_res)
best_lr = grid_search_lr.best_estimator_

# Perform Grid Search for Gradient Boosting
grid_search_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=3, scoring='accuracy')
grid_search_gb.fit(X_train_res, y_train_res)
best_gb = grid_search_gb.best_estimator_

print("\nTREINAMENTO COM RANDOM FOREST\n")
y_prev = best_rf.predict(X_test)
exibir_informacoes(y_test, y_prev)

print("\nTREINAMENTO COM REGRESSÃO LOGISTÍCA\n")
y_prev = best_lr.predict(X_test_scaled)
exibir_informacoes(y_test, y_prev)

print("\nTREINAMENTO COM GRADIENT BOOSTING\n")
y_prev = best_gb.predict(X_test)
exibir_informacoes(y_test, y_prev)

print("\nTREINAMENTO COM VOTING CLASSIFIER\n")
voting_class = VotingClassifier(estimators=[('rf', best_rf), ('lr', best_lr), ('gb', best_gb)], voting='soft')
voting_class.fit(X_train_res, y_train_res)
y_pred = voting_class.predict(X_test)
exibir_informacoes(y_test, y_pred)