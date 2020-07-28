
#!/usr/bin/env bash
#
# KNN 
#
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#   v1.0 27/07/2020 (Começou a funcionar)
#  Testado em: python 3.8.1

import pandas as pd
import numpy as np

# ---------------------------Tratamento de Erro------------------------------- #
base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# ----------------------Tratamento Valores ausentes--------------------------- #
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# ---------------------Escalonamento das Variáveis---------------------------- #
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# ---------------------Base de teste e treinamento---------------------------- #
from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)


# ---------------------------------SVC---------------------------------------- #
from sklearn.svm import SVC
classificador = SVC(kernel = 'rbf', random_state = 1, C = 2.0, gamma='auto')
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# ------------------------Verificando precisão ------------------------------- #
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)


