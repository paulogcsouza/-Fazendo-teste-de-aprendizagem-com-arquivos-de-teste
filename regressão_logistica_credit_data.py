
#!/usr/bin/env bash
#
# KNN 
#
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#   v1.0 25/07/2020 (Começou a funcionar)
#  Testado em: python 3.8.1

import pandas as pd 
import numpy as np

base = pd.read_csv('data_credito.csv')

# ---------------------------Tratamento de Erro------------------------------- #
base.loc[base.age < 0, 'age'] = 40.92

# -----------------Divisão previsores e Classificadores ---------------------- #

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
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# --------------------------Regressão Lógica---------------------------------- #
from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression(random_state = 1, solver='lbfgs')
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# ------------------------Verificando precisão ------------------------------- #
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
