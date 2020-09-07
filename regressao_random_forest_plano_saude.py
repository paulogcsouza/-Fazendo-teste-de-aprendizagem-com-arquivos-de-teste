#!/usr/bin/env bash
#
# 
#  Regressão Linear Utilizando Random Florest
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#   v1.0 07/09//2020 (Começou a funcionar)
#  Testado em: python 3.8.2
#
# ------------------------Importações e Variáveis----------------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = pd.read_csv('plano_saude2.csv')
X = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

# -----------------------------Random Florest--------------------------------- #
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10)      # número de arvores randomicas
regressor.fit(X, y)
score = regressor.score(X, y)

# -------------------------Gráfico Random Florest----------------------------- #
X_teste = np.arange(min(X), max(X), 0.1)     # escalonamento
X_teste = X_teste.reshape(-1,1)
plt.scatter(X, y)
plt.plot(X_teste, regressor.predict(X_teste), color = 'red')
plt.title('Regressão com random forest')
plt.xlabel('Idade')
plt.ylabel('Custo')

regressor.predict(np.array(40).reshape(1, -1))  