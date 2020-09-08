#!/usr/bin/env bash
#
# 
#  Regressão Linear Utilizando Redes Neurais
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#   v1.0 08/09//2020 (Começou a funcionar)
#  Testado em: python 3.8.2
#
# ------------------------Importações e Variáveis----------------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = pd.read_csv('plano_saude2.csv')
X = base.iloc[:, 0:1].values
y = base.iloc[:, 1:2].values                 # formato de matriz

# -----------------------Escalonamentodas Variaveis--------------------------- #
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# ------------------------------Rede Neural----------------------------------- #
from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor()
regressor.fit(X, y)

regressor.score(X, y)

# -------------------------Gráfico kernel rbf--------------------------------- #
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title('Regressão com redes neurais')
plt.xlabel('Idade')
plt.ylabel('Custo')

# ------------------------------Previsão-------------------------------------- #
previsao = scaler_y.inverse_transform(regressor.predict(scaler_x.transform(np.array(40).reshape(1, -1))))