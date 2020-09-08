#!/usr/bin/env bash
#
# 
#  Regressão Linear Utilizando Vetores de Suporte
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#  v1.0 08/09//2020 (Começou a funcionar)
#  Testado em: python 3.8.2
#
# ------------------------Importações e Variáveis----------------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = pd.read_csv('plano_saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1:2].values

# -----------------------------kernel linear---------------------------------- #
from sklearn.svm import SVR
regressor_linear = SVR(kernel = 'linear')
regressor_linear.fit(X, y.ravel())


# -------------------------Gráfico kernel linear------------------------------ #
plt.scatter(X, y)
plt.plot(X, regressor_linear.predict(X), color = 'red')
regressor_linear.score(X, y)

# ------------------------------kernel poly----------------------------------- #
regressor_poly = SVR(kernel = 'poly', degree = 3, gamma = 'auto')  #regressao polinomial 
regressor_poly.fit(X, y.ravel())


# -------------------------Gráfico kernel poly-------------------------------- #
plt.scatter(X, y)
plt.plot(X, regressor_poly.predict(X), color = 'red')
regressor_poly.score(X, y)

# ------------------------------kernel rbf----------------------------------- #
from sklearn.preprocessing import StandardScaler #escalonamento dos dados
scaler_x = StandardScaler()                           
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

regressor_rbf = SVR(kernel = 'rbf', gamma = 'auto')
regressor_rbf.fit(X, y.ravel())
# -------------------------Gráfico kernel rbf--------------------------------- #
plt.scatter(X, y)
plt.plot(X, regressor_rbf.predict(X), color = 'red')
regressor_rbf.score(X, y)

# ------------------------------Previsão-------------------------------------- #
previsao1 = scaler_y.inverse_transform(regressor_linear.predict(scaler_x.transform(np.array(40).reshape(1, -1))))
previsao2 = scaler_y.inverse_transform(regressor_poly.predict(scaler_x.transform(np.array(40).reshape(1, -1))))
previsao3 = scaler_y.inverse_transform(regressor_rbf.predict(scaler_x.transform(np.array(40).reshape(1, -1))))
