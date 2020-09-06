#!/usr/bin/env bash
#
# 
#  Regressão Linear para house price, com apenas uma variável (tamnho quadrdo da casa)
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#   v1.0 06/09//2020 (Começou a funcionar)
#   Testado em: python 3.8.2
#
# ------------------------Importações e Variáveis----------------------------- #
import pandas as pd

base = pd.read_csv('house_prices.csv')

X = base.iloc[:, 3:19].values                 # previsores no formato de matrix
y = base.iloc[:, 2].values                    # calssificador 

# ---------------------------Divisão das Bases-------------------------------- #
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
# ------------------------Treinamento Regressivo------------------------------ #
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)
score = regressor.score(X_treinamento, y_treinamento)  # precisão da base de treinamento

previsoes = regressor.predict(X_teste)


# --------------------------------Resultado----------------------------------- #
from sklearn.metrics import mean_absolute_error,  mean_squared_error
mae = mean_absolute_error(y_teste, previsoes)      # Médio dos valores absoluto
mse = mean_squared_error(y_teste, previsoes)       # Erro quadratico

regressor.score(X_teste, y_teste)                  # precisão da base de teste

#b0
regressor.intercept_
#bi (coeficiente)
len(regressor.coef_)


