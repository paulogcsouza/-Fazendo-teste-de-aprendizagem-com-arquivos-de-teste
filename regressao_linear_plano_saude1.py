#!/usr/bin/env bash
#
# 
#  Regressão Linear para plano casa
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#   v1.0 01/09//2020 (Começou a funcionar)
#  Testado em: python 3.8.2
#
# ------------------------------Importações----------------------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------Variáveis------------------------------------ #
base = pd.read_csv('plano_saude.csv')
X = base.iloc[:, 0].values           # Variável independente 
y = base.iloc[:, 1].values           # Variavel dependene

correlacao = np.corrcoef(X, y)       # coeficiente de corelação entre as variaáveis 

X = X.reshape(-1,1)                  # transforma o corpo do vetor em matrix 
# ---------------------------regressão Linear--------------------------------- #
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)                 # aprendizagem 

# ------------------------------Coeficientes---------------------------------- #
# b0
regressor.intercept_
# b1
regressor.coef_

# ---------------------------------Grafico------------------------------------ #
plt.scatter(X, y)                                   #Relação entre x e y por pontos
plt.plot(X, regressor.predict(X), color = 'red')    # reta dos privissores 
plt.title ("Regressão linear simples")
plt.xlabel("Idade", color = 'red')
plt.ylabel("Custo", color = 'red')

# -------------------------previsão pessoa com 40 anos-------------------------#
previsao1 = regressor.intercept_ + regressor.coef_ * 40
previsao2 = regressor.predict(np.array(40).reshape(1, -1)) # calculo manual da previsão 

# --------------------Avaliação da precisão  da Regressão--------------------- #
score = regressor.score(X,y)   

# -------------------------Valores Resuduais Grafico ------------------------- #
from yellowbrick.regressor import ResidualsPlot
visualizador = ResidualsPlot(regressor)
visualizador.fit(X, y)
visualizador.poof()
