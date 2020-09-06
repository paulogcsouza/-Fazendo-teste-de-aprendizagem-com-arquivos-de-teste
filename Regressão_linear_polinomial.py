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
#   Bash 5.0.17
# ------------------------Importações e Variáveis----------------------------- #
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

base = pd.read_csv('plano_saude2.csv')


x = base.iloc[:,0:1].values          # Formato de matriz com valeus  dos previsores
y = base.iloc[:,1].values            # Classificadores no formato matriz
 
# ------------------------Grafico Plano de Saude 2---------------------------- #

plt.scatter(x, y)
plt.plot(x,y)
plt.title('Grafico Plano de Saude 2')
plt.xlabel('Idade', color = 'red')
plt.ylabel('Custo', color = 'red')

# ------------------------Regressão Linear Simples---------------------------- #
from sklearn.linear_model import LinearRegression 
regression1 = LinearRegression ()
regression1.fit(x,y)
score1 = regression1.score(x, y)       # confiabilidade

# ------------------------------Grafico R.L.S--------------------------------- #
plt.scatter(x, y)
plt.plot(x, regression1.predict(x), color = 'red')
plt.title('Regressão linear')
plt.xlabel('Idade')
plt.ylabel('Custo')

# ---------------------------Regressão Polinomial----------------------------- #
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)                  # polinomio elevado até ao quadrado
x_poly = poly.fit_transform(x)
regression2 = LinearRegression()
regression2.fit(x_poly, y)
score2 = regression2.score(x_poly, y)

regression2.predict(poly.transform(np.array(40).reshape(1, -1)))  # previsão pessoa de 40 anos
 
# ----------------------Grafico Regressão Polinomial-------------------------- #
plt.scatter(x, y)
plt.plot(x, regression2.predict(poly.fit_transform(x)), color = 'red')
plt.title('Regressão polinomial')
plt.xlabel('Idade')
plt.ylabel('Custo')




