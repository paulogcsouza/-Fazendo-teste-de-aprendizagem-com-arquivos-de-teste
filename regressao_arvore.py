#!/usr/bin/env bash
#
# 
#  Regressão Linear Utilizando Arvore de decisão
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#   v1.0 01/09//2020 (Começou a funcionar)
#  Testado em: python 3.8.2
#
# ------------------------Importações e Variáveis----------------------------- #
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

base = pd.read_csv('plano_saude2.csv')
x = base.iloc[:,0:1].values          # Formato de matriz com valeus  dos previsores
y = base.iloc[:,1].values            # Classificadores no formato matriz


# -------------------------Regressão Com Arvores------------------------------ #
from sklearn.tree import DecisionTreeRegressor
regressao =  DecisionTreeRegressor()
regressao.fit(x,y)
score = regressao.score(x,y)

# -----------------------Grafico Plano de Saude (1)--------------------------- #

plt.style.use('classic') 
plt.scatter(x, y)
plt.plot(x, regressao.predict(x), color = 'green', linewidth=2)
plt.title('Grafico Plano de Saude 2')
plt.xlabel('Idade', color = 'red')
plt.ylabel('Custo', color = 'red')

# -----------------------Grafico Plano de Saude (2)--------------------------- #
X_teste = np.arange(min(x), max(x), 0.1)                       # escalona dcom alcance de por vez 0.1 
X_teste = X_teste.reshape(-1,1)                                # transforma em matriz x_teste
plt.scatter(x, y)
plt.plot(X_teste, regressao.predict(X_teste), color = 'red')
plt.title('Regressão com árvores')
plt.xlabel('Idade')
plt.ylabel('Custo')

# ------------------------Teste pessoa de 40 anos----------------------------- #
regressao.predict(np.array(40).reshape(1, -1))  # resposta ([1150.])