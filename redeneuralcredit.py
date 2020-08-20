#!/usr/bin/env bash
#
# 
#  Rede Neural aplicada aos dados do credit data
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#   v1.0 19/08//2020 (Começou a funcionar)
#  Testado em: python 3.8.1
# ------------------------------Importações----------------------------------- #
import pandas as pd
import numpy as np

# --------------------------Tratamento dos  Dados----------------------------- #
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
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)


# -------------------------------Rede Neural---------------------------------- #
from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(verbose = True,                # visualização da aprendizagem 
                              max_iter=1000,                 # maxímo de interações 
                              tol = 0.0000010,               # taxa de melhoramento mínimo para 2 épocas consectiva 
                              solver = 'adam',               # para pesos  
                              hidden_layer_sizes=(100, 100), # numero de camdas coultas e neuronios, nesse caso : 2 cmadas ocultas 110 neuronios cada
                              activation='relu')             # Escolhda da função de ativação para os neurônios 
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# ------------------------Verificando precisão ------------------------------- #
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)