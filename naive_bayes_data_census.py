#!/usr/bin/env bash
#
# |Método Naive Bayes Para Aprendizagem de Máquina
#
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#   v1.0 2/06/2020 (Começou a funcionar)
#  Testado em: python 3.8.1

import pandas as pd 
base = pd.read_csv('census.csv')

#Divisão entre Previsores e Classificadores 
previsores = base.iloc[:, 0:14].values # 0 até 13 coluna selecionada
classe = base.iloc[:, 14].values       # 14 coluna selecionada     

#Transformando Variaveis Categóricas em  Variáveius Discretas
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])


# Fazendo Avaliação dos resultados com tipos de pré processamento 
'''
# Acertando variaveis para Aprendizagem de máquina
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [1,3,5,6,7,8,9,13])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'   )
previsores = onehotencoder.fit_transform(previsores).toarray()
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

#Escalonamento das variáveis
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores) '''

#Divisão Entre Base de Treinamento e de Teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

#Aplicando Naive Bayes
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# Análise de Confianbilidade em % 
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)           # função que retorna o nível de confiança 
matriz = confusion_matrix(classe_teste, previsoes) 