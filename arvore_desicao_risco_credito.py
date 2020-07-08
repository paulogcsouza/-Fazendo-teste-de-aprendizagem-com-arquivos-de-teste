#!/usr/bin/env bash
#
# Arquivos de Configuração
#
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#   v1.0 6/7/2020 
#   
#  Testado em:
#   bash 4.4.20(1)
#   python 3.7.6



import pandas as pd

# -----------------------------Carregar Base de Dados------------------------- #
base = pd.read_csv('risco_credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

# ---------------------Atributos categóricos em discreto---------------------- #
from sklearn.preprocessing import LabelEncoder 
labelenconder = LabelEncoder()
previsores[:,0] = labelenconder.fit_transform(previsores[:,0])
previsores[:,1] = labelenconder.fit_transform(previsores[:,1])
previsores[:,2] = labelenconder.fit_transform(previsores[:,2])
previsores[:,3] = labelenconder.fit_transform(previsores[:,3])

# ---------------------------------------------------------------------------- #
from sklearn.tree import DecisionTreeClassifier, export
classificador = DecisionTreeClassifier(criterion = 'entropy')
classificador.fit(previsores,  classe) #constroe arevore

print(classificador.feature_importances_) #demonstra a importancia de cada ramo


export.export_graphviz(classificador,
                       out_file = 'arvore.dot',
                       feature_names = ['historia', 'divida', 'garantias', 'renda'],
                       class_names = classificador.classes_,
                       filled = True,
                       leaves_parallel=True)
# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])
print(classificador.classes_)
