#!/usr/bin/env bash
#
# 
#  Agrupamento com Algorítimo Apriori para 'mercado2.csv'
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#  v1.0 09/09//2020 (Começou a funcionar)
#  Testado em: python 3.8.2
#
# ------------------------Importações e Variáveis------------------------------#
import pandas as pd

dados = pd.read_csv('mercado2.csv', header = None)

# -------------------Conversão de DataFrame para Lista-------------------------#
transacoes = []
for i in range(0, 7501):                                                       # Percorrendo as linhas do arquivo
    transacoes.append([str(dados.values[i,j]) for j in range(0, 20)])          # percorrendo as Colunas

# --------------------------Algorítimo Apriori ------------------------------- #
from apyori import apriori
regras = apriori(transacoes, min_support = 0.003, min_confidence = 0.2, min_lift = 2.0, min_length = 2)

resultados = list(regras)
resultados

resultados2 = [list(x) for x in resultados]
resultados2
resultadoFormatado = []
for j in range(0, 3):                                                 # valores apenas do 3 primeiras associações 
    resultadoFormatado.append([list(x) for x in resultados2[j][2]])
resultadoFormatado


# min_support = 0.3                   suporte mínimo
# min_confidence = 0.8                confiança mínima
# min_lift = 2                        lift- indica qual forte é a associação 
# min_length = 2                      length - quantidade ménima que ele vai gerar por regra                