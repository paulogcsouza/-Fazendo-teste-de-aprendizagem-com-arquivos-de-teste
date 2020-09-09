
#!/usr/bin/env bash
#
# 
#  Agrupamento com Algorítimo Apriori para 'mercado.csv'
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#  v1.0 09/09//2020 (Começou a funcionar)
#  Testado em: python 3.8.2
#
# ------------------------Importações e Variáveis----------------------------- #

import pandas as pd
from apyori import apriori


dados = pd.read_csv('mercado.csv', header = None)                        # header = None informa que nao yem cabesalho o arquivo 
transacoes = []                                                          # Criando lista vazia
for i in range(0, 10):
    transacoes.append([str(dados.values[i,j]) for j in range(0, 4)])


# --------------------------Algorítimo Apriori ------------------------------- #
regras = apriori(transacoes, min_support = 0.3, min_confidence = 0.8, min_lift = 2, min_length = 2)
resultados = list(regras)
resultados

print (resultados)


# min_support = 0.3                   suporte mínimo
# min_confidence = 0.8                confiança mínima
# min_lift = 2                        lift- indica qual forte é a associação 
# min_length = 2                      length - quantidade ménima que ele vai gerar por regra                