
#!/usr/bin/env bash
#
# 
# CFombinação de Classificadores
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#   v1.0 02/09//2020 (Começou a funcionar)
#  Testado em: python 3.8.1
# ------------------------------Importações----------------------------------- #
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# ---------------------Carregamento das Aprendizagem ------------------------- #
svm = pickle.load(open('svm_finalizado.sav', 'rb'))
random_forest = pickle.load(open('random_forest_finalizado.sav', 'rb'))
mlp = pickle.load(open('mlp_finalizado.sav', 'rb'))

# ------------------------Testes e Escalonamento ----------------------------- #
novo_registro = [[50000, 40, 5000]]                     # Novo registro 
novo_registro = np.asarray(novo_registro)               # transforma np.array   
novo_registro = novo_registro.reshape(-1, 1)            # transforma coluna em linhas  
scaler = StandardScaler()                               # Escalonamento 
novo_registro = scaler.fit_transform(novo_registro)     # Escalonamento 
novo_registro = novo_registro.reshape(-1, 3)            # Volta pra coluna 

# -------------------------Aplicando Aprendizagem----------------------------- #
resposta_svm = svm.predict(novo_registro)
resposta_random_forest = random_forest.predict(novo_registro)
resposta_mlp = mlp.predict(novo_registro)
paga = 0
nao_paga = 0


# -----------------------------Classificação---------------------------------- #
if resposta_svm[0] == 1:
    paga += 1
else:
    nao_paga += 1
    
if resposta_random_forest[0] == 1:
    paga += 1
else:
    nao_paga += 1
    
if resposta_mlp[0] == 1:
    paga += 1
else:
    nao_paga += 1
    
if paga > nao_paga:
    print('Cliente pagará o empréstimo')
elif paga == nao_paga:
    print('Resultado empatado')
else:
    print('Cliente não pagará o empréstimo')