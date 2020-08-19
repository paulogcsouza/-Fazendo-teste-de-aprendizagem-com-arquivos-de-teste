#!/usr/bin/env bash
#
# APlicação Pybrain (fiz didáticos)
#
#  Email:      paulogcsouza86@hotmail.com
#  Autor:      Paulo Souza                 Manutenção: Paulo Souza
#
# ---------------------------------------------------------------------------- #
#  Histórico:
#   v1.0 17/08//2020 (Começou a funcionar)
#  Testado em: python 3.8.1

# ------------------------------Importações----------------------------------- #
from pybrain.structure import FeedForwardNetwork #estrutura da rede neural 
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit #camadas 
from pybrain.structure import FullConnection # ligação entre as camadas 

# -------------------------------Criando Rede--------------------------------- #
rede = FeedForwardNetwork()

camadaEntrada = LinearLayer(2) #dois neuronios camada de entrada 
camadaOculta = SigmoidLayer(3) #três neuronios camada intermediária 
camadaSaida = SigmoidLayer(1)  #um neuronio camada de saida
bias1 = BiasUnit() #bias para camada oculta
bias2 = BiasUnit() #bias para camada de saida


# --------------------------Add a Rede as camadas----------------------------- #
rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

entradaOculta = FullConnection(camadaEntrada, camadaOculta)
ocultaSaida = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2, camadaSaida)

rede.sortModules()

# ------------------------------print da rede--------------------------------- #
print(rede)
print(entradaOculta.params)
print(ocultaSaida.params)
print(biasOculta.params)
print(biasSaida.params)
