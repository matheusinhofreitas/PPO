import numpy as np


INSTANCIA_UNICA = False
SEMENTE_INSTANCIA_UNICA = 51
PASSOS_TREINAMENTO = 10000
USAR_LOG_TENSORBOARD = True # Para ver o log, execute o comando: tensorboard --logdir ./ppo_tensorboard/
#TOTAL_INSTANCIAS = 180#36
#TOTAL_INSTANCIAS = 12
TOTAL_INSTANCIAS = 2

CARREGAR_MODELO = False
CONTINUAR_TREINAMENTO = False
#NOME_MODELO = "MODELO_36_inst_UB=FO_inicial"
#NOME_MODELO = "MODELO_36_inst_1M"
NOME_MODELO = "MODELO_TESTE_ZOO"

TAMANHO = 10
TROCAS = int(np.round(0.3*TAMANHO))

if not INSTANCIA_UNICA:
   SEMENTE_INSTANCIA_UNICA = None
