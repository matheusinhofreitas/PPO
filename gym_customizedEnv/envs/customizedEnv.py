import os
import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO

from gym_customizedEnv.envs.Heuristicas.solver import *
from gym_customizedEnv.envs.Heuristicas.Executa_estrategias import *
from gym_customizedEnv.envs.Heuristicas.read import *
from gym_customizedEnv.envs.Heuristicas.calculaLB import *
from gym_customizedEnv.envs.Heuristicas.VND import *

import gym_customizedEnv.envs.variaveis as var

class CustomizedEnv(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env where the agent must learn to go always left. 
  """ 
  #TODO: Adicionar as vizinhanças do problema aqui.
  # Constantes paras as ações possíveis
  ACAO_SWAP_D_linha = 0 #sem restart
  ACAO_SWAP_D_linha_R1 = 1 #com restart
  ACAO_INSERT_D_linha = 2 #sem restart
  ACAO_INSERT_D_linha_R1 = 3 #com restart

  ACAO_SWAP_P_linha = 4 #sem restart
  ACAO_SWAP_P_linha_R1 = 5 #com restart
  ACAO_INSERT_P_linha = 6 #sem restart
  ACAO_INSERT_P_linha_R1 = 7 #com restart
  
  NUMERO_ACOES = 8
  
  instancia = None
  solucao = None
  instancia_selecionada = 0
  #função que calcula a FO do problema e retorna o Valor da FO  
  def Avalia(self, solucao):
    FO = 0
    solucao.calcula_funcaoObjetivo(self.instancia)
    FO = solucao.FO
    return FO

  #Chamar a função construtiva do problema e retorna a solucao construtiva.
  def Construtiva (self):
    vnd = VND(self.instancia)
    solucao = vnd.retornaSolucaoInicial()    
    
    return solucao
  



  #Vizinhanças do problema, retornando a melhor solucão encontrada
  def Solver (self, solucao, Escolha):
    vnd = VND(self.instancia)   
    new_solucao = Solucao( self.instancia.n_tarefas, self.instancia.n_maquinas, self.instancia.n_veiculos)
     
    if Escolha == CustomizedEnv.ACAO_SWAP_D_linha:
      new_solucao = vnd.Swap (solucao.ordemInsercaoRotas,solucao,0,0,var.TROCAS)

    if Escolha == CustomizedEnv.ACAO_SWAP_D_linha_R1:
      new_solucao = vnd.Swap (solucao.ordemInsercaoRotas,solucao,1,0, var.TROCAS)

    if Escolha == CustomizedEnv.ACAO_INSERT_D_linha:
      new_solucao = vnd.Insertion (solucao.ordemInsercaoRotas,solucao,0,0, var.TROCAS)

    if Escolha == CustomizedEnv.ACAO_INSERT_D_linha_R1:
      new_solucao = vnd.Insertion (solucao.ordemInsercaoRotas,solucao,1,0, var.TROCAS)

    if Escolha == CustomizedEnv.ACAO_SWAP_P_linha:
      new_solucao = vnd.Swap (solucao.ordemInsercaoMaquinas,solucao,0,1,var.TROCAS)
 
    if Escolha == CustomizedEnv.ACAO_SWAP_P_linha_R1:
      new_solucao = vnd.Swap (solucao.ordemInsercaoMaquinas,solucao,1,1,var.TROCAS)

    if Escolha == CustomizedEnv.ACAO_INSERT_P_linha:
      new_solucao = vnd.Insertion (solucao.ordemInsercaoMaquinas,solucao,0,1,var.TROCAS)

    if Escolha == CustomizedEnv.ACAO_INSERT_P_linha_R1:
      new_solucao = vnd.Insertion (solucao.ordemInsercaoMaquinas,solucao,1,1,var.TROCAS)   
    
    lista = self.GeraListaD(new_solucao)
    #new_solucao.calcula_funcaoObjetivo(self.instancia)
    return lista, new_solucao.FO, new_solucao
    #GAP = round(100*((UB-LB)/UB))
    
  #lêr uma nova instancia do problema
  #TODO: Ler apenas 1 instancia por vez.
  def criar_instancia(self, instancia_selecionada):
    #path = "./Instancias/Testes"#_3"
    path = os.path.join(os.path.dirname(__file__)) + "/../../Instancias/N_10"#_3"
    files = listar_arquivos(path)
    files.sort()
    instancia_selecionada = int(instancia_selecionada % var.TOTAL_INSTANCIAS)
    local = path +"/"+ files[instancia_selecionada]
    print(files[instancia_selecionada])     
    #carrega a instancia para memoria
    self.instancia = Instancia(local)
    #print("Instancia;"+files[instancia_selecionada]+";LB; "+ str(self.instancia.LB) + ";UB; "+ str(self.instancia.UB))
    #print("LB; "+ str(self.instancia.LB))
    #print("UB: "+ str(self.instancia.UB))
   

  #cria a solucao inicial + a Lista que tem a mesma composição da listaD
  #CompletionTime das tarefas + Data de entrega das tarefas + Atraso ponderado das tarefas  
  def usar_instancia(self):
    self.LB = self.instancia.LB
    solucao = self.Construtiva()
    self.FO_inicial = solucao.FO
    atraso_ponderado = [0]*(self.instancia.n_tarefas+1)
    for i in range(0, self.instancia.n_tarefas+1):
        atraso_ponderado[i] = int(solucao.atrasoTarefa[i] * self.instancia.penalidade_atraso[i]) 

    n = self.instancia.n_tarefas + 1
    self.Lista = [0]*n*3
    for i in range(0,n):
      self.Lista[i] = int(solucao.completionTime[i])
    j = 0
    for i in range(n,2*n):
      self.Lista[i] = int(solucao.dataEntrega[j])
      j+=1

    j = 0
    for i in range(2*n,len(self.Lista)):
      self.Lista[i] = int(atraso_ponderado[j])
      j+=1
     
    self.FO = solucao.FO
    self.FO_Best = solucao.FO#self.FO_Best_inicial
    return solucao

  #Funcao que gera a listaD, a listaD é composta pelos CompletionTime das tarefas + Data de entrega das tarefas + Atraso ponderado das tarefas
  def GeraListaD(self, solucao):

    n = self.instancia.n_tarefas
    listaD = [0]*n*3 + [0] + [0]
    
    listaD[0] = self.instancia.n_maquinas
    listaD[1] = self.instancia.n_tarefas
    
    j = 1
    for i in range(2,n+2):
      listaD[i] = int(solucao.completionTime[j])
      j+=1
    j = 1
    for i in range(n+2,2*n + 2 ):
      listaD[i] = int(solucao.dataEntrega[j])
      j+=1

    j = 1
    for i in range(2*n + 2,len(listaD)):
      listaD[i] = int(solucao.atrasoTarefa[j] * self.instancia.penalidade_atraso[j])
      j+=1
    

    return listaD

  def __init__(self, instancia_unica=False, seed=None):
    super(CustomizedEnv, self).__init__()

    print(f"Criando ambiente: {instancia_unica=} {seed=}" )  
    
    self.max_iter_sem_melhoras = var.TAMANHO
    tam = (var.TAMANHO)*3 + 2
    #tam = TAMANHO
    self.Dmax = 10*tam #TODO: Ver o que vai alterar aqui
    self.FOmax = tam*self.Dmax #TODO: Ver o que vai alterar aqui
    
    
    # Define action and observation space
    self.action_space = spaces.Discrete(CustomizedEnv.NUMERO_ACOES)
    self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(tam,), dtype=np.float64)

    self.seed(seed)
    self.instancia_unica = instancia_unica
    
    if not self.instancia_unica: 
      self.criar_instancia(self.instancia_selecionada)
      self.instancia_selecionada += 1

    self.iter_sem_melhoras = 0
    self.passo = 0
    self.ultima_acao = None
    self.ultima_recompensa = None    
  
  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    
    if not self.instancia_unica: 
      self.criar_instancia(self.instancia_selecionada)
      self.instancia_selecionada += 1
    else:
      self.criar_instancia(self.instancia_selecionada)  
    self.solucao = self.usar_instancia()
    
    #self.instancia.UB = self.solucao.FO    
    print("Instancia;"+self.instancia.nome+";LB; "+ str(self.instancia.LB) + ";UB; "+ str(self.instancia.UB))

    self.iter_sem_melhoras = 0
    self.DeltaBest = 0 #distancia da melhor solucão   
    self.passo = 0
    self.ultima_acao = None
    self.ultima_recompensa = None

    self.ListaD = self.GeraListaD(self.solucao)

    #return self.normalizar_estado([self.FO,self.FO_Best,self.DeltaBest,self.iter_sem_melhoras]+self.ListaD)
    #return self.normalizar_estado([self.iter_sem_melhoras]+self.ListaD)
    return self.normalizar_estado(self.ListaD)
  
    #TODO:Conversar com o Thiago para ver a questão de normalização de estados.
  def normalizar_estado(self, estado):
    # Precisamos pegar cada campo do estado e reescalonar seus valores mínimos e máximos para o intervalo [-1,1]
    estado_temp = estado.copy()
    #n de iteraçoes sem melhora
    #numero de máquinas
    estado_temp[0] =  (estado_temp[0] - 2) / (8 - 2)
    estado_temp[1] =  (estado_temp[1] - 10) / (50 - 10)

    n = self.instancia.n_tarefas
    #completionTime
    j = 1
    for i in range(2,n+2):
       estado_temp[i] = (estado_temp[i] - self.instancia.menor_completion_time[j])/ (self.instancia.maior_completion_time[j] - self.instancia.menor_completion_time[j])
       j+=1
    
    #dataEntrega
    j = 1
    for i in range(n+2,(2*n)+2):
       estado_temp[i] = (estado_temp[i] - self.instancia.menor_data_entrega[j] )/(self.instancia.maior_data_entrega[j] - self.instancia.menor_data_entrega[j])
       j+=1


    #atrasoTarefa
    j = 1
    for i in range((2*n)+2,len(estado_temp)):
       estado_temp[i] = (estado_temp[i] - self.instancia.menor_atraso_ponderado[j] ) / (self.instancia.maior_atraso_ponderado[j] - self.instancia.menor_atraso_ponderado[j])
       j+=1
    

    return np.clip(np.array(estado_temp)*2 - 1, self.observation_space.low, self.observation_space.high) 
     
  #TODO: ver como vai Ficar essa função aqui....
  def step(self, action):
    self.FO_anterior = self.FO
    self.Lista, self.FO, self.solucao = self.Solver(copy.deepcopy(self.solucao), action)
    #print("self.Lista :", self.Lista)
    self.ultima_acao = action
    self.ListaD = self.GeraListaD(self.solucao) #TODO: Tirar pois a listaD é igual a self.Lista 1
    #print("self.ListaD:", self.ListaD)
    #print()
    #recompensa = float(self.FO_anterior - self.FO)
    recompensa = float(-1* (self.FO - self.instancia.LB)/(self.instancia.UB - self.instancia.LB)*100 )
    #print("FO;"+str(self.FO) +";\t Recompensa;" + str(recompensa))
    #print(recompensa)

    self.ultima_recompensa = recompensa
    #recompensa = float(self.FO_Best - self.FO)

    if self.FO_Best > self.FO:
      self.FO_Best = self.FO
 
    self.iter_sem_melhoras += 1
      #self.DeltaBest = self.FO - self.FO_Best

    # if sum(self.ListaD) != self.FO:
    #   print(sum(self.ListaD))
    #   print(self.Lista)
    #   print(self.Avalia(self.Lista))
    #   print(self.FO)
    #   os.system("PAUSE")
    
    terminou_episodio = bool(self.FO == self.LB or self.iter_sem_melhoras == self.max_iter_sem_melhoras)

    # Optionally we can pass additional info, we are not using that for now
    info = {}

    self.passo += 1

    #return self.normalizar_estado([self.FO, self.FO_Best,self.DeltaBest, self.iter_sem_melhoras]+self.ListaD), recompensa, terminou_episodio, info
    #return self.normalizar_estado([self.iter_sem_melhoras]+self.ListaD), recompensa, terminou_episodio, info
    return self.normalizar_estado(self.ListaD), recompensa, terminou_episodio, info
  
  def render(self, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    
    if (self.passo > 0): 
      print(f'Passo {self.passo}') 
    else: 
      print('Instância:')
    
    print(f'\tÚltima ação: {self.ultima_acao}, FO: {self.FO}, FO_Best: {self.FO_Best}, Ism: {self.iter_sem_melhoras}, DeltaBest: {self.DeltaBest}')
    print(f'\tLista: {self.Lista}')
    print(f'\tListaD: {self.ListaD}')
    print(f'\tRecompensa: {self.ultima_recompensa}')

  def close(self):
    pass
  
  def seed(self, seed=None):
    self.rand_generator = np.random.RandomState(seed)
    self.action_space.seed(seed)
