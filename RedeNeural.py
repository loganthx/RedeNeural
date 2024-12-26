import numpy as np

def erro_medio(saida, val_esperado, incremento=False):
    N = val_esperado.shape[0]
    if incremento:
        return (2/N)*(saida - val_esperado)
    else:
        return (1/N)*(saida - val_esperado)**2

def relu(y, incremento=False):
    if incremento:
        return np.where(y>0, 1, 0)
    else:
        return np.where(y>0, y, 0)

def regulador(matriz):
    canais_entrada = matriz.shape[0]
    return matriz*np.sqrt(2/canais_entrada)

class CamadaLinear:
    def __init__(self, canais_entrada, canais_saida, ativacao, regulador=regulador):
        self.pesos = np.random.randn(canais_entrada, canais_saida)
        self.propensoes = np.zeros((1, canais_saida))
        if regulador:
            self.pesos = regulador(self.pesos)
            self.propensoes = regulador(self.propensoes)
        self.ativacao = ativacao
    def processar(self, x):
        y = x@self.pesos + self.propensoes
        z = self.ativacao(y) if self.ativacao else y
        return x, y, z
    def corrigir(self, dPeso, dProp, ta):
        self.pesos -= ta*dPeso
        self.propensoes -= ta*dProp
        

class RedeNeural:
    def __init__(self, camadas, ta):
        self.ta = ta
        self.camadas = camadas
        self.agenda = {}
        for n, camada in enumerate(self.camadas):
            self.agenda[f'camada{n}'] = {'x': None, 'y':None, 'z':None}
    def processar(self, x):
        for n, camada in enumerate(self.camadas):
            x, y, z = camada.processar(x)
            self.agenda[f'camada{n}']['x'] = x
            self.agenda[f'camada{n}']['y'] = y
            self.agenda[f'camada{n}']['z'] = z
            x = z
        return z   
    def corrigir(self, erro_):
        N = len(self.camadas)
        for n in reversed(range(N)):
            if n+1==N:
                if self.camadas[n].ativacao:
                    Z_ = erro_*self.camadas[n].ativacao(
                    self.agenda[f'camada{n}']['y'], 
                    incremento=True)
                    dProp = Z_.mean(axis=0, keepdims=True)
                    dPeso = (Z_.T@self.agenda[f'camada{n}']['x']).T
                    self.camadas[n].corrigir(dPeso, dProp, self.ta)
                else:  
                    Z_ = erro_ 
                    dProp = Z_.mean(axis=0, keepdims=True)
                    dPeso = (Z_.T @ self.agenda[f'camada{n}']['x']).T
                    self.camadas[n].corrigir(dPeso, dProp, self.ta)
            else:
                if self.camadas[n].ativacao:
                    Z_ = Z_@(self.camadas[n+1].pesos).T 
                    Z_ = Z_*self.camadas[n].ativacao(
                    self.agenda[f'camada{n}']['y'], 
                    incremento=True
                    )
                    dProp = Z_.mean(axis=0, keepdims=True)
                    dPeso = (Z_.T@self.agenda[f'camada{n}']['x']).T
                    self.camadas[n].corrigir(dPeso, dProp, self.ta)
                else:
                    Z_ = Z_@(self.camadas[n+1].pesos).T
                    dProp = Z_.mean(axis=0, keepdims=True)
                    dPeso = (Z_.T@self.agenda[f'camada{n}']['x']).T
                    self.camadas[n].corrigir(dPeso, dProp, self.ta)  