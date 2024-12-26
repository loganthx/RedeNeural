import numpy as np
from RedeNeural import CamadaLinear, RedeNeural, relu
from tensorflow.keras.datasets import mnist

def entropia_binaria_cruzada(saida,val_esperado,incremento=False):
    out = saida
    gt = val_esperado
    epsilon = 1e-8 
    out = np.clip(out, epsilon, 1 - epsilon)
    if incremento:
        return (out - gt) / (len(gt) * out * (1 - out))
    else: 
        return -np.mean(gt * np.log(out) 
        + (1 - gt) * np.log(1 - out))

def sigmoid(z, incremento=False):
    s = 1 / (1 + np.exp(-z))
    if incremento:
        return s * (1 - s)
    else:
        return s

 
(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()

treino_filtro = (y_treino==0) | (y_treino==1)
teste_filtro = (y_teste==0) | (y_teste==1)
x_treino = x_treino[treino_filtro]
y_treino = y_treino[treino_filtro]
x_teste = x_teste[teste_filtro] 
y_teste =  y_teste[teste_filtro]

x_treino = x_treino.reshape(-1, 28 * 28) / 255.0
x_teste = x_teste.reshape(-1, 28 * 28) / 255.0
y_treino = y_treino.astype(float).reshape(-1, 1)
y_teste = y_teste.astype(float).reshape(-1, 1)

num_amostras = 16
ta = 0.0001
N = num_amostras
epocas = 10
camadas = [
    CamadaLinear(28 * 28, 128, ativacao=relu),
    CamadaLinear(128, 64, ativacao=relu),
    CamadaLinear(64, 1, ativacao=sigmoid)
]
rede = RedeNeural(camadas, ta=ta)

# Treino
for epoc in range(epocas):
    erro_epoca = []
    indices = np.random.permutation(len(x_treino))
    x_treino, y_treino = x_treino[indices], y_treino[indices]
    for i in range(0, len(x_treino), num_amostras):
        x_batch = x_treino[i:i + num_amostras]
        y_batch = y_treino[i:i + num_amostras]
        saida = rede.processar(x_batch)
        erro = entropia_binaria_cruzada(
        saida, 
        y_batch)
        erro_ = entropia_binaria_cruzada(
        saida, 
        y_batch, incremento=True)
        rede.corrigir(erro_)
        erro_epoca.append(erro)
    print(f"Epoca {epoc + 1} Erro: {np.mean(erro_epoca)}")

# Avalie Treino
teste_saida = rede.processar(x_teste)
teste_erro = entropia_binaria_cruzada(teste_saida, y_teste)
teste_acuracia = np.mean((teste_saida > 0.5) == y_teste)
print(f"Erro Teste: {teste_erro}, Teste Acuracia {teste_acuracia}")