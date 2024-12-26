from RedeNeural import CamadaLinear,RedeNeural,relu,erro_medio
from PIL import Image # Biblioteca para carregar imagens
import os # Biblioteca para navegar em pastas
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle #embaralhar dados


def imagem(caminho):
	try:
		img = Image.open(caminho)
		img.verify()
		return True
	except (IOError, SyntaxError):
		return False

def dividir_em_12(img, dim=(28,28)):
	largura = 510
	altura = 511
	partes = []
	for linha in range(3):
		for coluna in range(4):
			esquerda = coluna*largura
			topo = linha*altura
			direita = esquerda + largura
			baixo = topo + altura
			pedaco = img.crop(
			(esquerda, topo, direita, baixo)
			).resize(dim)
			partes.append(pedaco)
	return partes

def funcao_temperatura(fotoNum):
	if fotoNum < 62: 
		return (fotoNum-5)*0.3 + 40
	else: 
		return (fotoNum-62)*0.05 + 57	

def gerar_amostras(caminho, modo='L'):
	amostras = []
	for root, dirs, files in os.walk(caminho):
		for file in files:
			caminho_img = f'{root}\\{file}'
			if '.' in file and imagem(caminho_img):
				fotoNum = int(
                file.split('.')[0])
				temp = funcao_temperatura(
                fotoNum)
				for img in dividir_em_12(
					Image.open(caminho_img)
					):
					amostras.append(
					[np.array(
					img.convert(modo)
					)/255.0, 
					temp])
	return amostras


amostras = gerar_amostras('data')
div=0.9975
num_amostras = 8
N = num_amostras

amostras = amostras[:int(div*len(amostras))]
shuffle(amostras)

teste = amostras[int(div*len(amostras)):]
amostras = amostras[:(len(amostras)//N)*N]
print("amostras ", len(amostras))
print("teste ", len(teste))

ta = 0.00001
N = num_amostras
epocas = 1000
camadas = [
    CamadaLinear(28 * 28, 128, ativacao=relu),
    CamadaLinear(128, 64, ativacao=relu),
    CamadaLinear(64, 1, ativacao=None)
]
rede = RedeNeural(camadas, ta=ta)

# Treino
for epoc in range(epocas):
    erro_epoca = []
    shuffle(amostras)
    for i in range(0, len(amostras), N):
        x = np.array([z[0] for z in amostras[i:i+N]])
        y = np.array([z[1] for z in amostras[i:i+N]])
        x = x.reshape(N, 28*28)
        y = y.reshape(N, 1)
        saida = rede.processar(x)
        erro = erro_medio(saida, y)
        erro_ = erro_medio(saida, y, incremento=True)
        rede.corrigir(erro_)
        erro_epoca.append(erro)
    print(f"Epoca {epoc + 1} Erro: {np.mean(erro_epoca)}")


#Teste
x = np.array([z[0] for z in teste])
y = np.array([z[1] for z in teste])
x = x.reshape(-1, 28*28)
y = y.reshape(-1, 1)
saida = rede.processar(x)
erro = erro_medio(saida, y)
acc = sum(
	abs(saida-y) < 0.5
	) / y.shape[0]
print(f"Erro no teste: {erro.mean()}")
print(f"Acuracia no teste: {acc}")


# Visualizar Resultados
fig, axs = plt.subplots(2, 4)
n=0
x = x.reshape(-1,28,28,1)
for linha in range(2):
	for coluna in range(4):
		axs[linha][coluna].imshow(x[n], cmap='gray')
		axs[linha][coluna].axis('off')
		axs[linha][coluna].set_title(
			f'{y[n].round(1)}|{saida[n].round(1)}'
			)
		n+=1
plt.show()


# Scatter Plot
dataset = amostras + teste
X = np.array(
	[z[0] for z in dataset]
	).reshape(-1,28*28)
Y = np.array(
	[z[1] for z in dataset]
	).reshape(-1,1)
preds = rede.processar(X)
plt.scatter(Y, preds, color='green')
plt.plot(Y, Y, color='blue', linestyle='--') 
plt.show()

