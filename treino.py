import time
import copy
import torch
from torch import nn

def escolher_dispositivo():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def avaliar(modelo, loader, dispositivo, guardar=False):
    perda_fn = nn.MSELoss(reduction="sum")
    modelo.eval()
    soma_perda = 0.0
    total = 0
    previsoes = []
    reais = []
    entradas = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(dispositivo)
            y = y.to(dispositivo)
            saida = modelo(X)
            soma_perda += perda_fn(saida, y).item()
            total += y.numel()
            if guardar:
                previsoes.append(saida.cpu())
                reais.append(y.cpu())
                entradas.append(X[:, -24:, :].cpu())
    if guardar:
        return soma_perda / total, torch.cat(previsoes), torch.cat(reais), torch.cat(entradas)
    return soma_perda / total

def treinar(modelo, loader_tr, loader_val, epocas, lr, dispositivo, mostrar=True):
    modelo.to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    perda_fn = nn.MSELoss()

    hist = {}
    hist["epoca"] = []
    hist["perda_treino"] = []
    hist["perda_val"] = []
    hist["tempo"] = []

    melhor_perda = None
    melhor_epoca = 0
    melhores_pesos = None
    tempo_total = 0.0

    ep = 1
    while ep <= epocas:
        modelo.train()
        inicio = time.time()
        soma = 0.0
        total = 0
        for X, y in loader_tr:
            X = X.to(dispositivo)
            y = y.to(dispositivo)
            otimizador.zero_grad()
            saida = modelo(X)
            perda = perda_fn(saida, y)
            perda.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            soma += perda.item() * len(y)
            total += len(y)
        if dispositivo == "cuda":
            torch.cuda.synchronize()
        duracao = time.time() - inicio
        tempo_total += duracao

        perda_val = avaliar(modelo, loader_val, dispositivo)

        hist["epoca"].append(ep)
        hist["perda_treino"].append(soma / total)
        hist["perda_val"].append(perda_val)
        hist["tempo"].append(duracao)

        if melhor_perda is None or perda_val < melhor_perda:
            melhor_perda = perda_val
            melhor_epoca = ep
            melhores_pesos = copy.deepcopy(modelo.state_dict())

        if mostrar:
            print(f"  época {ep:2d}  perda treino={soma / total:.4f}  perda val={perda_val:.4f}  ({duracao:.1f} s)")
        ep += 1

    hist["melhor_epoca"] = melhor_epoca
    hist["melhor_perda_val"] = melhor_perda
    hist["tempo_total"] = tempo_total
    hist["tempo_epoca"] = tempo_total / epocas
    modelo.load_state_dict(melhores_pesos)
    return hist
