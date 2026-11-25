import math
import torch
from torch import nn
from dados import N_VAR, PASSOS_TOKEN

def codificacao_posicional(m, n):
    pe = torch.zeros(m, n)
    p = 0
    while p < m:
        i = 0
        while i < n:
            ang = p / (10000 ** (i / n))
            pe[p][i] = math.sin(ang)
            if i + 1 < n:
                pe[p][i + 1] = math.cos(ang)
            i += 2
        p += 1
    return pe

class AtencaoMultiCabeca(nn.Module):
    def __init__(self, n, cabecas):
        super().__init__()
        self.n = n
        self.cabecas = cabecas
        self.dk = n // cabecas
        self.W_q = nn.Linear(n, n)
        self.W_k = nn.Linear(n, n)
        self.W_v = nn.Linear(n, n)
        self.W_o = nn.Linear(n, n)
        self.pesos = None

    def separar(self, x):
        lote, m, _ = x.shape
        x = x.reshape(lote, m, self.cabecas, self.dk)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        lote, m, _ = x.shape
        Q = self.separar(self.W_q(x))
        K = self.separar(self.W_k(x))
        V = self.separar(self.W_v(x))
        escores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)
        pesos = torch.softmax(escores, dim=-1)
        self.pesos = pesos.detach()
        saida = torch.matmul(pesos, V)
        saida = saida.permute(0, 2, 1, 3).reshape(lote, m, self.n)
        return self.W_o(saida)

class BlocoCodificador(nn.Module):
    def __init__(self, n, cabecas, dropout):
        super().__init__()
        self.atencao = AtencaoMultiCabeca(n, cabecas)
        self.norma1 = nn.LayerNorm(n)
        self.ffn = nn.Sequential(nn.Linear(n, 4 * n), nn.ReLU(), nn.Linear(4 * n, n))
        self.norma2 = nn.LayerNorm(n)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norma1(x + self.dropout(self.atencao(x)))
        x = self.norma2(x + self.dropout(self.ffn(x)))
        return x

class TransformerClima(nn.Module):
    def __init__(self, m, n, cabecas, blocos, h, dropout=0.1):
        super().__init__()
        self.m = m
        self.h = h
        self.dim_token = N_VAR * PASSOS_TOKEN
        self.projecao = nn.Linear(self.dim_token, n)
        self.register_buffer("pe", codificacao_posicional(m, n))
        self.dropout = nn.Dropout(dropout)
        lista = []
        b = 0
        while b < blocos:
            lista.append(BlocoCodificador(n, cabecas, dropout))
            b += 1
        self.blocos = nn.ModuleList(lista)
        self.saida = nn.Linear(n, h * self.dim_token)

    def representacao(self, X):
        z = self.projecao(X) + self.pe
        z = self.dropout(z)
        for bloco in self.blocos:
            z = bloco(z)
        return z

    def forward(self, X):
        z = self.representacao(X)
        ultimo = z[:, -1, :]
        y = self.saida(ultimo)
        return y.reshape(len(X), self.h, self.dim_token)

    def pesos_atencao(self):
        lista = []
        for bloco in self.blocos:
            lista.append(bloco.atencao.pesos)
        return lista

def contar_parametros(modelo):
    total = 0
    for p in modelo.parameters():
        if p.requires_grad:
            total += p.numel()
    return total

def nome_config(c):
    return "m=" + str(c["m"]) + " n=" + str(c["n"]) + " cab=" + str(c["cabecas"]) + " blocos=" + str(c["blocos"])
