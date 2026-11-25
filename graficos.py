import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dados import N_VAR, PASSOS_TOKEN, NOMES_CURTOS

PASTA_FIG = "resultados/figuras"

AZUL = '#0275d8'
VERMELHO = '#d9534f'
CINZA = '#777777'
VERDE = '#5cb85c'

def salvar_figura(nome):
    if not os.path.exists(PASTA_FIG):
        os.makedirs(PASTA_FIG)
    plt.savefig(PASTA_FIG + "/" + nome + ".pdf", bbox_inches="tight")
    plt.savefig(PASTA_FIG + "/" + nome + ".png", dpi=200, bbox_inches="tight")
    plt.close()

def grafico_temperatura(datas, temperatura, inicio_zoom, fim_zoom):
    fig, eixos = plt.subplots(2, 1, figsize=(6.4, 3.4), dpi=150)
    eixos[0].plot(datas, temperatura, color=AZUL, lw=0.3)
    eixos[0].set_ylabel("T (°C)", fontsize=8)
    eixos[0].tick_params(labelsize=7)
    eixos[0].grid(True, ls="--", alpha=0.4)
    eixos[1].plot(datas[inicio_zoom:fim_zoom], temperatura[inicio_zoom:fim_zoom], color=AZUL, lw=0.8)
    eixos[1].set_ylabel("T (°C)", fontsize=8)
    eixos[1].tick_params(labelsize=7)
    eixos[1].grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    salvar_figura("temperatura")

def grafico_correlacao(corr):
    plt.figure(figsize=(3.8, 3.4), dpi=150)
    plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
    plt.xticks(range(N_VAR), NOMES_CURTOS, rotation=90, fontsize=6)
    plt.yticks(range(N_VAR), NOMES_CURTOS, fontsize=6)
    salvar_figura("correlacao")

def grafico_perdas(hists, titulos, nome):
    fig, eixos = plt.subplots(1, len(hists), figsize=(6.4, 2.3), dpi=150)
    i = 0
    while i < len(hists):
        hist = hists[i]
        eixo = eixos[i]
        eixo.plot(hist["epoca"], hist["perda_treino"], "o-", ms=3, color=AZUL, label="treino")
        eixo.plot(hist["epoca"], hist["perda_val"], "s-", ms=3, color=VERMELHO, label="validação")
        eixo.axvline(hist["melhor_epoca"], color=CINZA, ls="--", lw=0.8, label="menor perda val.")
        eixo.set_xlabel("época", fontsize=8)
        eixo.set_ylabel("MSE (padronizado)", fontsize=8)
        eixo.set_title(titulos[i], fontsize=9)
        eixo.tick_params(labelsize=7)
        eixo.grid(True, ls="--", alpha=0.4)
        eixo.legend(frameon=False, fontsize=7)
        i += 1
    fig.tight_layout()
    salvar_figura(nome)

def grafico_horizonte(mse_modelo, mse_pers, mse_dia):
    horas = []
    k = 0
    while k < len(mse_modelo):
        horas.append(k + 1)
        k += 1
    plt.figure(figsize=(3.4, 2.4), dpi=150)
    plt.plot(horas, mse_modelo, "o-", ms=3, color=AZUL, label="Transformer")
    plt.plot(horas, mse_pers, "s-", ms=3, color=VERMELHO, label="persistência")
    plt.plot(horas, mse_dia, "^-", ms=3, color=CINZA, label="dia anterior")
    plt.xlabel("hora à frente", fontsize=8)
    plt.ylabel("MSE (padronizado)", fontsize=8)
    plt.tick_params(labelsize=7)
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(frameon=False, fontsize=7)
    salvar_figura("mse_horizonte")

def temperatura_de(tokens, media, desvio):
    lista = []
    for tok in tokens:
        p = 0
        while p < PASSOS_TOKEN:
            v = tok[p * N_VAR + 1].item()
            lista.append(v * float(desvio[1]) + float(media[1]))
            p += 1
    return lista

def grafico_previsoes(exemplos, media, desvio):
    fig, eixos = plt.subplots(1, len(exemplos), figsize=(6.4, 2.3), dpi=150)
    i = 0
    while i < len(exemplos):
        ex = exemplos[i]
        passado = temperatura_de(ex["entrada"], media, desvio)
        real = temperatura_de(ex["real"], media, desvio)
        prev = temperatura_de(ex["prev"], media, desvio)
        pers = temperatura_de(ex["persistencia"], media, desvio)
        x_pass = []
        x_fut = []
        k = 0
        while k < len(passado):
            x_pass.append((k - len(passado)) / PASSOS_TOKEN)
            k += 1
        k = 0
        while k < len(real):
            x_fut.append(k / PASSOS_TOKEN)
            k += 1
        eixo = eixos[i]
        eixo.plot(x_pass, passado, color=CINZA, lw=1, label="entrada")
        eixo.plot(x_fut, real, color="black", lw=1, label="real")
        eixo.plot(x_fut, prev, color=AZUL, lw=1.2, label="Transformer")
        eixo.plot(x_fut, pers, color=VERMELHO, lw=1, ls=":", label="persistência")
        eixo.axvline(0, color=CINZA, ls="--", lw=0.6)
        eixo.set_xlabel("horas", fontsize=8)
        eixo.set_ylabel("T (°C)", fontsize=8)
        eixo.set_title(ex["titulo"], fontsize=8)
        eixo.tick_params(labelsize=7)
        eixo.grid(True, ls="--", alpha=0.4)
        i += 1
    eixos[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    salvar_figura("previsoes_24h")

def grafico_mapas(matrizes, titulos, nome):
    fig, eixos = plt.subplots(1, len(matrizes), figsize=(3.2 * len(matrizes), 3.0), dpi=150, squeeze=False)
    i = 0
    while i < len(matrizes):
        eixo = eixos[0][i]
        mat = matrizes[i]
        m = mat.shape[0]
        img = eixo.imshow(mat, cmap="viridis")
        cb = fig.colorbar(img, ax=eixo, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6)
        rotulos = []
        posicoes = []
        k = m - 1
        while k >= 0:
            posicoes.append(k)
            rotulos.append(str(k - m + 1))
            k -= 12
        eixo.set_xticks(posicoes)
        eixo.set_xticklabels(rotulos, fontsize=6)
        eixo.set_yticks(posicoes)
        eixo.set_yticklabels(rotulos, fontsize=6)
        eixo.set_xlabel("chave (hora relativa)", fontsize=7)
        eixo.set_ylabel("consulta (hora relativa)", fontsize=7)
        eixo.set_title(titulos[i], fontsize=8)
        i += 1
    fig.tight_layout()
    salvar_figura(nome)

def grafico_defasagem(listas, rotulos, nome):
    plt.figure(figsize=(3.4, 2.4), dpi=150)
    cores = [AZUL, VERMELHO, CINZA, VERDE, "#f0ad4e", "#6f42c1", "#20c997", "#343a40"]
    i = 0
    while i < len(listas):
        x = []
        k = 0
        while k < len(listas[i]):
            x.append(k)
            k += 1
        plt.plot(x, listas[i], "o-", ms=2, lw=1, color=cores[i % len(cores)], label=rotulos[i])
        i += 1
    plt.xlabel("defasagem em relação ao último token (h)", fontsize=8)
    plt.ylabel("peso médio de atenção", fontsize=8)
    plt.tick_params(labelsize=7)
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(frameon=False, fontsize=6)
    salvar_figura(nome)
