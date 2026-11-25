import os
import json
import time
import torch
from dados import SEMENTE, N_VAR, PASSOS_TOKEN, VARIAVEIS, NOMES_CURTOS, preparar_dados, criar_loader
from modelos import TransformerClima, contar_parametros, nome_config
from treino import escolher_dispositivo, treinar, avaliar
from metricas import calcular_metricas, baseline_persistencia, baseline_dia_anterior
from mapas import coletar_pesos, peso_por_defasagem, entropia_media
from tabelas import numero_br, salvar_tabela
from graficos import grafico_temperatura, grafico_correlacao, grafico_perdas, grafico_horizonte, grafico_previsoes, grafico_mapas, grafico_defasagem

RODAR_BUSCA = True
CONFIG_FIXA = {"m": 48, "n": 64, "cabecas": 4, "blocos": 2}

REFERENCIA = {"m": 48, "n": 64, "cabecas": 4, "blocos": 2}
VARIACOES = [
    ["m", 24],
    ["m", 72],
    ["n", 32],
    ["n", 128],
    ["cabecas", 2],
    ["cabecas", 8],
    ["blocos", 1],
    ["blocos", 3],
]
EPOCAS_BUSCA = 4
PASSO_BUSCA = 3

EPOCAS = 15
LOTE = 128
LR = 0.001
DROPOUT = 0.1
N_AMOSTRAS_MAPA = 1000
OBJETIVOS = [1, 24]

def fixar_semente():
    torch.manual_seed(SEMENTE)

def descrever_dados(d):
    print("\n=== Dados")
    info = d["info"]
    print(f"registros brutos={info['registros_brutos']}  limpos={info['registros_limpos']}  de {info['inicio']} a {info['fim']}")
    print(f"erros de vento (-9999)={info['erros_vento']}  datas repetidas={info['datas_repetidas']}  buracos={info['buracos']}")
    print(f"tokens: treino={len(d['tr'])}  validação={len(d['val'])}  teste={len(d['te'])}")

    linhas = []
    for r in d["estatisticas"]:
        linhas.append([r["variavel"].replace("**3", "$^3$").replace("%", "\\%"), numero_br(r["media"], 2), numero_br(r["desvio"], 2),
                       numero_br(r["minimo"], 2), numero_br(r["maximo"], 2)])
    salvar_tabela("estatisticas", ["variável", "média", "desvio", "mínimo", "máximo"], linhas,
                  "Estatísticas das 14 variáveis após a limpeza (leituras de 10 minutos).", "tab:estatisticas")

    inicio = 0
    while inicio < len(d["datas"]) and str(d["datas"][inicio])[:10] != "2015-06-01":
        inicio += 1
    grafico_temperatura(d["datas"], d["temperatura"], inicio, inicio + 14 * 24)
    grafico_correlacao(d["correlacoes"])

    saida = dict(info)
    saida["tokens_treino"] = len(d["tr"])
    saida["tokens_validacao"] = len(d["val"])
    saida["tokens_teste"] = len(d["te"])
    saida["inicio_validacao"] = str(d["horas_val"][0])
    saida["inicio_teste"] = str(d["horas_te"][0])
    saida["estatisticas"] = d["estatisticas"]
    saida["correlacoes"] = d["correlacoes"]
    return saida

def criar_modelo(c, h):
    fixar_semente()
    return TransformerClima(c["m"], c["n"], c["cabecas"], c["blocos"], h, DROPOUT)

def busca(d, dispositivo):
    print("\n=== Busca de hiperparâmetros (objetivo 24 h)")
    configs = [dict(REFERENCIA)]
    for v in VARIACOES:
        c = dict(REFERENCIA)
        c[v[0]] = v[1]
        configs.append(c)

    resultados = []
    melhor = None
    for c in configs:
        loader_tr = criar_loader(d["tr"], c["m"], 24, LOTE, True, PASSO_BUSCA)
        loader_val = criar_loader(d["val"], c["m"], 24, 512, False)
        modelo = criar_modelo(c, 24)
        hist = treinar(modelo, loader_tr, loader_val, EPOCAS_BUSCA, LR, dispositivo, mostrar=False)
        r = {}
        r["config"] = c
        r["parametros"] = contar_parametros(modelo)
        r["perda_val"] = hist["melhor_perda_val"]
        r["melhor_epoca"] = hist["melhor_epoca"]
        r["tempo"] = hist["tempo_total"]
        resultados.append(r)
        print(f"{nome_config(c)}  parâmetros={r['parametros']}  perda val={r['perda_val']:.4f}  tempo={r['tempo']:.1f} s")
        if melhor is None or r["perda_val"] < resultados[melhor]["perda_val"]:
            melhor = len(resultados) - 1

    linhas = []
    i = 0
    while i < len(resultados):
        r = resultados[i]
        c = r["config"]
        if i == 0:
            mudou = "referência"
        else:
            mudou = VARIACOES[i - 1][0]
            if mudou == "cabecas":
                mudou = "cabeças"
        marca = ""
        if i == melhor:
            marca = " *"
        linhas.append([mudou + marca, str(c["m"]), str(c["n"]), str(c["cabecas"]), str(c["blocos"]), numero_br(r["parametros"]),
                       numero_br(r["perda_val"], 4), numero_br(r["tempo"], 0)])
        i += 1
    salvar_tabela("busca", ["variação", "$m$", "$n$", "cabeças", "blocos", "parâmetros", "MSE val.", "tempo (s)"], linhas,
                  "Busca de hiperparâmetros no objetivo de 24 h (" + str(EPOCAS_BUSCA) + " épocas, 1 a cada " + str(PASSO_BUSCA) +
                  " janelas de treino). Cada linha altera um único valor da referência; * = escolhida.", "tab:busca")
    print("escolhida: " + nome_config(resultados[melhor]["config"]))
    return resultados, resultados[melhor]["config"]

def rodar_objetivo(d, c, h, dispositivo):
    print(f"\n=== Objetivo: próximos {h} token(s)")
    loader_tr = criar_loader(d["tr"], c["m"], h, LOTE, True)
    loader_val = criar_loader(d["val"], c["m"], h, 512, False)
    loader_te = criar_loader(d["te"], c["m"], h, 512, False)
    modelo = criar_modelo(c, h)
    print(f"parâmetros: {contar_parametros(modelo)}  janelas de treino: {len(loader_tr.dataset)}  teste: {len(loader_te.dataset)}")
    hist = treinar(modelo, loader_tr, loader_val, EPOCAS, LR, dispositivo)
    print(f"menor perda de validação na época {hist['melhor_epoca']}; tempo total {hist['tempo_total']:.1f} s")

    perda_te, prev, real, entradas = avaliar(modelo, loader_te, dispositivo, guardar=True)
    r = {}
    r["parametros"] = contar_parametros(modelo)
    r["hist"] = hist
    r["janelas_treino"] = len(loader_tr.dataset)
    r["janelas_teste"] = len(loader_te.dataset)
    r["modelo"] = calcular_metricas(prev, real, d["desvio"])
    r["persistencia"] = calcular_metricas(baseline_persistencia(entradas, h), real, d["desvio"])
    r["dia_anterior"] = calcular_metricas(baseline_dia_anterior(entradas, h), real, d["desvio"])
    print(f"teste: MSE modelo={r['modelo']['mse']:.4f}  persistência={r['persistencia']['mse']:.4f}  dia anterior={r['dia_anterior']['mse']:.4f}")
    print(f"MAE de T (°C): modelo={r['modelo']['mae_fisico'][1]:.3f}  persistência={r['persistencia']['mae_fisico'][1]:.3f}")

    medias, vistas = coletar_pesos(modelo, loader_te, dispositivo, N_AMOSTRAS_MAPA)
    r["amostras_mapa"] = vistas
    r["entropia"] = []
    r["defasagem"] = []
    for mat_bloco in medias:
        ent = []
        defs = []
        k = 0
        while k < mat_bloco.shape[0]:
            ent.append(entropia_media(mat_bloco[k]))
            defs.append(peso_por_defasagem(mat_bloco[k]))
            k += 1
        r["entropia"].append(ent)
        r["defasagem"].append(defs)
    r["mapas"] = medias

    dados_prev = {}
    dados_prev["prev"] = prev
    dados_prev["real"] = real
    dados_prev["entradas"] = entradas
    return r, dados_prev

def graficos_atencao(res, c):
    ultimo = c["blocos"] - 1
    for h in OBJETIVOS:
        r = res[h]
        mats = [r["mapas"][0][0], r["mapas"][ultimo][0]]
        titulos = ["bloco 1, cabeça 1", "bloco " + str(ultimo + 1) + ", cabeça 1"]
        grafico_mapas(mats, titulos, "mapa_" + str(h) + "h")

        todas = []
        tit = []
        k = 0
        while k < c["cabecas"]:
            todas.append(r["mapas"][ultimo][k])
            tit.append("cabeça " + str(k + 1))
            k += 1
        grafico_mapas(todas, tit, "cabecas_" + str(h) + "h")

        listas = []
        rotulos = []
        k = 0
        while k < c["cabecas"]:
            listas.append(r["defasagem"][ultimo][k])
            rotulos.append("cabeça " + str(k + 1))
            k += 1
        grafico_defasagem(listas, rotulos, "defasagem_" + str(h) + "h")

def exemplos_previsao(d, dados_prev, m):
    n = len(dados_prev["prev"])
    escolhas = [int(n * 0.45), int(n * 0.85)]
    exemplos = []
    for i in escolhas:
        ex = {}
        ex["entrada"] = dados_prev["entradas"][i]
        ex["real"] = dados_prev["real"][i]
        ex["prev"] = dados_prev["prev"][i]
        ex["persistencia"] = baseline_persistencia(dados_prev["entradas"][i:i + 1], 24)[0]
        ex["titulo"] = "previsão a partir de " + str(d["horas_te"][i + m])[:16]
        exemplos.append(ex)
    grafico_previsoes(exemplos, d["media"], d["desvio"])
    return escolhas

def tabelas_finais(res):
    linhas = []
    for h in OBJETIVOS:
        r = res[h]
        nome = "próximo token (1 h)"
        if h == 24:
            nome = "24 tokens (24 h)"
        melhora = 1 - r["modelo"]["mse"] / r["persistencia"]["mse"]
        linhas.append([nome, numero_br(r["modelo"]["mse"], 4), numero_br(r["persistencia"]["mse"], 4),
                       numero_br(r["dia_anterior"]["mse"], 4), numero_br(melhora * 100, 1) + "\\%"])
    salvar_tabela("teste", ["objetivo", "Transformer", "persistência", "dia anterior", "redução"], linhas,
                  "MSE padronizado no teste. Redução = queda do MSE do Transformer em relação à persistência.", "tab:teste")

    linhas = []
    j = 0
    while j < N_VAR:
        linhas.append([NOMES_CURTOS[j], numero_br(res[1]["modelo"]["mae_fisico"][j], 3), numero_br(res[1]["persistencia"]["mae_fisico"][j], 3),
                       numero_br(res[24]["modelo"]["mae_fisico"][j], 3), numero_br(res[24]["persistencia"]["mae_fisico"][j], 3),
                       numero_br(res[24]["dia_anterior"]["mae_fisico"][j], 3)])
        j += 1
    salvar_tabela("mae_variaveis", ["variável", "1 h: modelo", "1 h: persist.", "24 h: modelo", "24 h: persist.", "24 h: dia ant."], linhas,
                  "Erro absoluto médio no teste, nas unidades de cada variável.", "tab:mae")

if __name__ == "__main__":
    if not os.path.exists("resultados"):
        os.makedirs("resultados")
    inicio_total = time.time()
    dispositivo = escolher_dispositivo()
    print(f"dispositivo: {dispositivo}")
    torch.set_num_threads(max(1, os.cpu_count()))

    d = preparar_dados()
    saida = {}
    saida["dispositivo"] = dispositivo
    saida["dados"] = descrever_dados(d)

    if RODAR_BUSCA:
        resultados_busca, c = busca(d, dispositivo)
        saida["busca"] = resultados_busca
    else:
        c = CONFIG_FIXA
    saida["config"] = c

    res = {}
    previsoes = {}
    for h in OBJETIVOS:
        res[h], previsoes[h] = rodar_objetivo(d, c, h, dispositivo)

    grafico_perdas([res[1]["hist"], res[24]["hist"]], ["próximo token (1 h)", "24 tokens (24 h)"], "perdas")
    grafico_horizonte(res[24]["modelo"]["mse_horizonte"], res[24]["persistencia"]["mse_horizonte"], res[24]["dia_anterior"]["mse_horizonte"])
    saida["exemplos_previsao"] = exemplos_previsao(d, previsoes[24], c["m"])
    graficos_atencao(res, c)
    tabelas_finais(res)

    for h in OBJETIVOS:
        r = dict(res[h])
        mapas = []
        for mat_bloco in r["mapas"]:
            mapas.append(mat_bloco.tolist())
        r["mapas"] = mapas
        saida["objetivo_" + str(h) + "h"] = r
    saida["tempo_total"] = time.time() - inicio_total

    arq = open("resultados/resultados.json", "w", encoding="utf-8")
    json.dump(saida, arq, indent=2, ensure_ascii=False, default=str)
    arq.close()
    print("\nfiguras e tabelas salvas na pasta resultados/")
