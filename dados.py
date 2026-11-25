import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

SEMENTE = 42
PASTA_DADOS = "data"
ARQ_ZIP = "jena_climate_2009_2016.csv.zip"
ARQ_CSV = "jena_climate_2009_2016.csv"
URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

PASSOS_TOKEN = 6
FRACAO_TREINO = 0.7
FRACAO_VAL = 0.1

VARIAVEIS = ["p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)", "VPact (mbar)",
             "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)"]
NOMES_CURTOS = ["p", "T", "Tpot", "Tdew", "rh", "VPmax", "VPact", "VPdef", "sh", "H2OC", "rho", "wv", "max. wv", "wd"]
N_VAR = 14

def baixar_jena():
    if not os.path.exists(PASTA_DADOS):
        os.makedirs(PASTA_DADOS)
    csv = PASTA_DADOS + "/" + ARQ_CSV
    if os.path.exists(csv):
        return csv
    zip_local = PASTA_DADOS + "/" + ARQ_ZIP
    if not os.path.exists(zip_local):
        if os.path.exists(ARQ_ZIP):
            zip_local = ARQ_ZIP
        else:
            print("baixando " + URL)
            urllib.request.urlretrieve(URL, zip_local)
    z = zipfile.ZipFile(zip_local)
    for nome in z.namelist():
        if nome.endswith(ARQ_CSV) and not nome.startswith("__MACOSX"):
            arq = open(csv, "wb")
            arq.write(z.read(nome))
            arq.close()
    z.close()
    return csv

def ler_tabela():
    df = pd.read_csv(baixar_jena())
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    return df

def limpar(df):
    info = {}
    info["registros_brutos"] = len(df)
    info["inicio"] = str(df["Date Time"].iloc[0])
    info["fim"] = str(df["Date Time"].iloc[-1])

    info["nulos"] = int(df[VARIAVEIS].isna().sum().sum())
    erros = {}
    for col in ["wv (m/s)", "max. wv (m/s)"]:
        ruins = df[col] < -9000
        erros[col] = int(ruins.sum())
        df.loc[ruins, col] = 0.0
    info["erros_vento"] = erros

    repetidos = df["Date Time"].duplicated()
    info["datas_repetidas"] = int(repetidos.sum())
    df = df[~repetidos]
    df = df.sort_values("Date Time").reset_index(drop=True)

    dif = df["Date Time"].diff().dt.total_seconds()
    buracos = dif > 600
    info["buracos"] = int(buracos.sum())
    falta = 0
    for v in dif[buracos].tolist():
        falta += int(round(v / 600)) - 1
    info["leituras_faltando"] = falta
    info["registros_limpos"] = len(df)
    return df, info

def estatisticas(df):
    linhas = []
    i = 0
    while i < N_VAR:
        col = df[VARIAVEIS[i]]
        r = {}
        r["variavel"] = VARIAVEIS[i]
        r["media"] = float(col.mean())
        r["desvio"] = float(col.std())
        r["minimo"] = float(col.min())
        r["maximo"] = float(col.max())
        linhas.append(r)
        i += 1
    return linhas

def correlacoes(df):
    return df[VARIAVEIS].corr().values.tolist()

def montar_tokens(df):
    valores = df[VARIAVEIS].values.astype(np.float32)
    n_tokens = len(valores) // PASSOS_TOKEN
    valores = valores[:n_tokens * PASSOS_TOKEN]
    tokens = valores.reshape(n_tokens, PASSOS_TOKEN * N_VAR)
    horas = df["Date Time"].iloc[0:n_tokens * PASSOS_TOKEN:PASSOS_TOKEN].tolist()
    return tokens, horas

def dividir(tokens):
    n = len(tokens)
    fim_tr = int(n * FRACAO_TREINO)
    fim_val = int(n * (FRACAO_TREINO + FRACAO_VAL))
    return tokens[:fim_tr], tokens[fim_tr:fim_val], tokens[fim_val:], fim_tr, fim_val

def media_desvio(tokens_tr):
    leituras = tokens_tr.reshape(-1, N_VAR)
    media = leituras.mean(axis=0)
    desvio = leituras.std(axis=0)
    return media, desvio

def normalizar(tokens, media, desvio):
    media_tok = np.tile(media, PASSOS_TOKEN)
    desvio_tok = np.tile(desvio, PASSOS_TOKEN)
    return torch.tensor((tokens - media_tok) / desvio_tok, dtype=torch.float32)

def preparar_dados():
    df = ler_tabela()
    df, info = limpar(df)
    tokens, horas = montar_tokens(df)
    tr, val, te, fim_tr, fim_val = dividir(tokens)
    media, desvio = media_desvio(tr)

    d = {}
    d["info"] = info
    d["estatisticas"] = estatisticas(df)
    d["correlacoes"] = correlacoes(df)
    d["media"] = media
    d["desvio"] = desvio
    d["tr"] = normalizar(tr, media, desvio)
    d["val"] = normalizar(val, media, desvio)
    d["te"] = normalizar(te, media, desvio)
    d["horas_tr"] = horas[:fim_tr]
    d["horas_val"] = horas[fim_tr:fim_val]
    d["horas_te"] = horas[fim_val:]
    d["temperatura"] = df["T (degC)"].values[::PASSOS_TOKEN]
    d["datas"] = df["Date Time"].values[::PASSOS_TOKEN]
    return d

class Janelas(Dataset):
    def __init__(self, tokens, m, h, passo=1):
        self.tokens = tokens
        self.m = m
        self.h = h
        self.inicios = []
        i = 0
        while i + m + h <= len(tokens):
            self.inicios.append(i)
            i += passo

    def __len__(self):
        return len(self.inicios)

    def __getitem__(self, k):
        i = self.inicios[k]
        X = self.tokens[i:i + self.m]
        y = self.tokens[i + self.m:i + self.m + self.h]
        return X, y

def criar_loader(tokens, m, h, lote, embaralhar, passo=1):
    gerador = torch.Generator()
    gerador.manual_seed(SEMENTE)
    return DataLoader(Janelas(tokens, m, h, passo), batch_size=lote, shuffle=embaralhar, generator=gerador)
