import torch

def coletar_pesos(modelo, loader, dispositivo, n_amostras):
    modelo.eval()
    somas = None
    vistas = 0
    with torch.no_grad():
        for X, y in loader:
            if vistas >= n_amostras:
                break
            falta = n_amostras - vistas
            X = X[:falta].to(dispositivo)
            modelo(X)
            pesos = modelo.pesos_atencao()
            if somas is None:
                somas = []
                for p in pesos:
                    somas.append(torch.zeros(p.shape[1], p.shape[2], p.shape[3]))
            b = 0
            while b < len(pesos):
                somas[b] += pesos[b].sum(dim=0).cpu()
                b += 1
            vistas += len(X)

    medias = []
    for s in somas:
        medias.append(s / vistas)
    return medias, vistas

def peso_por_defasagem(matriz):
    m = matriz.shape[0]
    ultima = matriz[m - 1]
    lista = []
    defasagem = 0
    while defasagem < m:
        lista.append(ultima[m - 1 - defasagem].item())
        defasagem += 1
    return lista

def entropia_media(matriz):
    m = matriz.shape[0]
    total = 0.0
    i = 0
    while i < m:
        linha = matriz[i]
        total += -(linha * torch.log(linha + 1e-12)).sum().item()
        i += 1
    return total / m
