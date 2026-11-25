from dados import N_VAR, PASSOS_TOKEN

def separar_leituras(t):
    return t.reshape(t.shape[0], t.shape[1] * PASSOS_TOKEN, N_VAR)

def baseline_persistencia(entradas, h):
    ultimo = entradas[:, -1:, :]
    return ultimo.repeat(1, h, 1)

def baseline_dia_anterior(entradas, h):
    return entradas[:, -24:, :][:, :h, :]

def mse(prev, real):
    return ((prev - real) ** 2).mean().item()

def mse_por_variavel(prev, real):
    p = separar_leituras(prev)
    r = separar_leituras(real)
    lista = []
    j = 0
    while j < N_VAR:
        lista.append(((p[:, :, j] - r[:, :, j]) ** 2).mean().item())
        j += 1
    return lista

def mae_fisico(prev, real, desvio):
    p = separar_leituras(prev)
    r = separar_leituras(real)
    lista = []
    j = 0
    while j < N_VAR:
        erro = (p[:, :, j] - r[:, :, j]).abs().mean().item()
        lista.append(erro * float(desvio[j]))
        j += 1
    return lista

def mse_por_horizonte(prev, real):
    lista = []
    k = 0
    while k < prev.shape[1]:
        lista.append(((prev[:, k, :] - real[:, k, :]) ** 2).mean().item())
        k += 1
    return lista

def calcular_metricas(prev, real, desvio):
    m = {}
    m["mse"] = mse(prev, real)
    m["mse_variavel"] = mse_por_variavel(prev, real)
    m["mae_fisico"] = mae_fisico(prev, real, desvio)
    m["mse_horizonte"] = mse_por_horizonte(prev, real)
    return m
