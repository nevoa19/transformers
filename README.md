# Atenção em séries temporais

Mini-projeto 04 (IA): previsão multivariada do conjunto Jena Climate 2009–2016 com um codificador Transformer em PyTorch, com dois objetivos (próximo token e próximas 24 h) e análise dos mapas de atenção.

## Requisitos
- Python 3.10 ou superior
- torch, numpy, pandas, matplotlib

    pip install -r requirements.txt

## Como executar
    python previsao.py

O código procura `jena_climate_2009_2016.csv.zip` na pasta do projeto ou em `data/`; se não achar, baixa de https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip.
Usa GPU (CUDA ou MPS) se tiver; senão, CPU. Em CPU com 2 núcleos o experimento inteiro leva cerca de 40 minutos.
Para pular a busca e usar uma configuração fixa, alterar `RODAR_BUSCA` para `False` e definir `CONFIG_FIXA` no início de `previsao.py`.

## Arquivos
- `previsao.py`: roda o experimento inteiro e gera tabelas e figuras
- `dados.py`: leitura e limpeza do CSV, estatísticas, montagem dos tokens, divisão cronológica, padronização e janelas deslizantes
- `modelos.py`: codificação posicional, atenção multi-cabeça (guarda os pesos), bloco codificador e o modelo completo
- `treino.py`: laço de treino com Adam, perda por época e avaliação
- `metricas.py`: MSE geral, por variável e por hora à frente, MAE nas unidades originais e os baselines
- `mapas.py`: média dos pesos de atenção, peso por defasagem e entropia
- `graficos.py`: figuras
- `tabelas.py`: tabelas em LaTeX

## Protocolo
- limpeza: valores −9999 de `wv` e `max. wv` trocados por 0; datas repetidas removidas
- token: 6 leituras consecutivas de 10 minutos (1 h) das 14 variáveis, ou seja, um vetor de 84 valores
- divisão cronológica dos tokens: 70% treino, 10% validação, 20% teste
- padronização: (x − média) / desvio de cada variável, com média e desvio só do treino
- janela: m tokens de entrada (L = m horas); alvo: o próximo token (h = 1) ou os 24 seguintes (h = 24)
- modelo: projeção linear 84 → n, codificação posicional senoidal, blocos codificadores (atenção multi-cabeça + soma e LayerNorm + FFN 4n com ReLU), dropout 0,1, e uma linear que leva a saída do último token aos 84·h valores previstos
- perda: MSE sobre os valores padronizados; Adam (taxa 0,001), lote 128, norma do gradiente limitada a 1
- busca: referência m = 48, n = 64, 4 cabeças, 2 blocos; cada variação muda um único valor; 4 épocas, 1 a cada 3 janelas de treino, objetivo de 24 h, escolha pela menor perda de validação
- treino final: 15 épocas para cada objetivo, ficando os pesos da época de menor perda de validação
- baselines: persistência (repete o último token) e dia anterior (repete as últimas 24 h)
- mapas de atenção: média dos pesos de 1.000 janelas do teste

## Saídas
Em `resultados/`:
- `tabelas/`: `estatisticas`, `busca`, `teste`, `mae_variaveis`
- `figuras/`: temperatura, correlação, curvas de perda, MSE por hora à frente, exemplos de previsão de 24 h, mapas de atenção, cabeças do último bloco e peso por defasagem
- `resultados.json`: todos os números

## Referências
A. Zhang, Z. C. Lipton, M. Li e A. J. Smola. *Dive into Deep Learning*. d2l.ai.
A. Vaswani et al. Attention is all you need. NeurIPS, 2017.
