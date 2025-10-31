# Detecção de Comunidades Sensível ao Viés via Programação Semidefinida e Heurística

**Autores:** Axl S. Andrade, Nelson Maculan, Ronaldo M. Gregório, Sérgio A. Monteiro, Vitor S. Ponciano

Este repositório fornece a implementação e a estrutura de validação experimental para a metodologia de detecção de comunidades sensível ao viés, conforme proposto no artigo: "*Detecção de Viés Social em Redes Sociais via Programação Semidefinida e Análise Estrutural de Grafos*".

O objetivo principal deste código é particionar grafos de redes sociais balanceando dois objetivos concorrentes:
1.  **Coesão Estrutural (Modularidade):** Comunidades devem ser densamente conectadas internamente.
2.  **Homogeneidade Ideológica (Viés):** Membros de uma mesma comunidade devem compartilhar um viés político/ideológico similar.

Este repositório fornece implementações para a solução exata (SDP) e a aproximação heurística (Enhanced Louvain) descritas no artigo.

## Metodologia

A metodologia é centrada em uma função objetivo unificada, controlada por um hiperparâmetro `alpha` (`α`), que pondera a importância da estrutura (`B`, a matriz de modularidade) versus a do viés (`C`, a matriz de covariância de viés).

`M = (1 - α) * B + α * C`

O problema é então resolvido maximizando `Tr(M*X)`, o que é feito por duas abordagens:
1.  **`BiasAwareSDP`**: Uma relaxação via Programação Semidefinida (SDP) que encontra a solução ótima, mas é computacionalmente cara, sendo viável apenas para grafos pequenos (< 1500 nós).
2.  **`EnhancedLouvainWithBias`**: Uma heurística rápida e escalável, baseada no algoritmo de Louvain, que aproxima a solução ótima em grafos de larga escala.

## Estrutura do Repositório

O código é modularizado para permitir a fácil alternância entre datasets (TwiBot-20 e TwiBot-22) e a clara separação de responsabilidades.

| Arquivo                         | Propósito                                                                                                                                                                |
| :------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `notebooks/Twi-Bot-20-22.ipynb` | **Notebook Principal.** Orquestra todo o pipeline experimental: carregamento, cálculo, execução dos algoritmos e avaliação.                                              |
| `src/config.py`                 | **Arquivo de Configuração Central.** O usuário deve editar este arquivo para selecionar o dataset (`DATASET_MODE`), caminhos e hiperparâmetros (como `ALPHA`).           |
| `src/data_utils.py`             | Contém as classes `TwiBot20Loader` (para o formato JSON único) e `TwiBot22Loader` (para o formato CSV + múltiplos JSONs).                                                |
| `src/bias_calculator.py`        | Contém a lógica para carregar o modelo de IA de viés político (`matous-volf/political-leaning-politics`) e calcular os scores de viés para ambos os formatos de dataset. |
| `src/heuristic.py`              | Implementação da heurística `EnhancedLouvainWithBias`.                                                                                                                   |
| `src/sdp_model.py`              | Implementação do solver exato `BiasAwareSDP` usando `cvxpy`.                                                                                                             |
| `src/evaluation.py`             | Implementação do `ComprehensiveEvaluator` para calcular todas as métricas (Modularidade, Separação de Viés, Pureza, etc.).                                               |

## Pré-requisitos e Instalação

O projeto requer Python 3.10+ e várias bibliotecas científicas. Recomenda-se o uso de um ambiente virtual.

1.  Clone este repositório:
    ```bash
    git clone https://github.com/axlandrade/bias-aware-community-detection
    cd bias-aware-community-detection
    ```

2.  Crie e ative um ambiente virtual:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # (Linux/macOS)
    # ou
    .\.venv\Scripts\activate   # (Windows)
    ```

3.  Instale as dependências. Para ambientes com GPU NVIDIA (recomendado), instale o PyTorch com suporte a CUDA primeiro:
    ```bash
    # Instalar PyTorch (CUDA 12.1 ou superior)
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    
    # Instalar o restante das bibliotecas
    pip install networkx python-louvain pandas tqdm psutil transformers matplotlib seaborn tabulate cvxpy jupyter
    ```
    (Alternativamente, crie um arquivo `requirements.txt` com as bibliotecas acima).

## Aquisição dos Dados (Importante)

Este repositório **não** distribui os datasets TwiBot-20 ou TwiBot-22. Devido aos termos de uso, os pesquisadores devem obter os dados diretamente de suas fontes oficiais e colocá-los na pasta raiz deste projeto.

* **TwiBot-20:** O dataset (contendo `train.json`, `dev.json`, etc.) pode ser solicitado em seu repositório oficial: [BunsenFeng/TwiBot-20](https://github.com/GabrielHam/TwiBot-20)
* **TwiBot-22:** O dataset (contendo `label.csv`, `edge.csv`, pasta `tweet/`, etc.) pode ser solicitado no site oficial: [LuoUndergradXJTU/TwiBot-22](https://twibot22.github.io/)

A estrutura de pastas esperada na raiz do projeto é:
```
/bias-aware-community-detection
    /TwiBot-20
        train.json
        dev.json
        test.json
        support.json
    /TwiBot-22
        /data
            label.csv
            edge.csv
        /tweet
            tweet_0.json
            ...
    /src
    	__init__.py
    	bias_calculator.py
        config.py
        data_utils.py
        evaluation.py
        heuristic.py
        sdp_model.py
    /notebooks
        Twi-Bot-20-22.ipynb
```

## Guia de Execução

A execução do experimento é controlada centralmente pelo `src/config.py`.

### 1. Selecionar o Dataset

Abra `src/config.py` e defina a flag `DATASET_MODE` para o dataset desejado:

```python
# Para rodar o TwiBot-20
DATASET_MODE = "TWIBOT_20"

# Para rodar o TwiBot-22
DATASET_MODE = "TWIBOT_22"
```

### 2. (Se TwiBot-20) Selecionar o Subconjunto

Se estiver usando o TwiBot-20, defina qual arquivo JSON usar (ex: `train.json`, `test.json`) no `src/config.py`:

```python
# Em src/config.py, dentro do bloco if DATASET_MODE == "TWIBOT_20":
DATASET_FILE_PATH = os.path.join(DATA_DIR, "train.json") # ou dev.json, test.json, support.json
```

### 3. (Se TwiBot-22) Definir Limite (Recomendado)

O TwiBot-22 é massivo. Para um teste inicial, é altamente recomendado limitar o número de nós a serem carregados. No notebook `notebooks/Twi-Bot-20-22.ipynb`, na célula "Passo 1", altere `max_nodes`:

```python
# Célula "Passo 1" do notebook
# Recomenda-se 50000 para um teste; None para o dataset completo (requer >= 64GB RAM)
G, bot_labels = data_loader.load_and_build_graph(max_nodes=50000) 
```

### 4. Executar o Notebook

Abra e execute o notebook `notebooks/Twi-Bot-20-22.ipynb` célula por célula. O notebook irá:
1.  Carregar a configuração e o `Loader` correto (TwiBot20Loader ou TwiBot22Loader).
2.  Carregar o grafo e os labels.
3.  Calcular os scores de viés (`BiasCalculator`), baixando o modelo de IA na primeira execução.
4.  Executar a Heurística (`EnhancedLouvainWithBias`) e o Louvain Padrão (Baseline).
5.  Gerar as tabelas de resultados comparativos (`ComprehensiveEvaluator`).
6.  (Opcional) Executar o solver SDP (`BiasAwareSDP`) se o grafo resultante for pequeno o suficiente (`< 1500` nós).

## Licença

Este projeto é licenciado sob os termos da [MIT License](https://github.com/axlandrade/bias-aware-community-detection/blob/main/LICENSE).

## Citação

Se você utilizar este código ou metodologia em sua pesquisa, por favor, cite o artigo original:

[Inseriraqui a citação formal do artigo quando publicado]