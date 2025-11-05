# Bias-Aware Community Detection

[English version here](README_en.md)

### Autores

**Axl S. Andrade**, **Nelson Maculan**, **Ronaldo M. Gregório**, **Sérgio A. Monteiro**, **Vitor S. Ponciano** 

---

## 1. Visão Geral

Este repositório apresenta uma implementação acadêmica da heurística *Bias-Aware Community Detection*, que propõe uma extensão do método de Louvain para incorporação de informações de viés político na estrutura de modularidade. A abordagem busca equilibrar estrutura topológica e alinhamento ideológico, maximizando a função objetivo:

$$(1 - \alpha) Q(C) + \alpha B(C)$$

onde:

- $Q(C)$ representa a modularidade estrutural;
- $B(C)$ quantifica a homogeneidade ideológica dentro das comunidades;
- $\alpha \in [0,1]$ é o parâmetro de ponderação entre estrutura e viés.

O método foi validado experimentalmente sobre o **Indiana University Twitter Dataset** publicado no **Harvard Dataverse**, demonstrando desempenho superior ao Louvain clássico ($\alpha = 0$) na detecção de comunidades ideologicamente coerentes.

---

## 2. Dataset: Indiana University Twitter Dataset

A base experimental deste projeto é o **Conjunto de Dados do Twitter da Universidade de Indiana** publicado no **Harvard Dataverse**.
Este conjunto de dados fornece uma representação em larga escala e empiricamente fundamentada da polarização social e ideológica no Twitter (X).

> **Citação:**  
> Dimitar Nikolov, Alessandro Flammini, and Filippo Menczer (2020).  
> *Replication Data for: Right and left, partisanship predicts vulnerability to misinformation.*  
> Harvard Dataverse, V2.  
> DOI: [10.7910/DVN/6CZHH5](https://doi.org/10.7910/DVN/6CZHH5)

---

### 2.1 Descrição do Dataset

O conjunto de dados consiste em registros anonimizados de usuários do Twitter, suas conexões de rede e comportamento de compartilhamento de conteúdo.
Os seguintes arquivos principais foram utilizados neste projeto:

| File                          | Description                                                                                                                                                                                  |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`anonymized-friends.json`** | Codifica as relações de amizade direcionadas ou não direcionadas entre os usuários, formando o gráfico social \( G = (V, E) \).                                                              |
| **`anonymized-shares.json`**  | Contém informações em nível de domínio sobre URLs compartilhadas ou retuitadas por cada usuário, permitindo a inferência de viés com base no conteúdo.                                       |
| **`measures.tab`**            | Fornece métricas em nível de vértice, incluindo *Partidarismo* e *Desinformação*. A pontuação de *Partidarismo* \((-1,1)\) foi usada exclusivamente para definir o vetor de viés \( b(v) \). |

---

### 2.2 Construção do grafo e modelagem do viés

Após o pré-processamento:

| Descrição                                  | Símbolo     | Valor |
| ------------------------------------------ | ----------- | ----- |
| Número de nós                              | \(          | V     | \) | 15.056    |
| Número de arestas                          | \(          | E     | \) | 2.544.068 |
| Grau médio                                 | \(\bar{k}\) | 337,8 |
| Usuários com rótulo de viés                | \(          | b     | \) | 15.056    |
| Usuários com rótulo de verdade fundamental | \(          | gt    | \) | 12.235    |

O grafo resultante é significativamente mais denso e ideologicamente mais rico do que o FACTOID, fornecendo um substrato mais realista para experimentos de detecção de comunidades.

---

## 3. Estrutura do Projeto

```
bias-aware-community-detection/
│
├── src/
│   ├── heuristic.py              # Bias-Aware Louvain heuristic implementation
│   ├── twitter_dataset.py        # Parsing and preprocessing of the Indiana Twitter dataset
│   ├── bias_calculator.py        # Computation of bias scores from measures.tab
│   ├── evaluation.py             # Evaluation of partitions (ARI, NMI, modularity)
│   ├── sdp_model.py              # Optional semidefinite relaxation (for theoretical comparison)
│   ├── data_utils.py             # Helper utilities for graph construction
│   ├── config.py                 # Global configuration and paths
│   └── __init__.py
│
├── TWITTER/
│   ├── anonymized-friends.json
│   ├── anonymized-shares.json
│   └── measures.tab
│
├── Heuristic_Validation.ipynb    # Main validation notebook
├── LICENSE
├── README_pt.md                  # Portuguese version of this document
└── README.md
```

### 3.1 Licenciamento e Atribuição

O dataset do Twitter da Indiana University é de acesso público sob os termos de uso do Harvard Dataverse.  
Todas as etapas de processamento e análise respeitam integralmente as diretrizes de anonimização e uso ético estabelecidas pelos autores originais.

---

## 4. Execução do Pipeline

O experimento pode ser reproduzido diretamente pelo notebook `Heuristic_Validation.ipynb`.

**Etapas:**

1. Instalar dependências.  
2. Colocar os arquivos do Dataset nas pastas indicadas.  
3. Executar o notebook completo.

O pipeline realiza:

- Construção do grafo social $G$;  
- Cálculo de $b(v)$ (bias_score) e das etiquetas de verdade de terreno;  
- Execução dos métodos:  
  - Louvain padrão ($\alpha = 0.0$)  
  - Heurística ciente de viés ($\alpha = 0.5$)  
- Avaliação comparativa via ARI e NMI.

> **Nota:** A execução completa do pipeline é recomendada no **Google Colab**, devido ao suporte nativo a GPUs (T4/V100), maior disponibilidade de memória RAM e à configuração automática do ambiente Python. Isso garante reprodutibilidade, acelera o cálculo de embeddings e simplifica o gerenciamento das dependências do projeto.

---

## 5. Figura 1 — Fluxo Experimental FACTOID

```text
┌──────────────────────────────┐
│        Twitter Dataset       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Cálculo de bias_score b(v)  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Enhanced Louvain (α ∈ [0,1]) │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Comparação com Louvain padrão│
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Avaliação (ARI / NMI)    │
└──────────────────────────────┘
```

*Figura 1 — Fluxo de processamento experimental do método proposto.*

---

## 6. Resultados Experimentais

| Método                        | α   | ARI   | NMI   |
| ----------------------------- | --- | ----- | ----- |
| Louvain (baseline)            | 0.0 | 0.898 | 0.827 |
| Bias-Aware Louvain (proposto) | 0.8 | 0.900 | 0.830 |

A heurística proposta alcançou melhor alinhamento com o *ground-truth* político, demonstrando que a incorporação do termo de viés $B(C)$ aprimora a separação ideológica sem comprometer a coesão estrutural.

**Adjusted Rand Index (ARI)** e **Normalized Mutual Information (NMI)** são métricas clássicas de avaliação de *clustering supervisionado* utilizadas para quantificar a similaridade entre partições obtidas ($C_{pred}$) e partições de referência ($C_{GT}$).  

- O **ARI** mede a concordância corrigida pelo acaso, variando de $-1$ (discordância total) a $1$ (correspondência perfeita).  
  $$ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}$$  
  onde $RI$ é o *Rand Index* bruto e $E[RI]$ é seu valor esperado sob uma distribuição aleatória.  

- O **NMI**, por sua vez, avalia a quantidade de informação mútua entre as partições, normalizada pelo conteúdo informativo médio:  
  $$NMI(C_{pred}, C_{GT}) = \frac{2 \, I(C_{pred}; C_{GT})}{H(C_{pred}) + H(C_{GT})}$$  
  com $I$ representando a informação mútua e $H$ as entropias.  

Valores mais altos de ambas as métricas indicam maior similaridade entre as partições — logo, melhor capacidade do algoritmo em identificar comunidades coerentes com a estrutura ideológica de referência.

---

## 7. Citação

Se este repositório for utilizado em publicações acadêmicas, favor citar como:

```bibtex
@misc{monteiro,
  author = {Axl S. Andrade and Nelson Maculan and Ronaldo M. Gregório and Sérgio A. Monteiro and Vitor S. Ponciano},
  title  = {Detecção de Viés Social em Redes Sociais via Programação Semidefinida e Análise Estrutural de Grafos: Implementação e Validação Experimental},
  year   = {2025},
  url    = {https://github.com/axlandrade/bias-aware-community-detection},
  note   = {Heuristic method for bias-aware community detection validated on the FACTOID dataset.}
}
```
