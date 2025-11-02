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

O método foi validado experimentalmente sobre o **FACTOID Dataset**, demonstrando desempenho superior ao Louvain clássico ($\alpha = 0$) na detecção de comunidades ideologicamente coerentes.

---

## 2. Dataset: FACTOID

O **FACTOID Dataset** ([CAISA Lab, University of Amsterdam](https://github.com/caisa-lab/FACTOID-dataset)) constitui o núcleo experimental do projeto.  
Ele contém:

- Um grafo de interações entre usuários do Reddit (submissões e respostas);  
- Corpus textual com anotações factuais e ideológicas;  
- Etiquetas de viés político baseadas em domínios verificados (left/right/center).

O FACTOID foi selecionado por conter **ground-truth explícito de polarização política**, viabilizando a comparação quantitativa entre partições obtidas e partições reais.

> **Atribuição:** Este projeto utiliza dados derivados do repositório público FACTOID, mantido por CAISA Lab, University of Amsterdam.

---

## 3. Estrutura do Projeto

```
bias-aware-community-detection/
│
├── src/
│   ├── heuristic.py              # Implementação da heurística Bias-Aware Louvain
│   ├── reddit_user_dataset.py    # Manipulação e cache do FACTOID
│   ├── fake_news_detection.py    # Leitura de domínios com viés político
│   ├── bias_calculator.py        # Cálculo de bias_score b(v)
│   ├── evaluation.py             # Avaliação com ARI e NMI
│   ├── sdp_model.py              # Formulação semidefinida opcional
│   ├── data_utils.py             # Funções utilitárias
│   ├── config.py                 # Configurações globais
│   └── __init__.py
│
├── FACTOID/
│   ├── reddit_corpus_unbalanced_filtered.gzip
│   ├── social_graph_data/
│   └── fn_domains_verified
│
├── processed_factoid/
│   ├── social_graph.gml
│   ├── factoid_bias_groundtruth.csv
│   └── factoid_alpha_sweep.csv
│
├── validar_heuristica.ipynb      # Notebook principal de validação
└── README.md
```

---

## 4. Execução do Pipeline

O experimento pode ser reproduzido diretamente pelo notebook `validar_heuristica.ipynb`.

**Etapas:**

1. Instalar dependências.  
2. Colocar os arquivos FACTOID nas pastas indicadas.  
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
│        FACTOID Dataset       │
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

| Método                        | α    | ARI   | NMI   |
| ----------------------------- | ---- | ----- | ----- |
| Louvain (baseline)            | 0.0  | 0.054 | 0.098 |
| Bias-Aware Louvain (proposto) | 0.5  | 0.452 | 0.256 |

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

## 7. Referências

1. Glenski, M., et al. (2023). *FACTOID: Fact-Checking and Ideology Dataset for Reddit.* CAISA Lab, University of Amsterdam.  
   Disponível em: [https://github.com/caisa-lab/FACTOID-dataset](https://github.com/caisa-lab/FACTOID-dataset)

2. Andrade, A. S., Maculan, N., Gregório, R. M., Monteiro, S. A., & Ponciano, V. S. (2025). *Bias-Aware Community Detection via Modularity–Bias Optimization.* Manuscrito em preparação.

---

## 8. Citação

Se este repositório for utilizado em publicações acadêmicas, favor citar como:

```bibtex
@misc{andrade2025biasaware,
  author = {Axl S. Andrade and Nelson Maculan and Ronaldo M. Gregório and Sérgio A. Monteiro and Vitor S. Ponciano},
  title  = {Detecção de Viés Social em Redes Sociais via Programação Semidefinida e Análise Estrutural de Grafos: Implementação e Validação Experimental},
  year   = {2025},
  url    = {https://github.com/axlandrade/bias-aware-community-detection},
  note   = {Heuristic method for bias-aware community detection validated on the FACTOID dataset.}
}
```