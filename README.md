# Bias-Aware Community Detection

[Portuguese version here](README_pt.md)


### Authors

**Axl S. Andrade**, **Nelson Maculan**, **Ronaldo M. Gregório**, **Sérgio A. Monteiro**, **Vitor S. Ponciano** 

---

## 1. Overview

This repository presents an academic implementation of the *Bias-Aware Community Detection* heuristic, which extends the Louvain method to incorporate political bias information into the modularity structure. The approach aims to balance topological structure and ideological alignment by maximizing the following objective function:

$$(1 - \alpha) Q(C) + \alpha B(C)$$

where:

- $Q(C)$ represents the structural modularity;
- $B(C)$ quantifies the ideological homogeneity within communities;
- $\alpha \in [0,1]$ is the weighting parameter between structure and bias.

The method was experimentally validated on the **FACTOID Dataset**, demonstrating superior performance to the classical Louvain algorithm ($\alpha = 0$) in detecting ideologically coherent communities.

---

## 2. Dataset: FACTOID

The **FACTOID Dataset** ([CAISA Lab, University of Amsterdam](https://github.com/caisa-lab/FACTOID-dataset)) constitutes the experimental core of this project.  
It includes:

- A graph of Reddit user interactions (submissions and replies);  
- Text corpus with factual and ideological annotations;  
- Political bias labels based on verified media domains (left/right/center).

FACTOID was chosen because it contains an **explicit ground-truth of political polarization**, enabling quantitative comparison between predicted and reference partitions.

> **Attribution:** This project uses data derived from the public FACTOID repository maintained by CAISA Lab, University of Amsterdam.

---

## 3. Project Structure

```
bias-aware-community-detection/
│
├── src/
│   ├── heuristic.py              # Bias-Aware Louvain heuristic implementation
│   ├── reddit_user_dataset.py    # FACTOID handling and caching
│   ├── fake_news_detection.py    # Reading of politically biased domains
│   ├── bias_calculator.py        # Computation of bias_score b(v)
│   ├── evaluation.py             # Evaluation with ARI and NMI
│   ├── sdp_model.py              # Optional semidefinite formulation
│   ├── data_utils.py             # Utility functions
│   ├── config.py                 # Global configurations
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
├── validar_heuristica.ipynb      # Main validation notebook
└── README.md
```

---

## 4. Pipeline Execution

The experiment can be fully reproduced using the notebook `validar_heuristica.ipynb`.

**Steps:**

1. Install dependencies.  
2. Place the FACTOID files in the specified directories.  
3. Execute the notebook end-to-end.

The pipeline performs:

- Construction of the social graph $G$;  
- Computation of $b(v)$ (bias_score) and ground-truth labels;  
- Execution of the following methods:  
  - Standard Louvain ($\alpha = 0.0$)  
  - Bias-Aware Heuristic ($\alpha = 0.5$)  
- Comparative evaluation via ARI and NMI metrics.

> **Note:** Full pipeline execution is recommended on **Google Colab**, due to native GPU support (T4/V100) and automatic Python environment setup. This ensures reproducibility, accelerates computation, and simplifies dependency management.

---

## 5. Figure 1 — FACTOID Experimental Workflow

```text
┌──────────────────────────────┐
│        FACTOID Dataset       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Computation of bias_score   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Enhanced Louvain (α ∈ [0,1]) │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Comparison with baseline     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Evaluation (ARI / NMI)   │
└──────────────────────────────┘
```

*Figure 1 — Experimental workflow of the proposed method.*

---

## 6. Experimental Results

| Method                        | α    | ARI   | NMI   |
| ----------------------------- | ---- | ----- | ----- |
| Louvain (baseline)            | 0.0  | 0.054 | 0.098 |
| Bias-Aware Louvain (proposed) | 0.5  | 0.452 | 0.256 |

The proposed heuristic achieved better alignment with the political ground-truth, showing that incorporating the bias term $B(C)$ improves ideological separation without compromising structural cohesion.

**Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)** are standard metrics for supervised clustering evaluation, used to quantify the similarity between predicted ($C_{pred}$) and ground-truth ($C_{GT}$) partitions.  

- **ARI** measures chance-corrected agreement, ranging from $-1$ (complete disagreement) to $1$ (perfect match):  
  $$ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}$$  
  where $RI$ is the raw Rand Index and $E[RI]$ its expected value under random partitioning.  

- **NMI** evaluates the mutual information between partitions, normalized by their average informational content:  
  $$NMI(C_{pred}, C_{GT}) = \frac{2 \, I(C_{pred}; C_{GT})}{H(C_{pred}) + H(C_{GT})}$$  
  where $I$ is the mutual information and $H$ represents entropy.  

Higher ARI and NMI values indicate stronger agreement between partitions, thus greater ability of the algorithm to detect ideologically consistent communities.

---

## 7. References

1. Glenski, M., et al. (2023). *FACTOID: Fact-Checking and Ideology Dataset for Reddit.* CAISA Lab, University of Amsterdam.  
   Available at: [https://github.com/caisa-lab/FACTOID-dataset](https://github.com/caisa-lab/FACTOID-dataset)

2. Andrade, A. S., Maculan, N., Gregório, R. M., Monteiro, S. A., & Ponciano, V. S. (2025). *Bias-Aware Community Detection via Modularity–Bias Optimization.* Manuscript in preparation.

---

## 8. Citation

If this repository is used in academic publications, please cite as:

```bibtex
@misc{monteiro,
  author = {Axl S. Andrade and Nelson Maculan and Ronaldo M. Gregório and Sérgio A. Monteiro and Vitor S. Ponciano},
  title  = {Bias-Aware Community Detection via Semidefinite Programming and Structural Graph Analysis: Implementation and Experimental Validation},
  year   = {2025},
  url    = {https://github.com/axlandrade/bias-aware-community-detection},
  note   = {Heuristic method for bias-aware community detection validated on the FACTOID dataset.}
}
```
