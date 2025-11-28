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

The method was experimentally validated on the **Indiana University Twitter Dataset** published on **Harvard Dataverse**, demonstrating superior performance to the classical Louvain algorithm ($\alpha = 0$) in detecting ideologically coherent communities.

---

## 2. Dataset: Indiana University Twitter Dataset

The experimental foundation of this project is the dataset to the **Indiana University Twitter Dataset** published on **Harvard Dataverse**.  
This dataset provides a large-scale, empirically grounded representation of social and ideological polarization on Twitter (X).

> **Citation:**  
> Dimitar Nikolov, Alessandro Flammini, and Filippo Menczer (2020).  
> *Replication Data for: Right and left, partisanship predicts vulnerability to misinformation.*  
> Harvard Dataverse, V2.  
> DOI: [10.7910/DVN/6CZHH5](https://doi.org/10.7910/DVN/6CZHH5)

---

### 2.1 Dataset Description

The dataset consists of anonymized records of Twitter users, their network connections, and content-sharing behavior.  
The following core files were employed in this project:

| File                          | Description                                                                                                                                                                |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`anonymized-friends.json`** | Encodes the directed or undirected friendship relations between users, forming the social graph \( G = (V, E) \).                                                          |
| **`anonymized-shares.json`**  | Contains domain-level information on URLs shared or retweeted by each user, allowing for content-based bias inference.                                                     |
| **`measures.tab`**            | Provides node-level metrics, including *Partisanship* and *Misinformation*. The *Partisanship* score \((-1,1)\) was used exclusively to define the bias vector \( b(v) \). |

---

### 2.2 Graph Construction and Bias Modeling

After preprocessing (with limited nodes):

| Description                   | Symbol | Value     |
| ----------------------------- | ------ | --------- |
| Number of nodes               | \(V\)  | 15,056    |
| Number of edges               | \(E\)  | 2,544,068 |
| Average degree                | \(k\)  | 337.8     |
| Users with bias label         | \(b\)  | 15,056    |
| Users with ground-truth label | \(gt\) | 12,235    |

The resulting graph is significantly denser and ideologically richer than FACTOID, providing a more realistic substrate for community detection experiments.

---

## 3. Project Structure

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

### 3.1 Licensing and Attribution

The Indiana Twitter dataset is publicly available under the Harvard Dataverse terms of use.  
All data handling procedures in this project strictly adhere to the anonymization and ethical usage policies defined by the dataset authors.

---

## 4. Pipeline Execution

The experiment can be fully reproduced using the notebook `Heuristic_Validation.ipynb`.

**Steps:**

1. Install dependencies.  
2. Place the Twitter Dataset files in the specified directories.  
3. Execute the notebook end-to-end.

The pipeline performs:

- Construction of the social graph $G$;  
- Computation of $b(v)$ (bias_score) and ground-truth labels;  
- Execution of the following methods:  
  - Standard Louvain ($\alpha = 0.0$)  
  - Bias-Aware Heuristic ($\alpha = [0,1]$)  
- Comparative evaluation via ARI and NMI metrics.

> **Note:** Full pipeline execution is recommended on **Google Colab**, due to native GPU support (T4/V100) and automatic Python environment setup. This ensures reproducibility, accelerates computation, and simplifies dependency management.

---

## 5. Figure 1 — FACTOID Experimental Workflow

```text
┌──────────────────────────────┐
│        Twitter Dataset       │
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

| Method                        | α   | ARI   | NMI   |
| ----------------------------- | --- | ----- | ----- |
| Louvain (baseline)            | 0.0 | 0.898 | 0.827 |
| Bias-Aware Louvain (proposed) | 0.8 | 0.900 | 0.830 |

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

## 7. Citation

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
