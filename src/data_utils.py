"""
data_utils.py
----------------------
Auxiliary utilities for data preprocessing, alignment, and normalization
in the Bias-Aware Community Detection framework.

This module provides low-level tools for managing user attributes and graph
metadata. It focuses on data integrity operations, ensuring all experimental
inputs — graphs, bias vectors, and ground-truth — remain consistent across
different datasets (FACTOID, Twitter, etc.).

Author: Axl S. Andrade et al.
Affiliation: Universidade Federal Rural do Rio de Janeiro (UFRRJ)
"""

import logging
import numpy as np
import pandas as pd
import networkx as nx


# --------------------------------------------------------------------------- #
#                            Normalization Utilities
# --------------------------------------------------------------------------- #
def normalize_bias_values(bias_dict: dict) -> dict:
    """
    Normalize bias values into the range [-1, +1].

    Parameters
    ----------
    bias_dict : dict
        Mapping {user_id: bias_value}, where values may vary in scale
        depending on dataset provenance.

    Returns
    -------
    dict
        Normalized bias mapping {user_id: normalized_bias ∈ [-1, +1]}.

    Notes
    -----
    - Bias normalization ensures comparability between datasets.
    - It follows a linear rescaling procedure:

        \\[
        b' = 2 \\cdot \\frac{b - b_{min}}{b_{max} - b_{min}} - 1
        \\]

    - Values outside the expected range are clipped.
    """
    if not bias_dict:
        logging.warning("⚠️ Empty bias dictionary received.")
        return {}

    values = np.array(list(bias_dict.values()), dtype=float)
    b_min, b_max = np.min(values), np.max(values)

    if b_max == b_min:
        logging.warning("⚠️ Constant bias vector detected; returning zeros.")
        return {k: 0.0 for k in bias_dict}

    scaled = {k: np.clip(2 * ((v - b_min) / (b_max - b_min)) - 1, -1, 1)
              for k, v in bias_dict.items()}
    return scaled


# --------------------------------------------------------------------------- #
#                          Graph and Data Alignment
# --------------------------------------------------------------------------- #
def align_graph_and_attributes(G: nx.Graph, bias_dict: dict, gt_dict: dict):
    """
    Align nodes, bias scores, and ground-truth vectors across datasets.

    Parameters
    ----------
    G : nx.Graph
        Input social graph.
    bias_dict : dict
        Mapping {node: bias_score}.
    gt_dict : dict
        Mapping {node: ground_truth_label}.

    Returns
    -------
    tuple (G_aligned, bias_aligned, gt_aligned)
        - G_aligned : induced subgraph with nodes present in both bias_dict and gt_dict.
        - bias_aligned : bias scores restricted to the aligned nodes.
        - gt_aligned : ground-truth restricted to the aligned nodes.

    Notes
    -----
    - This ensures that every node in the analysis has both structural and ideological data.
    - Missing nodes are automatically filtered to maintain data consistency.
    """
    logging.info("🧩 Aligning graph with bias and ground-truth data...")

    common_nodes = set(G.nodes()) & set(bias_dict.keys())
    if gt_dict:
        common_nodes &= set(gt_dict.keys())

    G_aligned = G.subgraph(common_nodes).copy()
    bias_aligned = {n: bias_dict[n] for n in common_nodes}
    gt_aligned = {n: gt_dict[n] for n in common_nodes} if gt_dict else {}

    logging.info(
        f"✅ Alignment complete: |V|={len(G_aligned):,} |E|={G_aligned.number_of_edges():,} "
        f"|b|={len(bias_aligned):,} |gt|={len(gt_aligned):,}"
    )

    return G_aligned, bias_aligned, gt_aligned


# --------------------------------------------------------------------------- #
#                             Statistical Summary
# --------------------------------------------------------------------------- #
def summarize_graph(G: nx.Graph, bias_dict: dict = None) -> pd.DataFrame:
    """
    Compute descriptive statistics for a given social graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    bias_dict : dict, optional
        Bias mapping to compute ideological summary statistics.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame containing:
            - num_nodes
            - num_edges
            - avg_degree
            - density
            - avg_bias (if provided)
            - std_bias (if provided)

    Notes
    -----
    - This summary is primarily used for dataset documentation tables
      (as shown in the FACTOID and Twitter experimental sections).
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = np.mean([deg for _, deg in G.degree()])
    density = nx.density(G)

    stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "density": density,
    }

    if bias_dict:
        biases = np.array(list(bias_dict.values()))
        stats.update({
            "avg_bias": float(np.mean(biases)),
            "std_bias": float(np.std(biases)),
        })

    df = pd.DataFrame([stats])
    logging.info(f"📊 Graph summary: {df.to_dict(orient='records')[0]}")
    return df
