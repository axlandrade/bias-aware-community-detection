"""
evaluation.py
----------------------
Evaluation metrics and comparative analysis for the Bias-Aware Community Detection
framework.

This module implements the evaluation layer used to quantitatively assess
community detection results under both structural and ideological criteria.
It provides measures such as Adjusted Rand Index (ARI), Normalized Mutual Information (NMI),
and Modularity, allowing a rigorous comparison between the traditional Louvain method
and its bias-aware extension.

Author: Axl S. Andrade et al.
Affiliation: Universidade Federal Rural do Rio de Janeiro (UFRRJ)
"""

import logging
import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import defaultdict


# --------------------------------------------------------------------------- #
#                          Community Evaluation Class
# --------------------------------------------------------------------------- #
class ComprehensiveEvaluator:
    """
    Evaluation framework for comparing community detection outcomes.

    This class defines modular functions for evaluating both the structural
    coherence (via modularity) and ideological alignment (via ARI and NMI)
    of detected communities.
    """

    # ------------------------------------------------------------------ #
    @staticmethod
    def evaluate_communities(G, partition: dict, bias_scores: dict, ground_truth: dict):
        """
        Compute modularity, ARI, and NMI for a given community assignment.

        Parameters
        ----------
        G : nx.Graph
            Social graph used for community detection.
        partition : dict
            Mapping {node: community_id} produced by the detection algorithm.
        bias_scores : dict
            Mapping {node: bias_score} representing ideological leanings.
        ground_truth : dict
            Mapping {node: label} of real ideological classes (0 = left, 1 = right).

        Returns
        -------
        dict
            Evaluation metrics including:
                - modularity
                - ARI
                - NMI
                - avg_bias_intra (intra-community bias homogeneity)

        Notes
        -----
        - Modularity (Q) quantifies the structural quality of communities.
        - ARI (Adjusted Rand Index) and NMI (Normalized Mutual Information)
          measure similarity between detected and true ideological partitions.
        - avg_bias_intra quantifies ideological cohesion:
              \( \bar{B} = \frac{1}{|C|} \sum_{c \in C} \sigma_c(b) \),
          where \( \sigma_c(b) \) is the standard deviation of bias within community c.
        """
        logging.info("🧩 Evaluating community partition...")

        # --- Defensive Checks --- #
        if not partition or not isinstance(partition, dict):
            raise ValueError("Partition must be a non-empty dictionary {node: community_id}.")

        if len(set(partition.values())) == 1:
            logging.warning("⚠️ Only one community detected; metrics may be degenerate.")

        # --- Modularity --- #
        try:
            comms = ComprehensiveEvaluator._dict_to_communities(partition)
            modularity = nx.algorithms.community.modularity(G, comms)
        except Exception as e:
            logging.warning(f"⚠️ Modularity computation failed: {e}")
            modularity = np.nan

        # --- Bias Homogeneity --- #
        avg_bias_intra = ComprehensiveEvaluator._compute_bias_homogeneity(partition, bias_scores)

        # --- ARI / NMI --- #
        common_nodes = list(set(partition.keys()) & set(ground_truth.keys()))
        if len(common_nodes) < 2:
            logging.warning("⚠️ Insufficient overlap for ARI/NMI computation.")
            ari, nmi = np.nan, np.nan
        else:
            y_true = [ground_truth[u] for u in common_nodes]
            y_pred = [partition[u] for u in common_nodes]
            ari = adjusted_rand_score(y_true, y_pred)
            nmi = normalized_mutual_info_score(y_true, y_pred)

        results = {
            "modularity": modularity,
            "ARI": ari,
            "NMI": nmi,
            "avg_bias_intra": avg_bias_intra,
            "num_communities": len(set(partition.values())),
        }

        logging.info(
            f"✅ Evaluation completed — Q={modularity:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}, intra-bias={avg_bias_intra:.3f}"
        )
        return results

    # ------------------------------------------------------------------ #
    @staticmethod
    def _dict_to_communities(partition: dict):
        """
        Convert a node-community mapping to a list of sets for modularity computation.

        Parameters
        ----------
        partition : dict
            Node-to-community mapping.

        Returns
        -------
        list[set]
            List of node sets, one per community.
        """
        comms = defaultdict(set)
        for node, comm in partition.items():
            comms[comm].add(node)
        return list(comms.values())

    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_bias_homogeneity(partition: dict, bias_scores: dict) -> float:
        """
        Compute the intra-community ideological bias homogeneity metric.

        Parameters
        ----------
        partition : dict
            Mapping {node: community_id}.
        bias_scores : dict
            Mapping {node: bias_score ∈ [-1, +1]}.

        Returns
        -------
        float
            Mean standard deviation of bias values across all communities.

        Interpretation
        --------------
        Lower values indicate greater ideological homogeneity within communities,
        reflecting stronger polarization capture.
        """
        comm_biases = defaultdict(list)
        for node, comm in partition.items():
            if node in bias_scores:
                comm_biases[comm].append(bias_scores[node])

        stdevs = [
            np.std(vals) for vals in comm_biases.values() if len(vals) > 1
        ]
        return np.mean(stdevs) if stdevs else np.nan
