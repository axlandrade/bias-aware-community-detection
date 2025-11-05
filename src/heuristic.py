"""
heuristic.py
----------------------
Enhanced Louvain Heuristic for Bias-Aware Community Detection.

Implements a two-phase optimization procedure that extends the classical Louvain
algorithm by incorporating ideological bias alignment into the modularity function.

The method aims to maximize the composite objective function:

    (1 - α) * Q(C) + α * B(C)

where:
    - Q(C): structural modularity of the community partition,
    - B(C): ideological cohesion (bias alignment) among nodes,
    - α ∈ [0, 1]: weighting factor balancing structure and ideology.

This implementation follows the methodology described in:
"Andrade et al. (2025). Bias-Aware Community Detection via Modularity–Bias Optimization."

Author: Axl S. Andrade et al.
Affiliation: Universidade Federal Rural do Rio de Janeiro (UFRRJ)
"""

import logging
import random
import time
import numpy as np
import networkx as nx
from collections import defaultdict
import community.community_louvain as louvain


# --------------------------------------------------------------------------- #
#                            Enhanced Louvain Class
# --------------------------------------------------------------------------- #
class EnhancedLouvainWithBias:
    """
    Bias-Aware extension of the Louvain algorithm.

    The algorithm operates in three main phases:

    (1) **Initialization** — computes a structural Louvain partition.
    (2) **Iterative Refinement** — iteratively reassigns nodes to communities
        maximizing a combined gain of structure and ideological bias.
    (3) **Balancing Phase** — optionally merges minor communities to ensure a
        fixed number of K communities.

    Attributes
    ----------
    alpha : float
        Trade-off parameter ∈ [0, 1] between structure (Q) and ideology (B).
    max_iterations_refine : int
        Maximum number of refinement iterations.
    verbose : bool
        Whether to display progress and intermediate results.
    communities : dict
        Final node-to-community assignment.
    execution_time : float
        Total runtime in seconds.
    """

    def __init__(self, alpha: float = 0.5, max_iterations: int = 100, verbose: bool = True):
        self.alpha = alpha
        self.max_iterations_refine = max_iterations
        self.verbose = verbose
        self.communities = {}
        self.execution_time = 0.0

    # ------------------------------------------------------------------ #
    def fit(self, G: nx.Graph, bias_scores: dict, num_communities: int = 2):
        """
        Execute the full Bias-Aware Louvain optimization procedure.

        Parameters
        ----------
        G : nx.Graph
            Undirected social graph.
        bias_scores : dict
            Mapping {node: bias_score} with b(v) ∈ [-1, +1].
        num_communities : int, optional
            Target number of communities after balancing (default=2).

        Notes
        -----
        The fitting process follows the theoretical structure:

        **Phase 1:**
            Standard Louvain partition to obtain C₀.

        **Phase 2:**
            Refinement according to gain function:
                Δ(v, c) = (1 - α) ΔQ(v, c) + α ΔB(v, c)
            where ΔQ is structural gain and ΔB is bias alignment gain.

        **Phase 3:**
            Merge minor communities to reach the desired K structure.
        """
        start_time = time.time()
        if self.verbose:
            logging.info(f"🎯 Starting Enhanced Louvain (α={self.alpha:.2f})")

        # --- Phase 1: Structural Louvain --- #
        if self.verbose:
            logging.info("   Phase 1: Running baseline Louvain partition...")
        partition = louvain.best_partition(G)

        # --- Phase 2: Iterative Refinement --- #
        if self.verbose:
            logging.info(f"   Phase 2: Refining communities (max_iter={self.max_iterations_refine})...")
        partition, total_moves = self._iterative_refinement(G, partition, bias_scores)

        # --- Phase 3: Balancing --- #
        if self.verbose:
            logging.info(f"   Phase 3: Balancing to {num_communities} communities...")
        final_partition = self._balance_communities(partition, num_communities)

        # Store results
        self.communities = final_partition
        self.execution_time = time.time() - start_time

        if self.verbose:
            logging.info(
                f"✅ Completed in {self.execution_time:.2f}s ({total_moves} node moves during refinement)"
            )
            self._print_community_stats(final_partition, bias_scores)

    # ------------------------------------------------------------------ #
    def _iterative_refinement(self, G, partition, bias_scores):
        """
        Iteratively optimize node assignments using the composite gain function.

        The refinement process reassigns each node v to a community c maximizing
        the total gain:

            Δ(v, c) = (1 - α) * ΔQ(v, c) + α * ΔB(v, c)

        where:

            ΔQ(v, c) = Δ(number of edges from v to members of c)
            ΔB(v, c) = reduction in |b(v) - mean(bias(c))|

        Parameters
        ----------
        G : nx.Graph
            Input graph.
        partition : dict
            Initial partition from Louvain.
        bias_scores : dict
            Node bias values.

        Returns
        -------
        tuple (dict, int)
            Updated partition and total number of node moves performed.
        """
        total_moves = 0
        nodes = list(G.nodes())

        for i in range(self.max_iterations_refine):
            moves_in_iter = 0
            random.shuffle(nodes)

            comm_avg_bias = self._get_community_avg_bias(partition, bias_scores)

            for node in nodes:
                current_comm = partition[node]
                node_bias = bias_scores.get(node, 0.0)

                # Count edges to each neighboring community
                neighbor_comms = defaultdict(int)
                for neighbor in G[node]:
                    neighbor_comms[partition[neighbor]] += 1

                # Current structural and ideological state
                current_links = neighbor_comms.get(current_comm, 0)
                current_bias_diff = abs(node_bias - comm_avg_bias.get(current_comm, 0))

                best_gain = -np.inf
                best_comm = current_comm

                for comm, links in neighbor_comms.items():
                    if comm == current_comm:
                        continue

                    ΔQ = links - current_links
                    ΔB = (abs(node_bias - comm_avg_bias.get(current_comm, 0))
                          - abs(node_bias - comm_avg_bias.get(comm, 0)))

                    gain = (1 - self.alpha) * ΔQ + self.alpha * ΔB

                    if gain > best_gain:
                        best_gain = gain
                        best_comm = comm

                # Reassign node if beneficial
                if best_comm != current_comm and best_gain > 0:
                    partition[node] = best_comm
                    moves_in_iter += 1
                    comm_avg_bias = self._get_community_avg_bias(partition, bias_scores)

            total_moves += moves_in_iter
            if self.verbose:
                logging.info(f"      Iter {i + 1}/{self.max_iterations_refine}: {moves_in_iter} moves")

            if moves_in_iter == 0:
                if self.verbose:
                    logging.info("      Convergence achieved.")
                break

        return partition, total_moves

    # ------------------------------------------------------------------ #
    def _get_community_avg_bias(self, partition, bias_scores):
        """
        Compute average ideological bias per community.

        Parameters
        ----------
        partition : dict
            Node-community mapping.
        bias_scores : dict
            Node bias values.

        Returns
        -------
        dict
            Mapping {community: avg_bias}.
        """
        comm_sum, comm_count = defaultdict(float), defaultdict(int)
        for node, comm in partition.items():
            comm_sum[comm] += bias_scores.get(node, 0.0)
            comm_count[comm] += 1

        return {c: comm_sum[c] / comm_count[c] for c in comm_sum if comm_count[c] > 0}

    # ------------------------------------------------------------------ #
    def _balance_communities(self, partition, num_communities):
        """
        Merge smaller communities to achieve the target number of groups (K).

        Parameters
        ----------
        partition : dict
            Node-to-community mapping.
        num_communities : int
            Desired number of final communities.

        Returns
        -------
        dict
            Balanced node-to-community mapping.
        """
        if num_communities is None:
            return partition

        unique = list(set(partition.values()))
        if len(unique) <= num_communities:
            return partition

        sizes = defaultdict(int)
        for comm in partition.values():
            sizes[comm] += 1

        top_comms = sorted(sizes, key=sizes.get, reverse=True)[:num_communities]
        top_set = set(top_comms)

        comm_map = {c: (c if c in top_set else top_comms[0]) for c in unique}
        return {n: comm_map[c] for n, c in partition.items()}

    # ------------------------------------------------------------------ #
    def _print_community_stats(self, partition, bias_scores):
        """
        Print detailed summary of community composition and bias averages.
        """
        logging.info("📊 Final community statistics:")
        comms = defaultdict(list)
        for node, comm in partition.items():
            comms[comm].append(bias_scores.get(node, 0.0))

        for comm, biases in sorted(comms.items()):
            if len(biases) > 0:
                avg = np.mean(biases)
                std = np.std(biases)
                logging.info(
                    f"  Community {comm:>3}: {len(biases):>6,} nodes | mean bias = {avg:+.3f} ± {std:.3f}"
                )
