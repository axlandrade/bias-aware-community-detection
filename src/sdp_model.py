# sdp_model.py
# -*- coding: utf-8 -*-
"""
Semidefinite Programming (SDP) for Bias-Aware Community Detection
-----------------------------------------------------------------

This module implements a 2-way community detection via a Max-Cut style SDP
relaxation with a bias-aware term. It supports:

- Building a combined weight matrix W = (1-α)*W_modularity + α*W_bias
- Solving the relaxed SDP:   minimize <W, X>  s.t. X ⪰ 0, diag(X)=1
- Random hyperplane rounding (Goemans–Williamson) to get {+1,-1} labels
- Optional recursive bisection to get k>2 communities

References (for formulation ideas):
- Goemans, Williamson (1995): Approximation Algorithms for Max-Cut and
  Satisfiability Problems Using Semidefinite Programming.
- Modularity-to-cut formulations are standard; here we use a weighted
  Max-Cut view to combine topology + bias terms.

Author: Axl S. Andrade et al.
"""

from __future__ import annotations
import math
import time
import logging
import numpy as np
import networkx as nx

try:
    import cvxpy as cp
except Exception as e:
    raise ImportError(
        "cvxpy is required for the SDP solver. Try: pip install cvxpy"
    ) from e

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("BiasAwareSDP")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _modularity_matrix(G: nx.Graph, gamma: float = 1.0) -> np.ndarray:
    """
    Compute the modularity matrix B for an undirected graph G:
        B_ij = A_ij - gamma * (k_i * k_j) / (2m)
    where m = |E|, k_i is degree of node i.

    Returns:
        B (np.ndarray) with node order matching the order provided externally.
        NOTE: you must pass node ordering and build B accordingly.
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}

    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    m = deg.sum() / 2.0
    if m <= 0:
        raise ValueError("Graph has no edges; modularity is undefined.")

    # Outer-product term
    P = gamma * np.outer(deg, deg) / (2.0 * m)
    B = A.toarray() - P
    # Force symmetry
    B = 0.5 * (B + B.T)
    return B, nodes


def _bias_weight_matrix(G: nx.Graph, nodes: list, b: dict[int, float], normalize=True) -> np.ndarray:
    """
    Construct a bias-based weight matrix to *encourage cutting* edges
    connecting nodes with dissimilar bias (partisanship).

    We build W_bias with:
        W_bias[i,j] = |b_i - b_j| * A_ij   (symmetric)
    This pushes the SDP/Max-Cut to separate pairs with large bias difference.

    If `normalize=True`, divide by max(|b_i - b_j|) over edges to keep scale ~[0,1].
    """
    n = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}
    # Adjacency
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr")

    # Precompute edge bias diffs
    bi = np.array([b.get(u, 0.0) for u in nodes], dtype=float)
    # We'll fill a dense matrix (SDP needs dense anyway)
    Wb = np.zeros((n, n), dtype=float)

    # Iterate edges efficiently from CSR
    # For each row i, go through its neighbors j>i
    maxdiff = 1e-12
    for i in range(n):
        start, end = A.indptr[i], A.indptr[i+1]
        js = A.indices[start:end]
        for j in js:
            if j <= i:
                continue
            diff = abs(bi[i] - bi[j])
            Wb[i, j] = diff
            Wb[j, i] = diff
            if diff > maxdiff:
                maxdiff = diff

    if normalize and maxdiff > 0:
        Wb /= maxdiff

    return Wb


def _gw_rounding(X: np.ndarray, n_rounds: int = 32, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Goemans–Williamson rounding:
      - Factor X ≈ V V^T (Cholesky/eig)
      - Sample random hyperplanes r ~ N(0, I)
      - Labels s = sign(V r)

    Returns:
      s ∈ {+1, -1}^n
    """
    n = X.shape[0]
    if rng is None:
        rng = np.random.default_rng(12345)

    # Numerical symmetrization and clipping
    X = 0.5 * (X + X.T)
    # EVD (robust even if PSD is near-singular)
    vals, vecs = np.linalg.eigh(X)
    vals = np.clip(vals, a_min=0.0, a_max=None)
    V = vecs @ np.diag(np.sqrt(vals))

    best_s = None
    best_score = -np.inf

    for _ in range(n_rounds):
        r = rng.standard_normal(size=(n,))
        proj = V @ r
        s = np.where(proj >= 0.0, 1.0, -1.0)

        # trivial score: separation magnitude
        score = np.sum(np.abs(proj))
        if score > best_score:
            best_score = score
            best_s = s

    return best_s


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class BiasAwareSDP:
    """
    Bias-Aware Community Detection via SDP Relaxation (2-way cut).

    Objective (relaxed):
        minimize  < W, X >
        s.t. X ⪰ 0, diag(X) = 1

    with
        W = (1-α) * W_modularity  +  α * W_bias

    Where:
    - W_modularity is derived from the modularity matrix B.
      We use a Max-Cut view: larger positive weights encourage a *cut*,
      which roughly aligns with maximizing modularity under a 2-way split.
    - W_bias puts positive weight on pairs (i,j) with large |b_i - b_j|
      *when they are adjacent*, to encourage separating dissimilar-bias neighbors.

    After solving, we do GW rounding to obtain labels s ∈ {+1,-1}.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 1.0,
        solver: str = "SCS",
        max_iters: int = 5_000,
        eps_abs: float = 1e-4,
        eps_rel: float = 1e-4,
        verbose: bool = False,
        random_rounds: int = 32,
    ):
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.solver = solver
        self.max_iters = int(max_iters)
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.verbose = bool(verbose)
        self.random_rounds = int(random_rounds)

        # Fitted attributes
        self.nodes_: list | None = None
        self.labels_: dict | None = None        # node -> {0,1}
        self.sdp_val_: float | None = None
        self.runtime_: float | None = None
        self.X_: np.ndarray | None = None       # relaxed solution

    def _build_W(self, G: nx.Graph, b: dict) -> tuple[np.ndarray, list]:
        B, nodes = _modularity_matrix(G, gamma=self.gamma)
        # Convert modularity matrix B into a cut-weight matrix:
        #   We take W_mod = -B (so minimizing <W,X> ~ maximizing trace(B X))
        W_mod = -B
        W_bias = _bias_weight_matrix(G, nodes, b, normalize=True)

        W = (1.0 - self.alpha) * W_mod + self.alpha * W_bias
        # Symmetrize as a guard
        W = 0.5 * (W + W.T)
        return W, nodes

    def fit(self, G: nx.Graph, bias_scores: dict[int, float]) -> dict:
        """
        Solve the SDP and return a partition {node -> community_id ∈ {0,1}}.

        Notes:
        - This is a *2-way* split. For k>2, use split_k via recursive bisection.

        Returns:
            partition (dict): node -> {0,1}
        """
        t0 = time.time()
        W, nodes = self._build_W(G, bias_scores)
        n = len(nodes)

        # SDP variable
        X = cp.Variable((n, n), symmetric=True)
        constraints = [
            cp.diag(X) == 1.0,
            X >> 0
        ]
        obj = cp.Minimize(cp.sum(cp.multiply(W, X)))

        prob = cp.Problem(obj, constraints)
        solver_opts = dict(max_iters=self.max_iters)
        if self.solver == "SCS":
            solver_opts.update(dict(eps_abs=self.eps_abs, eps_rel=self.eps_rel))

        logger.info(f"🔷 Solving SDP (n={n}, solver={self.solver}) ...")
        prob.solve(solver=self.solver, verbose=self.verbose, **solver_opts)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            logger.warning(f"[WARN] SDP status: {prob.status}. Proceeding with best X found.")

        X_val = X.value
        if X_val is None:
            raise RuntimeError("SDP failed to produce a solution (X is None).")

        # Rounding
        s = _gw_rounding(X_val, n_rounds=self.random_rounds)
        # Map to {0,1}
        labels = (s > 0).astype(int)

        partition = {u: int(labels[i]) for i, u in enumerate(nodes)}

        self.nodes_ = nodes
        self.labels_ = partition
        self.sdp_val_ = float(prob.value) if prob.value is not None else float("nan")
        self.runtime_ = time.time() - t0
        self.X_ = X_val

        logger.info(f"✅ SDP solved in {self.runtime_:.2f}s | objective={self.sdp_val_:.6f}")
        return partition

    # -------------------------- k-way (optional) --------------------------
    def split_k(self, G: nx.Graph, bias_scores: dict[int, float], k: int = 4, min_block: int = 50) -> dict:
        """
        Obtain k communities via recursive bisection:
          - Run 2-way SDP on the current block
          - Pick the largest block and split again, until reaching k blocks
          - Stop splitting blocks smaller than `min_block`

        Returns:
            partition: node -> community_id in {0,1,...,k-1}
        """
        assert k >= 2
        # Initial one-block
        blocks = [list(G.nodes())]
        part = {u: 0 for u in G.nodes()}
        next_label = 1

        while len(blocks) < k:
            # choose the largest block to split
            blocks.sort(key=len, reverse=True)
            S = blocks[0]
            if len(S) < max(min_block, 2):
                break

            G_sub = G.subgraph(S).copy()
            b_sub = {u: bias_scores.get(u, 0.0) for u in G_sub.nodes()}
            sub_part = self.fit(G_sub, b_sub)  # returns 0/1

            # reassign labels for nodes in S
            S0 = [u for u, c in sub_part.items() if c == 0]
            S1 = [u for u, c in sub_part.items() if c == 1]
            # Keep S0 as original label of the block, assign new label to S1
            old_label = part[S[0]]
            for u in S0:
                part[u] = old_label
            for u in S1:
                part[u] = next_label

            # update blocks
            blocks = []
            for lab in set(part.values()):
                blocks.append([u for u, c in part.items() if c == lab])

            next_label += 1
            if next_label >= k:
                break

        return part
