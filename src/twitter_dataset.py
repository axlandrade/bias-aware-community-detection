"""
twitter_dataset.py
----------------------
Data ingestion and preprocessing for the Twitter Political Bias Dataset
(Indiana University / Harvard Dataverse).

This module constructs the social graph and associated bias structures used for
Bias-Aware Community Detection. It handles the following pipeline:

    1. Load partisanship and misinformation measures (measures.tab)
    2. Construct user graph from anonymized-friends.json
    3. Compute ideological bias and ground-truth labels
    4. Return cleaned and aligned structures ready for model fitting

Author: Axl S. Andrade et al.
Affiliation: Universidade Federal Rural do Rio de Janeiro (UFRRJ)
"""

import json
import logging
import pandas as pd
import networkx as nx
from tqdm import tqdm
from src.bias_calculator import compute_bias_from_measures, generate_ground_truth


# --------------------------------------------------------------------------- #
#                          Data Loading and Graph Construction
# --------------------------------------------------------------------------- #
def load_measures_tab(path_tab: str) -> pd.DataFrame:
    """
    Load precomputed user measures from the 'measures.tab' file.

    Parameters
    ----------
    path_tab : str
        Path to the tab-delimited file (TSV format).

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
            - ID : user identifier
            - Partisanship : continuous score in [-1, 1]
            - Misinformation : proportion of misinformation exposure

    Notes
    -----
    - The dataset originates from the Twitter Political Bias corpus curated
      by the Observatory on Social Media (OSoMe) at Indiana University.
    - This dataset is publicly available via Harvard Dataverse.
    """
    logging.info(f"📂 Loading user measures from {path_tab} ...")
    df = pd.read_csv(path_tab, sep="\t")
    logging.info(f"✅ Measures loaded: {len(df):,} users.")
    return df


def load_friends_graph(path_json: str, limit: int = None) -> nx.Graph:
    """
    Construct an undirected social graph from anonymized-friends.json.

    Parameters
    ----------
    path_json : str
        Path to the anonymized friends JSON file.
    limit : int, optional
        Maximum number of users to load (for experimental subsets).

    Returns
    -------
    nx.Graph
        Undirected NetworkX graph representing user friendships.

    Notes
    -----
    - The JSON file maps user IDs to lists of friends (mutual follow relations).
    - The resulting graph may contain fewer than N nodes if users are filtered.
    """
    logging.info(f"🔧 Building social graph from {path_json} ...")
    G = nx.Graph()

    with open(path_json, "r") as f:
        # Manual iterative parsing to handle large JSON (>2GB)
        buffer = f.read().strip()
        if buffer.startswith("{") and buffer.endswith("}"):
            buffer = buffer[1:-1]
        pairs = buffer.split("],")
        total = len(pairs) if not limit else min(limit, len(pairs))

        for i, entry in enumerate(tqdm(pairs[:total], total=total)):
            try:
                k, v = entry.split(":", 1)
                user = int(k.strip().replace('"', '').replace("{", ""))
                friends = json.loads(v.strip() + "]" if not v.strip().endswith("]") else v.strip())
                for friend in friends:
                    G.add_edge(user, int(friend))
            except Exception:
                continue

    logging.info(f"📊 Graph built: |V|={G.number_of_nodes():,} |E|={G.number_of_edges():,}")
    return G


# --------------------------------------------------------------------------- #
#                             Dataset Construction
# --------------------------------------------------------------------------- #
def build_twitter_dataset(measures_path: str, friends_path: str, limit: int = None):
    """
    High-level dataset assembly function for the Twitter Political Bias corpus.

    Parameters
    ----------
    measures_path : str
        Path to the 'measures.tab' file.
    friends_path : str
        Path to the 'anonymized-friends.json' file.
    limit : int, optional
        Restrict the number of users for controlled experiments.

    Returns
    -------
    tuple (G, b, gt)
        - G : nx.Graph — social graph
        - b : dict — bias scores per user
        - gt : dict — ideological ground-truth labels

    Workflow
    --------
    1. Load precomputed measures
    2. Build the friendship graph
    3. Generate bias and ground-truth mappings
    4. Intersect all components for a consistent dataset
    """
    logging.info("🚀 Initiating Twitter dataset pipeline...")

    # Step 1: Measures
    df_measures = load_measures_tab(measures_path)

    # Step 2: Graph
    G = load_friends_graph(friends_path, limit)

    # Step 3: Bias + Ground-truth
    b = compute_bias_from_measures(df_measures)
    gt = generate_ground_truth(df_measures)

    # Step 4: Intersection
    common_users = list(set(G.nodes()) & set(b.keys()))
    G_sub = G.subgraph(common_users).copy()
    b = {u: b[u] for u in common_users if u in b}
    gt = {u: gt[u] for u in common_users if u in gt}

    logging.info(
        f"✅ Twitter dataset ready: |V|={len(G_sub):,} |E|={G_sub.number_of_edges():,} |b|={len(b):,} |gt|={len(gt):,}"
    )

    return G_sub, b, gt
