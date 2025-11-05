"""
config.py
----------------------
Centralized configuration module for Bias-Aware Community Detection.

Defines all key experimental parameters, file paths, and logging setup
used across the /src package. The configuration is dataset-agnostic,
allowing direct use in both the FACTOID and Twitter pipelines.

Author: Axl S. Andrade et al.
Affiliation: Universidade Federal Rural do Rio de Janeiro (UFRRJ)
"""

import os
import logging


# --------------------------------------------------------------------------- #
#                           Logging Configuration
# --------------------------------------------------------------------------- #
def setup_logging():
    """
    Configure global logging style for the experimental environment.

    Notes
    -----
    - Logs are printed to console with timestamps and message level.
    - Intended for reproducible and auditable academic experiments.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    logging.info("✅ Logging initialized.")


setup_logging()


# --------------------------------------------------------------------------- #
#                         Experiment-Level Constants
# --------------------------------------------------------------------------- #
ALPHA_FIXED = 0.5
"""
float: Default weighting parameter α for the Bias-Aware Louvain method.
α ∈ [0,1] controls the trade-off between structural modularity Q(C)
and ideological coherence B(C).
"""

ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
"""
list[float]: Discrete grid of α values used for parameter sweep experiments.
Each run evaluates performance (ARI, NMI) at a different α level.
"""

LIMIT_USERS = 15000
"""
int: Maximum number of users to load for large-scale datasets (e.g., Twitter).
This parameter allows sampling for performance benchmarking.
"""

# --------------------------------------------------------------------------- #
#                           Path Configuration
# --------------------------------------------------------------------------- #
BASE_PATH = os.getcwd()
"""
str: Base working directory for experiments. Adjust manually if needed.
"""

# FACTOID dataset paths
FACTOID_PATHS = {
    "GRAPH": os.path.join(BASE_PATH, "FACTOID", "social_graph_data", "social_graph.gml"),
    "BIAS": os.path.join(BASE_PATH, "FACTOID", "factoid_bias_groundtruth.csv"),
    "RESULTS": os.path.join(BASE_PATH, "FACTOID", "results"),
}

# Twitter dataset paths
TWITTER_PATHS = {
    "FRIENDS": os.path.join(BASE_PATH, "Twitter", "anonymized-friends.json"),
    "SHARES": os.path.join(BASE_PATH, "Twitter", "anonymized-shares.json"),
    "MEASURES": os.path.join(BASE_PATH, "Twitter", "measures.tab"),
    "RESULTS": os.path.join(BASE_PATH, "Twitter", "results"),
}


# --------------------------------------------------------------------------- #
#                           Execution Metadata
# --------------------------------------------------------------------------- #
EXPERIMENT_METADATA = {
    "authors": [
        "Axl S. Andrade",
        "Nelson Maculan",
        "Ronaldo M. Gregório",
        "Sérgio A. Monteiro",
        "Vitor S. Ponciano",
    ],
    "institution": "Universidade Federal Rural do Rio de Janeiro (UFRRJ)",
    "project": "Bias-Aware Community Detection",
    "version": "1.0.0",
}
"""
dict: Metadata used for experiment documentation and automatic logging.
"""

logging.info("⚙️ Configuration loaded successfully.")
logging.info(f"📦 Current base path: {BASE_PATH}")
logging.info(f"α fixed = {ALPHA_FIXED}, user limit = {LIMIT_USERS:,}")