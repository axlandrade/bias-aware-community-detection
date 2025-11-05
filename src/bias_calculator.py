"""
bias_calculator.py
----------------------
Computation of ideological bias and ground-truth labels for
Bias-Aware Community Detection using the Twitter Political Bias Dataset.

This module extracts user-level ideological orientation (bias) based solely on
the 'Partisanship' attribute. The 'Misinformation' variable, although available,
is excluded from computation to maintain conceptual alignment with the
modularity–bias framework, where bias quantifies ideological consistency.

Author: Axl S. Andrade et al.
Affiliation: Universidade Federal Rural do Rio de Janeiro (UFRRJ)
"""

import logging
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#                          Bias Score Computation
# --------------------------------------------------------------------------- #
def compute_bias_from_measures(df: pd.DataFrame) -> dict:
    """
    Compute individual ideological bias scores from the Partisanship column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns:
        - 'ID' : unique user identifier
        - 'Partisanship' : continuous ideological score in [-1, +1]

    Returns
    -------
    dict
        Mapping {user_id: bias_value}, where bias_value ∈ [-1, +1].

    Notes
    -----
    - Bias is derived exclusively from the 'Partisanship' field.
    - 'Misinformation' is retained in the dataset but not used for computation.
    - Missing or invalid entries are filtered automatically.
    """
    if 'Partisanship' not in df.columns:
        raise ValueError("Column 'Partisanship' not found in dataset.")

    df_valid = df[['ID', 'Partisanship']].dropna()
    bias_dict = {int(row.ID): float(row.Partisanship) for _, row in df_valid.iterrows()}

    logging.info(f"✅ Computed bias for {len(bias_dict):,} users "
                 f"(range = [{min(bias_dict.values()):.3f}, {max(bias_dict.values()):.3f}])")

    return bias_dict


# --------------------------------------------------------------------------- #
#                         Ground-Truth Label Generation
# --------------------------------------------------------------------------- #
def generate_ground_truth(df: pd.DataFrame) -> dict:
    """
    Generate binary ground-truth ideological labels based on Partisanship.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the 'Partisanship' column.

    Returns
    -------
    dict
        Mapping {user_id: label}, where:
            - label = 1 for right-leaning (Partisanship > 0)
            - label = 0 for left-leaning (Partisanship < 0)

    Notes
    -----
    - Users with Partisanship exactly equal to 0 are excluded.
    - This binary labeling supports ARI/NMI evaluation metrics
      that require discrete ground-truth categories.
    """
    if 'Partisanship' not in df.columns:
        raise ValueError("Column 'Partisanship' not found in dataset.")

    df_valid = df[['ID', 'Partisanship']].dropna()
    gt = {
        int(row.ID): (1 if row.Partisanship > 0 else 0)
        for _, row in df_valid.iterrows()
        if row.Partisanship != 0
    }

    logging.info(f"✅ Generated ground-truth labels for {len(gt):,} users "
                 f"({sum(gt.values()):,} right-leaning, {len(gt)-sum(gt.values()):,} left-leaning)")

    return gt


# --------------------------------------------------------------------------- #
#                            Optional Statistical Summary
# --------------------------------------------------------------------------- #
def summarize_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize ideological bias statistics for dataset documentation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the 'Partisanship' column.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with:
            - mean_bias
            - std_bias
            - skewness
            - proportion_right
            - proportion_left

    Notes
    -----
    - Provides a descriptive overview for academic reporting.
    - May be used to accompany the dataset description in publications.
    """
    values = df['Partisanship'].dropna().values
    summary = pd.DataFrame([{
        "mean_bias": np.mean(values),
        "std_bias": np.std(values),
        "skewness": pd.Series(values).skew(),
        "proportion_right": np.mean(values > 0),
        "proportion_left": np.mean(values < 0)
    }])

    logging.info(f"📊 Bias summary: {summary.to_dict(orient='records')[0]}")
    return summary
