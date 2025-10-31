# Bias-Aware Community Detection via Semidefinite Programming and Heuristics

**Authors:** Axl S. Andrade, Nelson Maculan, Ronaldo M. Gregório, Sérgio A. Monteiro, Vitor S. Ponciano

This repository provides the implementation and experimental framework for the bias-aware community detection methodology, as proposed in the paper: "*Social Bias Detection in Social Networks via Semidefinite Programming and Structural Graph Analysis*".

The primary objective of this code is to partition social network graphs by balancing two competing objectives:
1.  **Structural Cohesion (Modularity):** Communities should be densely connected internally.
2.  **Ideological Homogeneity (Bias):** Members within the same community should share a similar political/ideological bias.

This repository provides implementations for both the exact solution (SDP) and the heuristic approximation (Enhanced Louvain) described in the article.

## Methodology

The methodology is centered on a unified objective function, controlled by a hyperparameter `alpha` (`α`), which weights the importance of structure (`B`, the modularity matrix) versus bias (`C`, the bias covariance matrix).

`M = (1 - α) * B + α * C`

The problem is then solved by maximizing `Tr(M*X)`, which is approached in two ways:
1.  **`BiasAwareSDP`**: A Semidefinite Programming (SDP) relaxation that finds the optimal solution. It is computationally expensive and only feasible for small graphs (< 1500 nodes).
2.  **`EnhancedLouvainWithBias`**: A fast and scalable heuristic, based on the Louvain algorithm, which approximates the optimal solution in large-scale graphs.

## Repository Structure

The code is modularized to allow for easy switching between datasets (TwiBot-20 and TwiBot-22) and a clear separation of concerns.

| File                            | Purpose                                                                                                                                               |
| :------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `notebooks/Twi-Bot-20-22.ipynb` | **Main Notebook.** Orchestrates the entire experimental pipeline: loading, computation, algorithm execution, and evaluation.                          |
| `src/config.py`                 | **Central Configuration File.** The user must edit this file to select the dataset (`DATASET_MODE`), paths, and hyperparameters (like `ALPHA`).       |
| `src/data_utils.py`             | Contains the `TwiBot20Loader` (for the single JSON format) and `TwiBot22Loader` (for the CSV + multiple JSONs format) classes.                        |
| `src/bias_calculator.py`        | Contains the logic to load the political bias AI model (`matous-volf/political-leaning-politics`) and calculate bias scores for both dataset formats. |
| `src/heuristic.py`              | Implementation of the `EnhancedLouvainWithBias` heuristic.                                                                                            |
| `src/sdp_model.py`              | Implementation of the `BiasAwareSDP` exact solver using `cvxpy`.                                                                                      |
| `src/evaluation.py`             | Implementation of the `ComprehensiveEvaluator` to calculate all metrics (Modularity, Bias Separation, Purity, etc.).                                  |

## Prerequisites and Installation

The project requires Python 3.10+ and several scientific libraries. Using a virtual environment is highly recommended.

1.  Clone this repository:
    ```bash
    git clone [https://github.com/axlandrade/bias-aware-community-detection](https://github.com/axlandrade/bias-aware-community-detection)
    cd bias-aware-community-detection
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # (Linux/macOS)
    # or
    .\.venv\Scripts\activate   # (Windows)
    ```

3.  Install dependencies. For environments with an NVIDIA GPU (recommended), install PyTorch with CUDA support first:
    ```bash
    # Install PyTorch (CUDA 12.1 or higher)
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    
    # Install remaining libraries
    pip install networkx python-louvain pandas tqdm psutil transformers matplotlib seaborn tabulate cvxpy jupyter
    ```
    (Alternatively, create a `requirements.txt` file with the libraries above or run the installation cells in the notebook).

## Data Acquisition (Important)

This repository **does not** distribute the TwiBot-20 or TwiBot-22 datasets. Due to their terms of use, researchers must obtain the data directly from the official sources and place them in the project's root folder.

* **TwiBot-20:** The dataset (containing `train.json`, `dev.json`, etc.) can be requested from its official repository: [BunsenFeng/TwiBot-20](https://github.com/GabrielHam/TwiBot-20)
* **TwiBot-22:** The dataset (containing `label.csv`, `edge.csv`, `tweet/` folder, etc.) can be requested from the official website: [LuoUndergradXJTU/TwiBot-22](https://twibot22.github.io/)

The expected folder structure in the project root is:
```
/bias-aware-community-detection
    /TwiBot-20
        train.json
        dev.json
        test.json
        support.json
    /TwiBot-22
        /data
            label.csv
            edge.csv
        /tweet
            tweet_0.json
            ...
    /src
        __init__.py
        bias_calculator.py
        config.py
        data_utils.py
        evaluation.py
        heuristic.py
        sdp_model.py
    /notebooks
        Twi-Bot-20-22.ipynb
```

## Execution Guide

The experiment is controlled centrally via `src/config.py`.

### 1. Select the Dataset

Open `src/config.py` and set the `DATASET_MODE` flag to the desired dataset:

```python
# To run TwiBot-20
DATASET_MODE = "TWIBOT_20"

# To run TwiBot-22
DATASET_MODE = "TWIBOT_22"
```

### 2. (If TwiBot-20) Select the Subset

If using TwiBot-20, define which JSON file to use (e.g., `train.json`, `test.json`) in `src/config.py`:

```python
# In src/config.py, inside the if DATASET_MODE == "TWIBOT_20" block:
DATASET_FILE_PATH = os.path.join(DATA_DIR, "train.json") # or dev.json, test.json, support.json
```

### 3. (If TwiBot-22) Set Limit (Recommended)

TwiBot-22 is massive. For an initial test, it is highly recommended to limit the number of nodes to load. In the `notebooks/Twi-Bot-20-22.ipynb` notebook, in the "Step 1" cell, change `max_nodes`:

```python
# In the notebook's "Step 1" cell
# 50000 is recommended for a test; None for the full dataset (requires >= 64GB RAM)
G, bot_labels = data_loader.load_and_build_graph(max_nodes=50000) 
```

### 4. Run the Notebook

Open and run the `notebooks/Twi-Bot-20-22.ipynb` notebook cell by cell. The notebook will:
1.  Load the configuration and the correct `Loader` (TwiBot20Loader or TwiBot22Loader).
2.  Load the graph and labels.
3.  Calculate bias scores (`BiasCalculator`), downloading the AI model on the first run.
4.  Run the Heuristic (`EnhancedLouvainWithBias`) and the Standard Louvain (Baseline).
5.  Generate comparative results tables (`ComprehensiveEvaluator`).
6.  (Optional) Run the SDP solver (`BiasAwareSDP`) if the resulting graph is small enough (`< 1500` nodes).

## License

This project is licensed under the terms of the [MIT License](https://github.com/axlandrade/bias-aware-community-detection/blob/main/LICENSE).

## Citation

If you use this code or methodology in your research, please cite the original paper:

[Insert formal paper citation here when published]