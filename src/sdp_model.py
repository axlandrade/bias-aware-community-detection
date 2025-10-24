# src/sdp_model.py

import networkx as nx
import numpy as np
import cvxpy as cp
import time
from typing import Dict, Tuple, List, Optional

class BiasAwareSDP:
    """
    Implementação da detecção de comunidades com viés via Programação Semidefinida (SDP).

    Esta classe formula o problema como um programa semidefinido, maximizando uma
    função objetivo que combina modularidade estrutural e homogeneidade de viés.
    É a implementação matematicamente exata (relaxada) descrita no artigo.

    Attributes:
        alpha (float): Parâmetro de balanço entre estrutura (0.0) e viés (1.0).
        partition (Optional[Dict[int, int]]): Dicionário mapeando cada nó à sua comunidade (0 ou 1).
        execution_time (float): Tempo de execução do método `fit()` em segundos.
    """

    def __init__(self, alpha: float = 0.5, verbose: bool = False):
        """
        Inicializa o detector SDP.

        Args:
            alpha (float): Parâmetro de balanço entre 0 (apenas estrutura) e 1 (apenas viés).
            verbose (bool): Se True, imprime o log do solver CVXPY.
        """
        self.alpha = alpha
        self.verbose = verbose
        self.partition = None
        self.X_solution = None
        self.execution_time = 0
        self.objective_value = 0

    def fit(self, G: nx.Graph, bias_scores: Dict[int, float]):
        """
        Executa o algoritmo de detecção de comunidades no grafo fornecido.

        Args:
            G (nx.Graph): O grafo a ser particionado.
            bias_scores (Dict[int, float]): Dicionário com o score de viés para cada nó.
        """
        if not G.nodes():
            print("⚠️ Aviso: O grafo está vazio.")
            self.partition = {}
            return
            
        start_time = time.time()
        nodes = list(G.nodes())
        n = len(nodes)

        # 1. Construir matrizes de modularidade (B) e viés (C)
        B = self._build_modularity_matrix(G, nodes)
        C = self._build_bias_matrix(bias_scores, nodes)

        # 2. Resolver o problema SDP
        X_solution, obj_value = self._solve_sdp(B, C, n)
        self.X_solution = X_solution
        self.objective_value = obj_value if obj_value is not None else 0

        # 3. Arredondar a solução para obter a partição
        if X_solution is not None:
            partition_idx = self._round_solution(X_solution)
            self.partition = {nodes[i]: partition_idx[i] for i in range(n)}
        else:
            self.partition = {node: 0 for node in nodes} # Fallback

        self.execution_time = time.time() - start_time

    def _build_modularity_matrix(self, G: nx.Graph, nodes: List[int]) -> np.ndarray:
        """Constrói a matriz de modularidade B."""
        m = G.number_of_edges()
        if m == 0:
            return nx.to_numpy_array(G, nodelist=nodes)
            
        A = nx.to_numpy_array(G, nodelist=nodes)
        degrees = np.array([G.degree(node) for node in nodes])
        B = A - np.outer(degrees, degrees) / (2 * m)
        return B

    def _build_bias_matrix(self, bias_scores: Dict[int, float], nodes: List[int]) -> np.ndarray:
        """Constrói a matriz de viés C."""
        bias_vector = np.array([bias_scores[node] for node in nodes])
        C = np.outer(bias_vector, bias_vector)
        return C

    def _solve_sdp(self, B: np.ndarray, C: np.ndarray, n: int) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Define e resolve o problema de otimização SDP."""
        X = cp.Variable((n, n), symmetric=True)
        
        # Matriz objetivo combinada
        M = (1 - self.alpha) * B + self.alpha * C

        objective = cp.Maximize(cp.trace(M @ X))
        constraints = [X >> 0, cp.diag(X) == 1]
        
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.SCS, verbose=self.verbose)
            if problem.status in ["infeasible", "unbounded"]:
                print(f"⚠️ Aviso: Solver retornou status '{problem.status}'.")
                return None, None
            return X.value, problem.value
        except cp.error.SolverError:
            print("⚠️ Aviso: Erro no solver SCS. O problema pode ser muito grande ou mal condicionado.")
            return None, None

    def _round_solution(self, X: np.ndarray) -> Dict[int, int]:
        """Arredonda a matriz solução X usando decomposição espectral."""
        eigenvalues, eigenvectors = np.linalg.eigh(X)
        principal_eigenvector = eigenvectors[:, -1] # Autovetor associado ao maior autovalor

        # Particiona os nós com base no sinal do componente do autovetor
        partition = {i: 0 if principal_eigenvector[i] >= 0 else 1 for i in range(len(principal_eigenvector))}
        
        # Caso de fallback se todos os nós caírem na mesma comunidade
        if len(set(partition.values())) == 1:
            second_eigenvector = eigenvectors[:, -2]
            partition = {i: 0 if second_eigenvector[i] >= 0 else 1 for i in range(len(second_eigenvector))}
            
        return partition

    def get_communities(self) -> Dict[int, int]:
        """
        Retorna a partição de comunidades calculada.

        Returns:
            Dict[int, int]: Dicionário {nó: id_comunidade}.

        Raises:
            ValueError: Se o método `fit()` não tiver sido executado.
        """
        if self.partition is None:
            raise ValueError("O método `fit()` deve ser executado primeiro.")
        return self.partition