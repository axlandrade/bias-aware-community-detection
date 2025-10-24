# src/heuristic.py

import networkx as nx
import numpy as np
import random
import time
from collections import defaultdict
import community.community_louvain as community_louvain
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, List, Optional

class EnhancedLouvainWithBias:
    """
    Implementação da heurística eficiente para detecção de comunidades com viés.

    Este método utiliza o algoritmo de Louvain como ponto de partida e, em seguida,
    refina iterativamente a partição para otimizar a mesma função objetivo do SDP,
    oferecendo um grande ganho de velocidade.

    Attributes:
        alpha (float): Parâmetro de balanço entre estrutura (0.0) e viés (1.0).
        partition (Optional[Dict[int, int]]): Dicionário final da partição de comunidades.
        execution_time (float): Tempo de execução do método fit().
    """
    def __init__(self, alpha: float = 0.4, max_iterations: int = 100, verbose: bool = False):
        """
        Inicializa a heurística.

        Args:
            alpha (float): Parâmetro de balanço.
            max_iterations (int): Número máximo de iterações de refinamento.
            verbose (bool): Se True, imprime informações de progresso.
        """
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.partition = None
        self.execution_time = 0

    def fit(self, G: nx.Graph, bias_scores: Dict[int, float], num_communities: int = 2):
        """
        Executa o algoritmo de detecção heurístico.

        Args:
            G (nx.Graph): O grafo a ser particionado.
            bias_scores (Dict[int, float]): Scores de viés para cada nó.
            num_communities (int): O número desejado de comunidades final.
        """
        start_time = time.time()
        
        # 1. Obter partição inicial de alta modularidade com Louvain
        partition = community_louvain.best_partition(G)

        # 2. Mesclar comunidades com base na similaridade de viés se houver mais que o alvo
        if len(set(partition.values())) > num_communities:
            partition = self._merge_communities(partition, bias_scores, num_communities)

        # 3. Refinar a partição iterativamente considerando o viés
        if self.alpha > 0:
            partition = self._refine_with_bias(G, partition, bias_scores)

        self.partition = partition
        self.execution_time = time.time() - start_time
        
    def _merge_communities(self, partition: Dict[int, int], bias_scores: Dict[int, float], target_num: int) -> Dict[int, int]:
        """Mescla comunidades usando clustering hierárquico no viés médio."""
        community_biases = defaultdict(list)
        for node, comm in partition.items():
            community_biases[comm].append(bias_scores[node])

        avg_biases = {comm: np.mean(biases) for comm, biases in community_biases.items()}
        comm_ids = list(avg_biases.keys())
        bias_values = np.array([avg_biases[c] for c in comm_ids]).reshape(-1, 1)

        clustering = AgglomerativeClustering(n_clusters=target_num)
        new_labels = clustering.fit_predict(bias_values)
        comm_mapping = {comm_ids[i]: new_labels[i] for i in range(len(comm_ids))}

        return {node: comm_mapping[old_comm] for node, old_comm in partition.items()}
        
    def _refine_with_bias(self, G: nx.Graph, partition: Dict[int, int], bias_scores: Dict[int, float]) -> Dict[int, int]:
        """Refina iterativamente a partição para maximizar o ganho combinado."""
        improved = True
        iteration = 0
        while improved and iteration < self.max_iterations:
            improved = False
            iteration += 1
            
            # Recalcula o viés médio de cada comunidade a cada iteração
            community_biases = defaultdict(list)
            for node, comm in partition.items():
                community_biases[comm].append(bias_scores[node])
            avg_biases = {comm: np.mean(biases) for comm, biases in community_biases.items()}

            nodes = list(G.nodes())
            random.shuffle(nodes)

            for node in nodes:
                current_comm = partition[node]
                neighbor_comms = {partition[n] for n in G.neighbors(node) if partition[n] != current_comm}

                if not neighbor_comms:
                    continue

                # Encontra o melhor movimento para o nó
                best_gain = 0
                best_comm = current_comm
                for target_comm in neighbor_comms:
                    gain = self._compute_gain(G, node, current_comm, target_comm,
                                             partition, bias_scores, avg_biases)
                    if gain > best_gain:
                        best_gain = gain
                        best_comm = target_comm
                
                # Se um movimento melhorou o ganho, atualiza a partição
                if best_comm != current_comm:
                    partition[node] = best_comm
                    improved = True
                    
        return partition

    def _compute_gain(self, G: nx.Graph, node: int, current_comm: int, target_comm: int,
                     partition: Dict[int, int], bias_scores: Dict[int, float], 
                     avg_biases: Dict[int, float]) -> float:
        """
        Calcula o ganho de mover um nó para uma nova comunidade.
        Esta é a função central que emula o objetivo do SDP.
        """
        # --- Ganho Estrutural ---
        # Favorece mover o nó para uma comunidade onde ele tem mais vizinhos.
        # É uma aproximação local da mudança na modularidade.
        neighbors = list(G.neighbors(node))
        if not neighbors:
            structural_gain = 0
        else:
            neighbors_in_target = sum(1 for n in neighbors if partition[n] == target_comm)
            neighbors_in_current = sum(1 for n in neighbors if partition[n] == current_comm)
            structural_gain = (neighbors_in_target - neighbors_in_current) / len(neighbors)

        # --- Ganho de Viés ---
        # Favorece mover o nó para uma comunidade cujo viés médio é mais
        # próximo do seu próprio viés. Maximiza a homogeneidade.
        node_bias = bias_scores[node]
        current_bias_dist = abs(node_bias - avg_biases[current_comm])
        target_bias_dist = abs(node_bias - avg_biases[target_comm])
        bias_gain = current_bias_dist - target_bias_dist

        # O ganho total é a média ponderada pelo parâmetro alpha.
        return (1 - self.alpha) * structural_gain + self.alpha * bias_gain

    def get_communities(self) -> Dict[int, int]:
        """Retorna a partição de comunidades calculada."""
        if self.partition is None:
            raise ValueError("O método `fit()` deve ser executado primeiro.")
        return self.partition