import networkx as nx
import community.community_louvain as louvain
import numpy as np
from collections import defaultdict
import time

class EnhancedLouvainWithBias:
    def __init__(self, alpha=0.5, max_iterations=20, verbose=True):
        self.alpha = alpha  # Balance entre estrutura e viés
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.communities = None
        self.execution_time = 0
        
    def fit(self, G, bias_scores, num_communities=2):
        """Executa a detecção de comunidades considerando viés"""
        start_time = time.time()
        
        if self.verbose:
            print(f"🎯 Executando Enhanced Louvain (α={self.alpha})...")
        
        # Fase 1: Detecção inicial com Louvain padrão
        initial_partition = louvain.best_partition(G)
        
        # Fase 2: Refinamento considerando viés
        refined_partition = self._refine_with_bias(G, initial_partition, bias_scores, num_communities)
        
        self.communities = refined_partition
        self.execution_time = time.time() - start_time
        
        if self.verbose:
            print(f"✅ Concluído em {self.execution_time:.2f}s")
            self._print_community_stats(refined_partition, bias_scores)
    
    def _refine_with_bias(self, G, partition, bias_scores, num_communities):
        """Refina as comunidades considerando similaridade de viés"""
        nodes = list(G.nodes())
        
        # Calcular viés médio por comunidade inicial
        comm_bias = defaultdict(list)
        for node, comm in partition.items():
            comm_bias[comm].append(bias_scores.get(node, 0))
        
        comm_avg_bias = {comm: np.mean(biases) for comm, biases in comm_bias.items()}
        
        # Reatribuir nós baseado na similaridade de viés
        refined_partition = {}
        
        for node in nodes:
            current_comm = partition[node]
            node_bias = bias_scores.get(node, 0)
            
            # Encontrar comunidade com viés mais similar
            best_comm = current_comm
            min_bias_diff = abs(node_bias - comm_avg_bias[current_comm])
            
            for comm, avg_bias in comm_avg_bias.items():
                bias_diff = abs(node_bias - avg_bias)
                if bias_diff < min_bias_diff:
                    min_bias_diff = bias_diff
                    best_comm = comm
            
            refined_partition[node] = best_comm
        
        # Garantir número desejado de comunidades
        return self._balance_communities(refined_partition, num_communities)
    
    def _balance_communities(self, partition, num_communities):
        """Balanceia comunidades se necessário"""
        unique_comms = set(partition.values())
        
        if len(unique_comms) <= num_communities:
            return partition
        
        # Se há muitas comunidades, fundir as menores
        comm_sizes = defaultdict(int)
        for comm in partition.values():
            comm_sizes[comm] += 1
        
        # Manter apenas as N maiores comunidades
        top_comms = sorted(comm_sizes.keys(), key=lambda x: comm_sizes[x], reverse=True)[:num_communities]
        
        # Reatribuir nós de comunidades menores
        refined = {}
        for node, comm in partition.items():
            if comm in top_comms:
                refined[node] = comm
            else:
                # Atribuir à comunidade mais próxima em tamanho
                refined[node] = top_comms[0]
        
        return refined
    
    def _print_community_stats(self, partition, bias_scores):
        """Imprime estatísticas das comunidades"""
        comm_stats = defaultdict(list)
        for node, comm in partition.items():
            comm_stats[comm].append(bias_scores.get(node, 0))
        
        print("📊 Estatísticas das Comunidades:")
        for comm, biases in comm_stats.items():
            avg_bias = np.mean(biases)
            std_bias = np.std(biases)
            print(f"  Comunidade {comm}: {len(biases)} nós, "
                  f"viés médio: {avg_bias:.3f} (±{std_bias:.3f})")
    
    def get_communities(self):
        return self.communities