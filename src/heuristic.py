# src/heuristic.py - VERSÃO CORRIGIDA (DE ACORDO COM O ARTIGO)

import networkx as nx
import community.community_louvain as louvain
import numpy as np
from collections import defaultdict
import time
import random
from .config import Config # Import relativo

class EnhancedLouvainWithBias:
    """
    Implementa a heurística descrita na Seção 5.2 do artigo.
    Combina modularidade (Louvain) com um refinamento iterativo que
    balanceia ganho estrutural e ganho de viés usando 'alpha'.
    """
    def __init__(self, alpha=1, max_iterations=100, verbose=True):
        self.alpha = alpha
        self.max_iterations_refine = max_iterations # Iterações para o refinamento
        self.verbose = verbose
        self.communities = None
        self.execution_time = 0
        self.config = Config()
        
    def fit(self, G, bias_scores, num_communities=2):
        """Executa a detecção de comunidades."""
        start_time = time.time()
        
        if not G.nodes():
            if self.verbose: print("⚠️ Grafo vazio.")
            self.communities = {}
            return

        if self.verbose:
            print(f"🎯 Executando Enhanced Louvain (α={self.alpha})...")
        
        # 1. Obter partição inicial com Louvain padrão
        if self.verbose: print("   Fase 1: Executando Louvain padrão para partição inicial...")
        partition = louvain.best_partition(G, random_state=self.config.RANDOM_STATE)
        
        # 2. Refinamento iterativo (O ALGORITMO DO ARTIGO)
        if self.verbose: print(f"   Fase 2: Iniciando refinamento iterativo (max_iter={self.max_iterations_refine})...")
        partition, moves = self._iterative_refinement(G, partition, bias_scores)
        
        # 3. Balancear/Mesclar comunidades para o número desejado
        # (O artigo menciona isso como passo opcional)
        if self.verbose: print(f"   Fase 3: Balanceando para {num_communities} comunidades...")
        final_partition = self._balance_communities(partition, num_communities)
        
        self.communities = final_partition
        self.execution_time = time.time() - start_time
        
        if self.verbose:
            print(f"✅ Concluído em {self.execution_time:.2f}s ({moves} movimentos no refinamento)")
            self._print_community_stats(self.communities, bias_scores)

    def _get_community_avg_bias(self, partition, bias_scores):
        """Calcula o viés médio de cada comunidade."""
        comm_bias_sum = defaultdict(float)
        comm_bias_count = defaultdict(int)
        for node, comm in partition.items():
            comm_bias_sum[comm] += bias_scores.get(node, 0)
            comm_bias_count[comm] += 1
        
        comm_avg_bias = {
            comm: (comm_bias_sum[comm] / comm_bias_count[comm])
            for comm in comm_bias_sum
            if comm_bias_count[comm] > 0
        }
        return comm_avg_bias

    def _iterative_refinement(self, G, partition, bias_scores):
        """
        Implementa o Refinamento Iterativo da Seção 5.2 [cite: 122-129].
        Move nós entre comunidades para maximizar o ganho combinado.
        """
        nodes = list(G.nodes())
        total_moves = 0
        
        for i in range(self.max_iterations_refine):
            moves_in_iter = 0
            random.shuffle(nodes) # Processar em ordem aleatória
            
            # Calcular médias de viés no início de cada iteração
            comm_avg_bias = self._get_community_avg_bias(partition, bias_scores)
            
            for node in nodes:
                current_comm = partition[node]
                node_bias = bias_scores.get(node, 0)
                
                # Vizinhos e suas comunidades
                neighbor_comms = defaultdict(int)
                for neighbor in G[node]:
                    if neighbor != node:
                        neighbor_comms[partition[neighbor]] += 1
                
                # Calcular ganho para a comunidade atual
                current_structural_gain = neighbor_comms.get(current_comm, 0)
                current_bias_gain = -abs(node_bias - comm_avg_bias.get(current_comm, 0)) # Negativo, pois queremos minimizar a distância
                
                best_comm = current_comm
                best_gain = -np.inf # Começa negativo

                # Avaliar mover para outras comunidades
                for comm, structural_links in neighbor_comms.items():
                    if comm == current_comm: continue
                    
                    # 1. Ganho Estrutural (Eq. Artigo [cite: 126])
                    # (links para nova comm) - (links para comm atual)
                    gain_structural = structural_links - current_structural_gain
                    
                    # 2. Ganho de Viés (Eq. Artigo [cite: 128])
                    # (distância atual) - (distância nova) -> queremos maximizar isso
                    gain_bias = (abs(node_bias - comm_avg_bias.get(current_comm, 0)) - 
                                 abs(node_bias - comm_avg_bias.get(comm, 0)))

                    # 3. Ganho Total (Eq. Artigo [cite: 123])
                    total_gain = (1 - self.alpha) * gain_structural + self.alpha * gain_bias
                    
                    if total_gain > best_gain:
                        best_gain = total_gain
                        best_comm = comm
                
                # Se um movimento for benéfico, execute-o
                if best_comm != current_comm and best_gain > 0:
                    partition[node] = best_comm
                    moves_in_iter += 1
                    
                    # Recalcular médias de viés após o movimento (caro, mas preciso)
                    comm_avg_bias = self._get_community_avg_bias(partition, bias_scores)

            total_moves += moves_in_iter
            if self.verbose: print(f"      Iter {i+1}/{self.max_iterations_refine}: {moves_in_iter} movimentos.")
            if moves_in_iter == 0: # Convergência [cite: 129]
                if self.verbose: print("      Convergência atingida.")
                break
                
        return partition, total_moves

    def _balance_communities(self, partition, num_communities):
        """Garante o número K de comunidades (Seção 5.2.2 [cite: 120])"""
        unique_comms = list(set(partition.values()))
        
        if len(unique_comms) <= num_communities:
            return partition
        
        # Se há muitas comunidades, fundir as menores nas K maiores
        comm_sizes = defaultdict(int)
        for comm in partition.values():
            comm_sizes[comm] += 1
        
        # Manter apenas as N maiores comunidades
        top_comms = sorted(comm_sizes, key=comm_sizes.get, reverse=True)[:num_communities]
        top_comms_set = set(top_comms)
        
        # Mapear comunidades pequenas para a maior (top_comms[0])
        comm_map = {
            comm: (comm if comm in top_comms_set else top_comms[0])
            for comm in unique_comms
        }
        
        return {node: comm_map[comm] for node, comm in partition.items()}
    
    def _print_community_stats(self, partition, bias_scores):
        """Imprime estatísticas."""
        comm_stats = defaultdict(list)
        for node, comm in partition.items():
            comm_stats[comm].append(bias_scores.get(node, 0))
        
        print("📊 Estatísticas Finais das Comunidades:")
        for comm in sorted(comm_stats.keys()):
            biases = comm_stats[comm]
            if biases:
                avg_bias = np.mean(biases)
                std_bias = np.std(biases)
                print(f"  Comunidade {comm}: {len(biases):>6,} nós, "
                      f"viés médio: {avg_bias:+.3f} (±{std_bias:.3f})")
    
    def get_communities(self):
        return self.communities