# src/data_utils.py - VERSÃO CORRIGIDA
import pandas as pd
import networkx as nx
import json
import os
import time
import gc
import pickle
import numpy as np
from tqdm import tqdm
from .config import Config # Import relativo
import psutil

# Função auxiliar de memória
def print_memory_usage(label=""):
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"   {label} RAM Usada: {mem_info.rss / (1024 * 1024):,.1f} MB")
    except Exception: pass

class TwiBotDataLoader:
    def __init__(self):
        self.config = Config()
        self.config.create_dirs()

    def load_and_build_graph(self, max_nodes):  # ✅ AGORA ACEITA max_nodes
        """
        Carrega o grafo NetworkX e labels do .pkl/.json se existirem.
        Caso contrário, constrói do zero lendo os CSVs.
        
        Args:
            max_nodes (int, optional): Número máximo de nós para carregar (útil para testes)
        """
        print("📊 Fase 1: Carregando/Construindo Grafo (NetworkX)...")
        print_memory_usage("Início Carga Grafo")
        
        graph_file = self.config.GRAPH_SAVE_FILE
        labels_file = self.config.LABELS_SAVE_FILE

        # --- Tentar Carregar Arquivos Salvos ---
        if os.path.exists(graph_file) and os.path.exists(labels_file):
            print(f"   Arquivos encontrados! Carregando grafo e labels pré-processados...")
            try:
                with open(graph_file, 'rb') as f:
                    G_nx = pickle.load(f)
                with open(labels_file, 'r', encoding='utf-8') as f:
                    bot_labels = json.load(f)
                
                # ✅ APLICAR max_nodes SE ESPECIFICADO
                if max_nodes and G_nx.number_of_nodes() > max_nodes:
                    print(f"   🔄 Amostrando {max_nodes} nós para teste rápido...")
                    sampled_nodes = list(G_nx.nodes())[:max_nodes]
                    G_nx = G_nx.subgraph(sampled_nodes).copy()
                    bot_labels = {node: bot_labels[node] for node in sampled_nodes if node in bot_labels}
                
                if isinstance(G_nx, nx.Graph) and G_nx.number_of_nodes() > 1:
                    print(f"   ✅ Grafo NetworkX carregado: {G_nx.number_of_nodes():,} nós, {G_nx.number_of_edges():,} arestas.")
                    print_memory_usage("Após carregar grafo")
                    return G_nx, bot_labels
                else:
                    print(f"   ⚠️ Arquivos de cache inválidos. Reconstruindo...")
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar arquivos: {e}. Reconstruindo...")
        else:
            print(f"   Arquivos não encontrados. Construindo grafo do zero...")

        # --- Construir do Zero ---
        try:
            # Carregar TODOS os labels
            label_df = pd.read_csv(f"{self.config.TWIBOT_PATH}/label.csv", dtype={'id': str})
            bot_labels_real = dict(zip(label_df['id'], label_df['label'] == 'bot'))
            
            # ✅ APLICAR max_nodes DURANTE A CONSTRUÇÃO
            if max_nodes:
                print(f"   🔄 Limitando para {max_nodes} nós...")
                all_users = list(bot_labels_real.keys())[:max_nodes]
                bot_labels_real = {k: bot_labels_real[k] for k in all_users}
            
            valid_user_ids_str_set = set(bot_labels_real.keys())
            print(f"   ✅ {len(bot_labels_real):,} labels carregados.")
            del label_df; gc.collect()
            
            # Carregar edge.csv em chunks para economizar memória
            print("   🔗 Carregando edge.csv em chunks...")
            edge_file_path = f"{self.config.TWIBOT_PATH}/edge.csv"
            
            G_nx_full = nx.Graph()
            G_nx_full.add_nodes_from(valid_user_ids_str_set)  # Adicionar todos os nós primeiro
            
            chunk_size = 100000
            edge_count = 0
            user_relations = ['following', 'followers']
            
            for chunk in tqdm(pd.read_csv(edge_file_path, chunksize=chunk_size, 
                                        dtype={'source_id': str, 'target_id': str, 'relation': str})):
                
                chunk_filtered = chunk[
                    chunk['relation'].isin(user_relations) &
                    chunk['source_id'].isin(valid_user_ids_str_set) &
                    chunk['target_id'].isin(valid_user_ids_str_set)
                ]
                
                # Adicionar arestas
                for _, row in chunk_filtered.iterrows():
                    G_nx_full.add_edge(row['source_id'], row['target_id'])
                    edge_count += 1
                
                # ✅ PARAR MAIS CEDO SE max_nodes ESTIVER ATIVO
                if max_nodes and edge_count > (max_nodes * 2):
                    break

            print(f"   Grafo inicial: {G_nx_full.number_of_nodes():,} nós, {G_nx_full.number_of_edges():,} arestas.")
            
            # Pegar o maior componente conectado
            print("   Encontrando maior componente conectado...")
            if G_nx_full.number_of_nodes() > 0:
                largest_cc_nodes = max(nx.connected_components(G_nx_full), key=len)
                G_nx = G_nx_full.subgraph(largest_cc_nodes).copy()
            else:
                G_nx = G_nx_full.copy()
                
            del G_nx_full; gc.collect()

            # Filtrar labels para o grafo final
            bot_labels = {node: bot_labels_real[node] for node in G_nx.nodes() if node in bot_labels_real}
            
            print(f"   📊 Grafo final (maior CC): {G_nx.number_of_nodes():,} nós, {G_nx.number_of_edges():,} arestas.")
            print_memory_usage("Após construir grafo")

            # --- Salvar Resultados ---
            print("\n   💾 Salvando grafo processado e labels para uso futuro...")
            try:
                with open(graph_file, 'wb') as f: 
                    pickle.dump(G_nx, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(labels_file, 'w') as f: 
                    json.dump(bot_labels, f)
                print("   ✅ Arquivos salvos.")
            except Exception as e:
                print(f"   ⚠️ Erro ao salvar arquivos: {e}")
                
            return G_nx, bot_labels
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            print("   Voltando para dados sintéticos.")
            return self._generate_synthetic_data(max_nodes)

    def _generate_synthetic_data(self, max_nodes=100):
        """Gera dados sintéticos para teste"""
        print("🔄 Gerando dados sintéticos para teste...")
        
        # ✅ RESPEITAR max_nodes MESMO NOS DADOS SINTÉTICOS
        num_nodes = max_nodes if max_nodes else 100
        
        # Criar grafo com estrutura de comunidade mais realista
        G = nx.planted_partition_graph(2, num_nodes//2, 0.3, 0.05, seed=42)
        G = nx.relabel_nodes(G, {i: f"u{i}" for i in G.nodes()})
        
        # Criar viés artificial nas comunidades
        bias_by_community = {0: 0.7, 1: -0.6}  # Comunidade 0 tende positiva, 1 negativa
        
        bot_labels = {}
        for node in G.nodes():
            community = int(node[1:]) % 2  # Simples baseado no ID
            bot_labels[node] = (hash(node) % 6 == 0)  # ~16% bots
        
        print(f"✅ Dados sintéticos: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
        return G, bot_labels

    def get_sample_bias_scores(self, G, polarization_strength=0.8):
        """
        Gera scores de viés sintéticos polarizados para teste rápido
        Útil quando você quer testar o algoritmo sem calcular viés real
        """
        print("🎨 Gerando scores de viés sintéticos para teste...")
        
        bias_scores = {}
        nodes = list(G.nodes())
        
        # Criar dois grupos claramente polarizados
        for i, node in enumerate(nodes):
            if i % 2 == 0:
                # Grupo "positivo" - viés principalmente positivo
                base_bias = polarization_strength
                noise = np.random.normal(0, 0.2)
                bias_scores[node] = np.clip(base_bias + noise, -1, 1)
            else:
                # Grupo "negativo" - viés principalmente negativo  
                base_bias = -polarization_strength
                noise = np.random.normal(0, 0.2)
                bias_scores[node] = np.clip(base_bias + noise, -1, 1)
        
        print(f"✅ Scores sintéticos gerados: viés médio = {np.mean(list(bias_scores.values())):.3f}")
        return bias_scores