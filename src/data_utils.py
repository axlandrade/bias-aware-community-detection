# src/data_utils.py
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

    def load_and_build_graph(self): # Removido max_nodes
        """
        Carrega o grafo NetworkX e labels do .pkl/.json se existirem.
        Caso contrário, constrói do zero lendo os CSVs (RAM-intensivo) e salva.
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
                
                if isinstance(G_nx, nx.Graph) and G_nx.number_of_nodes() > 1:
                    print(f"   ✅ Grafo NetworkX carregado: {G_nx.number_of_nodes():,} nós, {G_nx.number_of_edges():,} arestas.")
                    print_memory_usage("Após carregar grafo")
                    return G_nx, bot_labels
                else:
                    print(f"   ⚠️ Arquivos de cache inválidos. Reconstruindo...")
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar arquivos: {e}. Reconstruindo...")
        else:
            print(f"   Arquivos não encontrados. Construindo grafo do zero (pode demorar)...")

        # --- Construir do Zero (RAM-Intensivo) ---
        try:
            # Carregar TODOS os labels
            label_df = pd.read_csv(f"{self.config.TWIBOT_PATH}/label.csv", dtype={'id': str})
            bot_labels_real = dict(zip(label_df['id'], label_df['label'] == 'bot'))
            valid_user_ids_str_set = set(bot_labels_real.keys())
            print(f"   ✅ {len(bot_labels_real):,} labels carregados.")
            del label_df; gc.collect()
            
            # Carregar TODO o edge.csv na RAM
            print("   🔗 Carregando edge.csv (RAM-intensivo)...")
            edge_file_path = f"{self.config.TWIBOT_PATH}/edge.csv"
            edge_df = pd.read_csv(edge_file_path, dtype={'source_id': str, 'target_id': str, 'relation': str})
            print_memory_usage(f"Após carregar {os.path.basename(edge_file_path)}")

            # Filtrar na memória
            print("   Filtrando arestas (seguir/seguidor)...")
            user_relations = ['following', 'followers']
            filtered_df = edge_df[
                edge_df['relation'].isin(user_relations) &
                edge_df['source_id'].isin(valid_user_ids_str_set) &
                edge_df['target_id'].isin(valid_user_ids_str_set)
            ]
            del edge_df; gc.collect() # Liberar o DataFrame gigante
            
            print("   Construindo grafo NetworkX...")
            G_nx_full = nx.from_pandas_edgelist(filtered_df, 'source_id', 'target_id')
            del filtered_df; gc.collect()

            print(f"   Grafo inicial: {G_nx_full.number_of_nodes():,} nós, {G_nx_full.number_of_edges():,} arestas.")
            
            # Pegar o maior componente conectado
            print("   Encontrando maior componente conectado...")
            largest_cc_nodes = max(nx.connected_components(G_nx_full), key=len)
            G_nx = G_nx_full.subgraph(largest_cc_nodes).copy() # Grafo final
            del G_nx_full; gc.collect()

            # Filtrar labels para o grafo final
            bot_labels = {node: bot_labels_real[node] for node in G_nx.nodes() if node in bot_labels_real}
            
            print(f"   📊 Grafo final (maior CC): {G_nx.number_of_nodes():,} nós, {G_nx.number_of_edges():,} arestas.")
            print_memory_usage("Após construir grafo")

            # --- Salvar Resultados ---
            print("\n   💾 Salvando grafo processado e labels para uso futuro...")
            try:
                with open(graph_file, 'wb') as f: pickle.dump(G_nx, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(labels_file, 'w') as f: json.dump(bot_labels, f)
                print("   ✅ Arquivos salvos.")
            except Exception as e:
                print(f"   ⚠️ Erro ao salvar arquivos: {e}")
                
            return G_nx, bot_labels
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            print("   Voltando para dados sintéticos.")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Gera dados sintéticos para teste"""
        print("🔄 Gerando dados sintéticos para teste...")
        G = nx.karate_club_graph() # Usar Karate que é mais interessante
        G = nx.relabel_nodes(G, {i: f"u{i}" for i in G.nodes()}) # Converter para IDs string
        
        bot_labels = {node: (hash(node) % 5 == 0) for node in G.nodes()} # ~20% bots
        print(f"✅ Dados sintéticos: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
        return G, bot_labels