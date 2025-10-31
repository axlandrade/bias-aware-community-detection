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

def print_memory_usage(label=""): # Função auxiliar
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"   {label} RAM Usada: {mem_info.rss / (1024 * 1024):,.1f} MB")
    except Exception: pass

# ======================================================================
#  CARREGADOR TWIBOT-20 (JSON Único)
#  (Seu código funcional, intocado)
# ======================================================================
class TwiBot20Loader:
    def __init__(self):
        self.config = Config()
        self.config.create_dirs()

    def _load_users_from_file(self, filepath, max_nodes):
        """Carrega usuários do TwiBot-20 (JSON ou JSONL)."""
        users = []
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    print("   Formato detectado: Lista JSON.")
                    if max_nodes:
                        return data[:max_nodes]
                    return data
            except json.JSONDecodeError:
                print("   Formato detectado: JSON-Lines.")
                f.seek(0) 
                for i, line in enumerate(f):
                    if max_nodes and i >= max_nodes:
                        break
                    try:
                        users.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Aviso: Ignorando linha mal formatada: {i+1}")
                return users
        if not users:
             raise ValueError("Não foi possível ler dados do arquivo JSON/JSONL.")
        return users

    def load_and_build_graph(self, max_nodes=None):
        """Carrega o grafo e labels do TwiBot-20 (JSON) ou de arquivos de cache."""
        print("📊 Fase 1: Carregando/Construindo Grafo (TwiBot-20)...")
        
        graph_file = self.config.GRAPH_SAVE_FILE
        labels_file = self.config.LABELS_SAVE_FILE

        if os.path.exists(graph_file) and os.path.exists(labels_file) and not max_nodes:
            print(f"   Arquivos de cache encontrados! Carregando...")
            try:
                with open(graph_file, 'rb') as f: G_nx = pickle.load(f)
                with open(labels_file, 'r') as f: bot_labels = json.load(f)
                print(f"   ✅ Grafo (cache): {G_nx.number_of_nodes():,} nós, {G_nx.number_of_edges():,} arestas.")
                return G_nx, bot_labels
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar cache: {e}. Reconstruindo...")
        
        print("   Cache não encontrado ou 'max_nodes' ativo. Construindo do zero...")
        
        print(f"   Lendo {self.config.DATASET_FILE_PATH}...")
        users_data = self._load_users_from_file(self.config.DATASET_FILE_PATH, max_nodes)
        if not users_data:
            raise ValueError("Nenhum dado de usuário foi carregado.")

        G_nx = nx.Graph()
        bot_labels = {}
        
        print(f"   Passo 1/2: Adicionando {len(users_data):,} nós e coletando labels...")
        node_ids_in_sample = set()
        for user in tqdm(users_data, desc="   Adicionando Nós"):
            user_id = user.get('ID')
            if not user_id: continue
            
            node_ids_in_sample.add(user_id)
            G_nx.add_node(user_id)
            bot_labels[user_id] = 1 if user.get('label') == '1' else 0 # 1=bot, 0=human

        print(f"   Passo 2/2: Adicionando arestas (da chave 'neighbor')...")
        edge_count = 0
        for user in tqdm(users_data, desc="   Adicionando Arestas"):
            user_id = user.get('ID')
            if not user_id: continue
            
            neighbor_data = user.get('neighbor') or {}
            following_list = neighbor_data.get('following', [])
            follower_list = neighbor_data.get('followers', [])
            
            for target_id in following_list:
                if target_id in node_ids_in_sample:
                    G_nx.add_edge(user_id, target_id)
                    edge_count += 1
            for target_id in follower_list:
                if target_id in node_ids_in_sample:
                    G_nx.add_edge(user_id, target_id)
                    edge_count += 1
        
        print(f"   Grafo inicial: {G_nx.number_of_nodes():,} nós, {G_nx.number_of_edges():,} arestas (bruto: {edge_count}).")
        
        print("   Encontrando maior componente conectado...")
        if G_nx.number_of_nodes() > 0:
            largest_cc_nodes = max(nx.connected_components(G_nx), key=len)
            G_nx_final = G_nx.subgraph(largest_cc_nodes).copy()
            bot_labels_final = {node: bot_labels[node] for node in G_nx_final.nodes() if node in bot_labels}
        else:
            G_nx_final = G_nx
            bot_labels_final = bot_labels
            
        del G_nx; gc.collect()

        print(f"   📊 Grafo final (maior CC): {G_nx_final.number_of_nodes():,} nós, {G_nx_final.number_of_edges():,} arestas.")

        if not max_nodes:
            print("\n   💾 Salvando grafo processado e labels para uso futuro...")
            try:
                with open(graph_file, 'wb') as f: 
                    pickle.dump(G_nx_final, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(labels_file, 'w') as f: 
                    json.dump(bot_labels_final, f)
                print("   ✅ Arquivos de cache salvos.")
            except Exception as e:
                print(f"   ⚠️ Erro ao salvar cache: {e}")
                
        return G_nx_final, bot_labels_final


# ======================================================================
#  NOVO CARREGADOR TWIBOT-22 (CSV + Múltiplos JSONs)
#  (Adicionado sem alterar o TwiBot20Loader)
# ======================================================================
class TwiBot22Loader:
    """Carrega dados do TwiBot-22 (formato label.csv, edge.csv)."""
    def __init__(self):
        self.config = Config()
        if self.config.DATASET_MODE != "TWIBOT_22":
            raise RuntimeError("TwiBot22Loader chamado, mas DATASET_MODE no config.py não é 'TWIBOT_22'")
        self.config.create_dirs()

    def load_and_build_graph(self, max_nodes=None):
        print("📊 Fase 1: Carregando/Construindo Grafo (TwiBot-22)...")
        print_memory_usage("Início Carga Grafo")

        graph_file = self.config.GRAPH_SAVE_FILE
        labels_file = self.config.LABELS_SAVE_FILE

        if os.path.exists(graph_file) and os.path.exists(labels_file) and not max_nodes:
            print(f"   Arquivos de cache encontrados! Carregando...")
            try:
                with open(graph_file, 'rb') as f: G_nx = pickle.load(f)
                with open(labels_file, 'r') as f: bot_labels = json.load(f)
                print(f"   ✅ Grafo (cache): {G_nx.number_of_nodes():,} nós, {G_nx.number_of_edges():,} arestas.")
                return G_nx, bot_labels
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar cache: {e}. Reconstruindo...")
        
        print("   Cache não encontrado ou 'max_nodes' ativo. Construindo do zero...")
        
        label_path = os.path.join(self.config.GRAPH_DATA_PATH, "label.csv")
        try:
            label_df = pd.read_csv(label_path, dtype={'id': str})
        except Exception as e:
            print(f"❌ ERRO: Não foi possível carregar '{label_path}'. Verifique o caminho.")
            raise e
            
        label_df['label'] = label_df['label'].apply(lambda x: 1 if x.lower() == 'bot' else 0)
        
        if max_nodes:
            print(f"   🔄 Limitando para {max_nodes} nós...")
            label_df = label_df.head(max_nodes)
            
        bot_labels = dict(zip(label_df['id'], label_df['label']))
        user_ids_in_graph = set(label_df['id'])
        print(f"   ✅ {len(bot_labels):,} labels carregados.")

        edge_path = os.path.join(self.config.GRAPH_DATA_PATH, "edge.csv")
        G_nx = nx.Graph()
        G_nx.add_nodes_from(user_ids_in_graph)
        
        print(f"   🔗 Carregando {edge_path} em chunks...")
        chunksize = 1_000_000
        edge_count = 0
        try:
            with pd.read_csv(edge_path, dtype={'source_id': str, 'target_id': str}, chunksize=chunksize) as reader:
                for chunk in tqdm(reader, desc="   Chunks de Arestas"):
                    chunk = chunk[chunk['source_id'].isin(user_ids_in_graph) & 
                                  chunk['target_id'].isin(user_ids_in_graph)]
                    
                    edges_to_add = list(zip(chunk['source_id'], chunk['target_id']))
                    G_nx.add_edges_from(edges_to_add)
                    edge_count += len(edges_to_add)
        except Exception as e:
            print(f"❌ ERRO: Não foi possível carregar '{edge_path}'.")
            raise e
            
        print(f"   Grafo inicial: {G_nx.number_of_nodes():,} nós, {G_nx.number_of_edges():,} arestas.")

        print("   Encontrando maior componente conectado...")
        if G_nx.number_of_nodes() > 0:
            largest_cc_nodes = max(nx.connected_components(G_nx), key=len)
            G_nx_final = G_nx.subgraph(largest_cc_nodes).copy()
            bot_labels_final = {node: bot_labels[node] for node in G_nx_final.nodes() if node in bot_labels}
        else:
            G_nx_final = G_nx
            bot_labels_final = bot_labels
        
        del G_nx, label_df; gc.collect()
        print_memory_usage("Após construir grafo")

        print(f"   📊 Grafo final (maior CC): {G_nx_final.number_of_nodes():,} nós, {G_nx_final.number_of_edges():,} arestas.")

        if not max_nodes:
            print("\n   💾 Salvando grafo processado e labels para uso futuro...")
            try:
                with open(graph_file, 'wb') as f: 
                    pickle.dump(G_nx_final, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(labels_file, 'w') as f: 
                    json.dump(bot_labels_final, f)
                print("   ✅ Arquivos de cache salvos.")
            except Exception as e:
                print(f"   ⚠️ Erro ao salvar cache: {e}")
                
        return G_nx_final, bot_labels_final