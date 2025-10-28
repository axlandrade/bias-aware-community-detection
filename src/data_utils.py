# src/data_utils.py
import networkx as nx
import json
import os
import time
import gc
import pickle
import numpy as np
from tqdm import tqdm
from .config import Config

class TwiBot20Loader: # <-- Nome correto da classe
    def __init__(self):
        self.config = Config()
        self.config.create_dirs()

    def _load_users_from_file(self, filepath, max_nodes):
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
        print(f"   Lendo {self.config.TWIBOT20_FILE}...")
        users_data = self._load_users_from_file(self.config.TWIBOT20_FILE, max_nodes)
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
            bot_labels[user_id] = 1 if user.get('label') == 'bot' else 0

        print(f"   Passo 2/2: Adicionando arestas...")
        edge_count = 0
        for user in tqdm(users_data, desc="   Adicionando Arestas"):
            user_id = user.get('ID')
            if not user_id: continue
            following_list = (user.get('neighbor') or {}).get('following', [])
            for target_id in following_list:
                if target_id in node_ids_in_sample:
                    G_nx.add_edge(user_id, target_id)
                    edge_count += 1
        
        print(f"   Grafo inicial: {G_nx.number_of_nodes():,} nós, {edge_count:,} arestas.")
        
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