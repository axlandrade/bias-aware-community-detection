import pandas as pd
import networkx as nx
import json
import os
from tqdm import tqdm
from config import Config

class TwiBotDataLoader:
    def __init__(self):
        self.config = Config()
        
    def load_and_build_graph(self, max_nodes=5000):
        """Carrega dados e constrói grafo de forma eficiente"""
        print("📊 Carregando dados do TwiBot-22...")
        
        try:
            # Carregar labels
            label_df = pd.read_csv(f"{self.config.TWIBOT_PATH}/label.csv")
            bot_labels = dict(zip(label_df['id'].astype(str), label_df['label'] == 'bot'))
            print(f"✅ {len(bot_labels)} labels carregados")
            
            # Amostrar nós para teste rápido (remova para produção)
            sampled_users = list(bot_labels.keys())[:max_nodes]
            bot_labels = {k: bot_labels[k] for k in sampled_users}
            
            # Construir grafo
            G = nx.Graph()
            G.add_nodes_from(sampled_users)
            
            # Adicionar arestas
            print("🔗 Construindo arestas...")
            chunk_size = 50000
            edge_count = 0
            
            for chunk in tqdm(pd.read_csv(f"{self.config.TWIBOT_PATH}/edge.csv", 
                                        chunksize=chunk_size)):
                # Filtrar para usuários amostrados
                chunk['source_str'] = chunk['source_id'].astype(str)
                chunk['target_str'] = chunk['target_id'].astype(str)
                
                valid_edges = chunk[
                    (chunk['source_str'].isin(sampled_users)) & 
                    (chunk['target_str'].isin(sampled_users)) &
                    (chunk['relation'].isin(['following', 'followers']))
                ]
                
                # Adicionar arestas
                for _, row in valid_edges.iterrows():
                    G.add_edge(row['source_str'], row['target_str'])
                    edge_count += 1
                    
                if edge_count > 10000:  # Limitar para teste rápido
                    break
            
            print(f"✅ Grafo construído: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
            return G, bot_labels
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            # Fallback: gerar dados sintéticos
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Gera dados sintéticos para teste"""
        print("🔄 Gerando dados sintéticos para teste...")
        
        # Criar grafo de teste
        G = nx.erdos_renyi_graph(200, 0.1, seed=self.config.RANDOM_STATE)
        
        # Converter para nós com string IDs
        mapping = {i: str(i) for i in range(200)}
        G = nx.relabel_nodes(G, mapping)
        
        # Gerar labels sintéticos
        bot_labels = {str(i): i % 5 == 0 for i in range(200)}  # 20% bots
        
        print(f"✅ Dados sintéticos: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
        return G, bot_labels