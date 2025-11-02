import pandas as pd
import networkx as nx
import community as community_louvain
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm
import pickle
import os
import datetime
import sys

from src.heuristic import EnhancedLouvainWithBias

# --- Importar classes dos arquivos do repositório FACTOID ---
# (Certifique-se que esses .py estejam no mesmo diretório ou no PYTHONPATH)
try:
    from reddit_user_dataset import RedditUserDataset, EvaluatedUser
    from fake_news_detection import AnnotatedFakeNewsDetector
except ImportError:
    print("Erro: Não foi possível encontrar 'reddit_user_dataset.py' ou 'fake_news_detection.py'.")
    print("Certifique-se de que eles estão no mesmo diretório.")
    sys.exit(1)

# --- 0. Definição de Caminhos e Parâmetros ---
# (!!! AJUSTE ESSES CAMINHOS !!!)
PATH_BASE_DATASET = 'FACTOID/reddit_corpus_unbalanced_filtered.gzip'
PATH_SOCIAL_FOLDER = 'FACTOID/social_graph_data/' # Pasta com os .txt de interação
PATH_ANNOTATED_DOMAINS = 'FACTOID/fn_domains_annotated' # Ground-truth de viés
PATH_CACHE_GRAFO = 'processed_factoid/social_graph.gml'
PATH_CACHE_INPUTS = 'processed_factoid/validation_inputs.pkl' # Cache para b(v) e C_GT

# Mapeamento para converter o ground-truth em scores numéricos
BIAS_STRING_TO_SCORE = {
    "left": -1.0,
    "left-center": -0.5,
    "center": 0.0,
    "right-center": 0.5,
    "right": 1.0,
    # Adicione outros mapeamentos se necessário (ex: 'fake', 'conspiracy')
    "fake": 0.0, 
    "conspiracy": 0.0,
    "default": 0.0
}

def construir_grafo_social(dataset, social_folder):
    """
    PASSO 1: Construir o Grafo Social (G)
    Usa as funções do reddit_user_dataset.py
    """
    print("Iniciando Passo 1: Construção do Grafo Social (G)...")
    
    # 1.1 Cachear interações (lê todos os .txt da pasta social_graph_data)
    print(f"Lendo interações sociais de {social_folder}...")
    dataset.cache_social_graph(social_folder)
    
    # 1.2 Construir o grafo para um timeframe "aberto" (pegar todas as interações)
    timeframe = (datetime.date(2000, 1, 1), datetime.date(2025, 1, 1))
    
    # Cria a coluna 'social_graph' no dataframe
    dataset_com_grafo = dataset.load_social_graph_from_cache(timeframe, inplace=False)
    
    # 1.3 Converter para NetworkX
    G = nx.Graph()
    user_ids = set(dataset.data_frame['user_id'])
    
    for _, row in tqdm(dataset_com_grafo.data_frame.iterrows(), total=len(dataset_com_grafo.data_frame), desc="Construindo G"):
        user_a = row['user_id']
        G.add_node(user_a) # Adiciona o nó
        
        # 'social_graph' é um dict {user_b: interaction_count}
        for user_b, weight in row['social_graph'].items():
            if user_b in user_ids: # Garantir que o user_b está no nosso dataset
                G.add_edge(user_a, user_b, weight=weight)
                
    print(f"Grafo Social (G) construído: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")
    
    # Salvar em cache para não repetir
    nx.write_gml(G, PATH_CACHE_GRAFO)
    print(f"Grafo salvo em {PATH_CACHE_GRAFO}")
    return G

def gerar_inputs_de_vies(dataset, domains_file):
    """
    PASSO 2: Gerar C_GT (Ground-Truth) e b(v) (Vetor de Input)
    Usa AnnotatedFakeNewsDetector  para mapear posts -> domínios -> viés
    """
    print("\nIniciando Passo 2: Geração de C_GT e b(v)...")
    
    # 2.1 Carregar o detector com o arquivo de domínios anotados
    detector = AnnotatedFakeNewsDetector(domain_file_path=domains_file, label='fake') [cite: 379, 382-383]
    print(f"Domínios de viés carregados. Total: {len(detector.bias_map)}")
    
    ground_truth_map = {} # C_GT (para validação final)
    bias_vector_map = {} # b(v) (para input da heurística)
    
    # 2.2 Iterar sobre cada usuário e seus posts
    for _, row in tqdm(dataset.data_frame.iterrows(), total=len(dataset.data_frame), desc="Mapeando Viés de Usuários"):
        user_id = row['user_id']
        user_obj = EvaluatedUser(user_id, "REDDIT") [cite: 344-351]
        user_obj.own_posts = row['documents']
        
        # Esta função retorna um mapa de posts e seus domínios de viés [cite: 396-410]
        post_annotations = detector.candidate(user_obj, content_index=1) [cite: 396]
        
        user_bias_labels = [] # Coleta rótulos (ex: 'left', 'right')
        for post_id, annotations in post_annotations.items():
            for (domain, label, bias_list, factuality) in annotations:
                user_bias_labels.extend(bias_list) # O 'bias_list' é o nosso ground-truth [cite: 385]
        
        # 2.3 Determinar viés dominante (mais comum)
        if user_bias_labels:
            dominant_bias_str = max(set(user_bias_labels), key=user_bias_labels.count)
        else:
            dominant_bias_str = 'center' # Assumir neutro/centro se nenhum domínio foi postado

        # 2.4 Salvar os dois formatos
        ground_truth_map[user_id] = dominant_bias_str
        bias_vector_map[user_id] = BIAS_STRING_TO_SCORE.get(dominant_bias_str, BIAS_STRING_TO_SCORE["default"])
            
    print(f"Inputs de viés (C_GT e b(v)) gerados para {len(ground_truth_map)} usuários.")
    
    # 2.5 Salvar em cache
    with open(PATH_CACHE_INPUTS, 'wb') as f:
        pickle.dump({'gt_map': ground_truth_map, 'b_v_map': bias_vector_map}, f)
    print(f"Inputs salvos em {PATH_CACHE_INPUTS}")
    return ground_truth_map, bias_vector_map

# --------------------------------------------------------------------
# (!!! INSERIR SEU CÓDIGO AQUI !!!)
def executar_validacao(G, b_v_map, gt_map):
    """
    PASSO 4: Execução Comparativa (Louvain vs. Heurística)
    """
    print("\nIniciando Passo 4: Execução Comparativa...")
    
    # 4.1 Preparar dados para algoritmos
    # Garantir que os nós do grafo, b(v) e C_GT estejam alinhados
    nodes = list(G.nodes())
    labels_true = []
    labels_louvain = []
    labels_heuristica = []
    
    # 4.2 Executar Baseline (Louvain, alpha=0.0)
    print("Executando Baseline (Louvain Padrão, alpha=0.0)...")
    partition_louvain = community_louvain.best_partition(G, weight='weight', random_state=42)
    
    # 4.3 Executar Proposta (Enhanced Louvain, alpha=0.5)
    print("Executando Proposta (Enhanced Louvain, alpha=0.5)...")
    
    # Instanciar sua classe do arquivo heuristic.py
    enhanced_model = EnhancedLouvainWithBias(
        alpha=0.5, 
        max_iterations=100, 
        verbose=True
    ) [cite: 7-13]
    
    # Executar o método .fit() [cite: 15-37]
    enhanced_model.fit(G, b_v_map, num_communities=2)
    partition_heuristica = enhanced_model.get_communities() [cite: 157-158]

    # 4.4 Calcular Métricas de Validação
    print("\nCalculando métricas de validação...")
    
    # Alinhar os resultados das partições com o ground-truth
    # Garantir que estamos comparando apenas os nós que existem em todos os conjuntos
    valid_nodes = (
        set(nodes) & 
        set(gt_map.keys()) & 
        set(partition_louvain.keys()) & 
        set(partition_heuristica.keys())
    )
    
    if not valid_nodes:
        print("Erro: Nenhum nó em comum encontrado entre o grafo e os mapas de viés.")
        return

    for node in valid_nodes:
        labels_true.append(gt_map[node])
        labels_louvain.append(partition_louvain[node])
        labels_heuristica.append(partition_heuristica[node])
            
    ari_louvain = adjusted_rand_score(labels_true, labels_louvain)
    nmi_louvain = normalized_mutual_info_score(labels_true, labels_louvain)
    
    ari_heuristica = adjusted_rand_score(labels_true, labels_heuristica)
    nmi_heuristica = normalized_mutual_info_score(labels_true, labels_heuristica)
    
    print("\n" + "="*50)
    print("--- RESULTADOS FINAIS DA VALIDAÇÃO (Parte 1) ---")
    print(f"Baseline (Louvain, \u03B1=0.0) | ARI: {ari_louvain:.4f} | NMI: {nmi_louvain:.4f}")
    print(f"Proposta (Heurística, \u03B1=0.5) | ARI: {ari_heuristica:.4f} | NMI: {nmi_heuristica:.4f}")
    print("="*50)
    
    if ari_heuristica > ari_louvain:
        print("\n[SUCESSO] Validação bem-sucedida: A Heurística (alpha=0.5) superou o Louvain padrão.")
    else:
        print("\n[FALHA] Validação falhou: A Heurística não produziu uma partição melhor que o Louvain padrão.")

if __name__ == "__main__":
    
    # --- PASSO 1: Carregar/Construir Grafo G ---
    G = None
    if os.path.exists(PATH_CACHE_GRAFO):
        print(f"Carregando Grafo (G) do cache: {PATH_CACHE_GRAFO}")
        G = nx.read_gml(PATH_CACHE_GRAFO)
    else:
        print("Cache do grafo não encontrado. Construindo do zero...")
        # Carregar o dataset principal (contém posts e usuários)
        base_dataset = RedditUserDataset.load_from_file(PATH_BASE_DATASET, compression='gzip')
        print(f"Dataset base carregado com {len(base_dataset.data_frame)} usuários.")
        G = construir_grafo_social(base_dataset, PATH_SOCIAL_FOLDER)
    
    print(f"Grafo (G) pronto: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")

    # --- PASSO 2/3: Carregar/Gerar Inputs de Viés ---
    gt_map = None
    b_v_map = None
    if os.path.exists(PATH_CACHE_INPUTS):
        print(f"Carregando inputs de viés (C_GT, b(v)) do cache: {PATH_CACHE_INPUTS}")
        with open(PATH_CACHE_INPUTS, 'rb') as f:
            data = pickle.load(f)
            gt_map = data['gt_map']
            b_v_map = data['b_v_map']
    else:
        print("Cache de inputs de viés não encontrado. Construindo do zero...")
        # Recarregar o dataset se não foi carregado no Passo 1
        if 'base_dataset' not in locals():
            base_dataset = RedditUserDataset.load_from_file(PATH_BASE_DATASET, compression='gzip')
            print(f"Dataset base carregado com {len(base_dataset.data_frame)} usuários.")
        gt_map, b_v_map = gerar_inputs_de_vies(base_dataset, PATH_ANNOTATED_DOMAINS)

    print(f"Inputs de Viés (C_GT e b(v)) prontos.")
    
    # --- PASSO 4: Executar Validação ---
    executar_validacao(G, b_v_map, gt_map)