# src/data_utils.py
import networkx as nx
import numpy as np
import random
from typing import Tuple, Dict

def generate_misaligned_network(
    n_nodes: int = 100, 
    structural_communities: int = 3, 
    bias_communities: int = 2,
    p_intra: float = 0.2, 
    p_inter: float = 0.03, 
    bot_ratio: float = 0.25
) -> Tuple[nx.Graph, Dict[int, float], Dict[int, bool]]:
    """
    Gera uma rede sintética com desalinhamento entre estrutura e viés.

    Cria um grafo com uma estrutura de comunidades (blocos densos) e atribui scores
    de viés de forma que as comunidades ideológicas não correspondam perfeitamente
    às comunidades estruturais.

    Args:
        n_nodes (int): Número total de nós no grafo.
        structural_communities (int): Número de comunidades baseadas na estrutura.
        bias_communities (int): Número de grupos ideológicos (clusters de viés).
        p_intra (float): Probabilidade de conexão entre nós na mesma comunidade estrutural.
        p_inter (float): Probabilidade de conexão entre nós em comunidades estruturais diferentes.
        bot_ratio (float): Proporção de nós que serão rotulados como bots.

    Returns:
        Tuple[nx.Graph, Dict[int, float], Dict[int, bool]]:
            - G: O grafo gerado.
            - bias_scores: Dicionário de scores de viés.
            - bot_labels: Dicionário de rótulos de bots.
    """
    # 1. Gerar comunidades estruturais usando um modelo de blocos estocásticos
    nodes_per_struct = n_nodes // structural_communities
    structural_assignment = {
        node: min(node // nodes_per_struct, structural_communities - 1)
        for node in range(n_nodes)
    }
    
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            prob = p_intra if structural_assignment[i] == structural_assignment[j] else p_inter
            if random.random() < prob:
                G.add_edge(i, j)

    # 2. Gerar scores de viés com base em uma partição ideológica diferente
    bias_scores = {}
    nodes_per_bias = n_nodes // bias_communities
    for node in range(n_nodes):
        bias_group = min(node // nodes_per_bias, bias_communities - 1)
        base_bias = 0.8 if bias_group == 0 else -0.8
        noise = np.random.normal(0, 0.15)
        bias_scores[node] = np.clip(base_bias + noise, -1, 1)

    # 3. Gerar rótulos de bots, priorizando nós com viés extremo e alto grau
    max_degree = max(dict(G.degree()).values()) if G.number_of_edges() > 0 else 1
    bot_scores = {
        node: 0.7 * abs(bias_scores[node]) + 0.3 * (G.degree(node) / max_degree)
        for node in range(n_nodes)
    }

    n_bots = int(n_nodes * bot_ratio)
    bot_candidates = sorted(bot_scores, key=bot_scores.get, reverse=True)
    actual_bots = set(bot_candidates[:n_bots])
    bot_labels = {node: node in actual_bots for node in range(n_nodes)}

    return G, bias_scores, bot_labels


def generate_twibot_like_network(n_users=500, bot_ratio=0.14, avg_degree=30, polarization=0.7):
    """
    Gera rede sintética com características do TwiBot-22
    """

    print(f"🤖 Gerando rede estilo TwiBot-22...")
    print(f"   Usuários: {n_users:,}")
    print(f"   Bots esperados: {int(n_users * bot_ratio):,} ({bot_ratio:.1%})")

    # Rede scale-free (Barabási-Albert)
    m = avg_degree // 2
    G = nx.barabasi_albert_graph(n_users, m, seed=42)

    print(f"   Arestas: {G.number_of_edges():,}")
    print(f"   Grau médio: {2*G.number_of_edges()/G.number_of_nodes():.1f}")

    # Identificar hubs
    degrees = dict(G.degree())
    degree_threshold = np.percentile(list(degrees.values()), 90)
    hubs = [node for node, deg in degrees.items() if deg >= degree_threshold]

    # Gerar viés (distribuição bimodal - polarização)
    bias_scores = {}
    for node in G.nodes():
        base_bias = -polarization if random.random() < 0.5 else polarization
        noise = np.random.normal(0, 0.2)
        bias_scores[node] = np.clip(base_bias + noise, -1, 1)

    # Gerar bots (concentrados em extremos e hubs)
    bot_scores = {}
    for node in G.nodes():
        extremism = abs(bias_scores[node])
        is_hub = 1.0 if node in hubs else 0.3
        bot_scores[node] = 0.6 * extremism + 0.4 * is_hub

    n_bots = int(n_users * bot_ratio)
    bot_candidates = sorted(range(n_users), key=lambda x: bot_scores[x], reverse=True)

    # 70% coordenados + 30% aleatórios
    actual_bots = set(bot_candidates[:int(n_bots * 0.7)])
    remaining = [n for n in range(n_users) if n not in actual_bots]
    actual_bots.update(random.sample(remaining, n_bots - len(actual_bots)))

    bot_labels = {node: node in actual_bots for node in G.nodes()}

    # Estatísticas
    bot_biases = [bias_scores[n] for n in G.nodes() if bot_labels[n]]
    human_biases = [bias_scores[n] for n in G.nodes() if not bot_labels[n]]

    print(f"\n📊 Estatísticas:")
    print(f"   Bots: {sum(bot_labels.values())} ({sum(bot_labels.values())/n_users:.1%})")
    print(f"   Viés médio (bots): {np.mean(bot_biases):.3f} ± {np.std(bot_biases):.3f}")
    print(f"   Viés médio (humanos): {np.mean(human_biases):.3f} ± {np.std(human_biases):.3f}")

    left = sum(1 for b in bias_scores.values() if b < -0.3)
    right = sum(1 for b in bias_scores.values() if b > 0.3)
    print(f"   Esquerda: {left} ({left/n_users:.1%})")
    print(f"   Direita: {right} ({right/n_users:.1%})")
    print(f"   Centro: {n_users-left-right} ({(n_users-left-right)/n_users:.1%})")

    return G, bias_scores, bot_labels