# src/evaluation.py
import networkx as nx  # <<< ADICIONAR
import numpy as np     # <<< ADICIONAR
import community.community_louvain as community_louvain # <<< ADICIONAR

from typing import Dict, Optional
from collections import defaultdict

class ComprehensiveEvaluator:
    """Agrupa métodos estáticos para avaliar a qualidade das partições de comunidades."""

    @staticmethod
    def evaluate_communities(
        G: nx.Graph, 
        partition: Dict[int, int],
        bias_scores: Dict[int, float],
        bot_labels: Optional[Dict[int, bool]] = None
    ) -> Dict[str, float]:
        """
        Calcula um conjunto de métricas para avaliar uma partição de comunidade.

        Args:
            G (nx.Graph): O grafo original.
            partition (Dict[int, int]): Dicionário {nó: id_comunidade}.
            bias_scores (Dict[int, float]): Dicionário {nó: score_de_viés}.
            bot_labels (Optional[Dict[int, bool]]): Dicionário {nó: is_bot}.

        Returns:
            Dict[str, float]: Dicionário com os nomes e valores das métricas.
        """
        metrics = {}

        # Métrica Estrutural: Modularidade de Newman-Girvan
        # Mede a força da divisão do grafo em comunidades. Valores mais altos são melhores.
        metrics['modularity'] = community_louvain.modularity(partition, G)

        # Métricas de Viés
        community_biases = defaultdict(list)
        for node, comm in partition.items():
            community_biases[comm].append(bias_scores[node])

        # Pureza de Viés: Mede a homogeneidade ideológica dentro das comunidades.
        # Baseado no inverso do desvio padrão médio intra-comunidade. Valores mais altos são melhores.
        within_comm_std = [np.std(biases) for biases in community_biases.values() if len(biases) > 1]
        avg_within_std = np.mean(within_comm_std) if within_comm_std else 0
        metrics['bias_purity'] = 1 / (1 + avg_within_std)

        # Separação de Viés: Mede o quão ideologicamente distintas as comunidades são entre si.
        # Baseado no desvio padrão dos vieses médios das comunidades. Valores mais altos são melhores.
        avg_biases = [np.mean(biases) for biases in community_biases.values()]
        metrics['bias_separation'] = np.std(avg_biases) if len(avg_biases) > 1 else 0

        # Métrica de Bots (se disponível)
        if bot_labels is not None:
            community_bots = defaultdict(list)
            for node, comm in partition.items():
                community_bots[comm].append(bot_labels[node])

            # Concentração de Bots: Mede a proporção máxima de bots em qualquer comunidade.
            # Útil para verificar se o método agrupa contas maliciosas.
            bot_concentrations = [sum(bots) / len(bots) for bots in community_bots.values() if bots]
            metrics['bot_concentration_max'] = max(bot_concentrations) if bot_concentrations else 0
            metrics['bot_concentration_min'] = min(bot_concentrations) if bot_concentrations else 0

        metrics['num_communities'] = len(set(partition.values()))

        return metrics