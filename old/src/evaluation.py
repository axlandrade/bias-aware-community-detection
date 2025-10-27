import networkx as nx
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd

class ComprehensiveEvaluator:
    @staticmethod
    def evaluate_communities(G, partition, bias_scores, bot_labels=None):
        """Avalia qualidade das comunidades detectadas"""
        metrics = {}
        
        # Métricas estruturais
        metrics['modularity'] = nx.algorithms.community.modularity(G, ComprehensiveEvaluator._get_communities_set(partition))
        metrics['num_communities'] = len(set(partition.values()))
        
        # Métricas de viés
        bias_metrics = ComprehensiveEvaluator._calculate_bias_metrics(partition, bias_scores)
        metrics.update(bias_metrics)
        
        # Métricas de bots (se disponível)
        if bot_labels:
            bot_metrics = ComprehensiveEvaluator._calculate_bot_metrics(partition, bot_labels)
            metrics.update(bot_metrics)
        
        return metrics
    
    @staticmethod
    def _get_communities_set(partition):
        """Converte partição para formato do NetworkX"""
        communities = {}
        for node, comm in partition.items():
            if comm not in communities:
                communities[comm] = set()
            communities[comm].add(node)
        return list(communities.values())
    
    @staticmethod
    def _calculate_bias_metrics(partition, bias_scores):
        """Calcula métricas de qualidade do viés"""
        metrics = {}
        
        # Agrupar viés por comunidade
        comm_biases = {}
        for node, comm in partition.items():
            if comm not in comm_biases:
                comm_biases[comm] = []
            comm_biases[comm].append(bias_scores.get(node, 0))
        
        # Pureza intra-comunidade (baixa variância é bom)
        intra_variances = [np.var(biases) for biases in comm_biases.values() if len(biases) > 1]
        metrics['bias_purity'] = 1 - (np.mean(intra_variances) if intra_variances else 0)
        
        # Separação inter-comunidade (alta variância entre médias é bom)
        comm_means = [np.mean(biases) for biases in comm_biases.values() if biases]
        metrics['bias_separation'] = np.var(comm_means) if len(comm_means) > 1 else 0
        
        return metrics
    
    @staticmethod
    def _calculate_bot_metrics(partition, bot_labels):
        """Calcula métricas relacionadas a bots"""
        metrics = {}
        
        comm_bots = {}
        for node, comm in partition.items():
            if comm not in comm_bots:
                comm_bots[comm] = []
            if node in bot_labels:
                comm_bots[comm].append(bot_labels[node])
        
        # Concentração máxima de bots
        bot_concentrations = []
        for comm, bots in comm_bots.items():
            if bots:
                bot_ratio = sum(bots) / len(bots)
                bot_concentrations.append(bot_ratio)
        
        metrics['bot_concentration_max'] = max(bot_concentrations) if bot_concentrations else 0
        metrics['bot_concentration_avg'] = np.mean(bot_concentrations) if bot_concentrations else 0
        
        return metrics
    
    @staticmethod
    def print_comparison(metrics1, metrics2, name1="Método 1", name2="Método 2"):
        """Imprime comparação entre dois métodos"""
        print(f"\n📈 COMPARAÇÃO: {name1} vs {name2}")
        print("-" * 50)
        
        for metric in ['modularity', 'bias_purity', 'bias_separation', 'bot_concentration_max']:
            if metric in metrics1 and metric in metrics2:
                val1 = metrics1[metric]
                val2 = metrics2[metric]
                
                if val2 != 0:
                    improvement = ((val1 / val2) - 1) * 100
                else:
                    improvement = 0
                
                print(f"{metric:>20}: {val1:7.4f} vs {val2:7.4f} ({improvement:+.1f}%)")