# src/config.py
import os
import torch

class Config:
    # --- Caminhos Base ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # --- Arquivo de Dados (TwiBot-20) ---
    # !! Coloque aqui o nome do arquivo JSON que você quer usar !!
    # (Baseado no README, pode ser 'train.json' ou 'support.json')
    TWIBOT20_FILE = os.path.join(PROJECT_ROOT, "TwiBot-20", "train.json") 
    
    # --- Arquivos de Cache (para TwiBot-20) ---
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data_tw20_json") # Cache p/ JSON
    GRAPH_SAVE_FILE = os.path.join(PROCESSED_DIR, "tw20_graph.pkl")
    LABELS_SAVE_FILE = os.path.join(PROCESSED_DIR, "tw20_labels.json")
    BIAS_SCORES_FILE = os.path.join(PROCESSED_DIR, "tw20_bias_scores.json")
    
    # --- Configurações do Modelo de Viés ---
    # BIAS_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    BATCH_SIZE = 32
    MAX_LENGTH = 256
    
    # --- Hardware e Paralelismo ---
    DEVICE = 0 if torch.cuda.is_available() else -1
    NUM_WORKERS = 10
    
    # --- Algoritmo ---
    ALPHA = 0.5
    RANDOM_STATE = 42

    @classmethod
    def create_dirs(cls):
        """Cria diretório de cache."""
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)
        # Garantir que o arquivo de dados existe
        if not os.path.exists(cls.TWIBOT20_FILE):
            print(f"AVISO: Arquivo de dados não encontrado: {cls.TWIBOT20_FILE}")