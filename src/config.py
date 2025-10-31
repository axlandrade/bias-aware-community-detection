# src/config.py
import os
import torch

class Config:
    # --- SELETOR DE DATASET ---
    # Mude esta linha para "TWIBOT_20" ou "TWIBOT_22"
    DATASET_MODE = "TWIBOT_22" 
    # --------------------------

    # --- Caminhos Base ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # --- Configurações do Modelo de Viés (COMPARTILHADO) ---
    # Modelo que funcionou (matous-volf/political-leaning-politics)
    BIAS_MODEL_NAME = "matous-volf/political-leaning-politics"
    BIAS_TOKENIZER_NAME = "launch/POLITICS"
    
    BATCH_SIZE = 64
    MAX_LENGTH = 256
    
    # --- Hardware e Paralelismo (COMPARTILHADO) ---
    DEVICE = 0 if torch.cuda.is_available() else -1
    NUM_WORKERS = os.cpu_count()
    
    # --- Algoritmo (COMPARTILHADO) ---
    ALPHA = 0.5
    RANDOM_STATE = 42

    # --- CAMINHOS E CACHE (SELECIONADO AUTOMATICAMENTE) ---
    if DATASET_MODE == "TWIBOT_20":
        # --- Configurações TwiBot-20 (JSON Único) ---
        print("[Config] Modo TwiBot-20 (JSON) ativado.")
        DATA_DIR = os.path.join(PROJECT_ROOT, "TwiBot-20")
        # Apontar para o 'test.json' que funcionou
        DATASET_FILE_PATH = os.path.join(DATA_DIR, "dev.json") 
        TWIBOT20_FILE = os.path.join(DATA_DIR, "dev.json")

        PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data_tw20_json")
        GRAPH_SAVE_FILE = os.path.join(PROCESSED_DIR, "tw20_graph.pkl")
        LABELS_SAVE_FILE = os.path.join(PROCESSED_DIR, "tw20_labels.json")
        BIAS_SCORES_FILE = os.path.join(PROCESSED_DIR, "tw20_bias_scores.json")

    elif DATASET_MODE == "TWIBOT_22":
        # --- Configurações TwiBot-22 (CSV + Múltiplos JSONs) ---
        print("[Config] Modo TwiBot-22 (CSV/JSONs) ativado.")
        DATA_DIR = os.path.join(PROJECT_ROOT, "TwiBot-22") # Assumindo pasta raiz 'TwiBot-22'
        
        # Caminhos específicos do TwiBot-22
        GRAPH_DATA_PATH = os.path.join(DATA_DIR, "data") # Para label.csv, edge.csv
        TWEET_DATA_PATH = os.path.join(DATA_DIR, "tweet") # Para tweet_0.json etc.
        
        PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data_tw22") # Novo cache
        GRAPH_SAVE_FILE = os.path.join(PROCESSED_DIR, "tw22_graph.pkl")
        LABELS_SAVE_FILE = os.path.join(PROCESSED_DIR, "tw22_labels.json")
        BIAS_SCORES_FILE = os.path.join(PROCESSED_DIR, "tw22_bias_scores.json")
    
    else:
        raise ValueError(f"DATASET_MODE desconhecido: '{DATASET_MODE}' em config.py")

    @classmethod
    def create_dirs(cls):
        """Cria diretório de cache."""
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)