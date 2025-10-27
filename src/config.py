# src/config.py
import os
import torch

class Config:
    # --- Configurações de Hardware e Modelo ---
    BIAS_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    BATCH_SIZE = 8192  # Podemos aumentar o batch size com mais VRAM/RAM
    MAX_LENGTH = 4096
    
    # Usar todos os cores disponíveis
    NUM_WORKERS = os.cpu_count() or 2
    
    DEVICE = 0 if torch.cuda.is_available() else -1
    DEVICE_NAME = "GPU" if torch.cuda.is_available() else "CPU"
    
    # --- Configurações do Algoritmo ---
    ALPHA = 0.5
    MAX_ITERATIONS = 20
    NUM_COMMUNITIES = 2
    
    # --- Configurações de Aleatoriedade ---
    RANDOM_STATE = 42
    
    # --- Caminhos e Nomes de Arquivos ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TWIBOT_PATH = os.path.join(PROJECT_ROOT, "data", "TwiBot22")
    
    # Diretório para salvar arquivos processados
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
    
    # Arquivos de "cache"
    GRAPH_SAVE_FILE = os.path.join(PROCESSED_DATA_DIR, "nx_full_graph.pkl")
    LABELS_SAVE_FILE = os.path.join(PROCESSED_DATA_DIR, "nx_full_labels.json")
    BIAS_SCORES_FILE = os.path.join(PROCESSED_DATA_DIR, "calculated_bias_scores.json")

    @classmethod
    def create_dirs(cls):
        """Cria diretórios necessários"""
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        print(f"Diretório de processados verificado: {cls.PROCESSED_DATA_DIR}")