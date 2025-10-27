# src/config.py - VERSÃO CORRIGIDA
import os
import torch

class Config:
    # Paths - CORRIGIDO para Colab
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TWIBOT_PATH = os.path.join(PROJECT_ROOT, "data", "TwiBot22")
    
    # 🔥 CORREÇÃO: Definir caminhos absolutos para arquivos
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data")
    GRAPH_SAVE_FILE = os.path.join(PROCESSED_DIR, "processed_graph.pkl")
    LABELS_SAVE_FILE = os.path.join(PROCESSED_DIR, "processed_labels.json")
    BIAS_SCORES_FILE = os.path.join(PROCESSED_DIR, "bias_scores.json")
    
    # Model settings
    BIAS_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    BATCH_SIZE = 64
    MAX_LENGTH = 256
    
    # Device configuration
    DEVICE = 0 if torch.cuda.is_available() else -1
    
    # Parallel processing
    NUM_WORKERS = 2
    
    # Algorithm settings
    ALPHA = 0.5
    MAX_ITERATIONS = 20
    RANDOM_STATE = 42
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories - CORRIGIDO"""
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)
        print(f"✅ Diretório criado: {cls.PROCESSED_DIR}")