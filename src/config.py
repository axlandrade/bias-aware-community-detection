# src/config.py
import os
import torch

class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # --- Arquivo de Dados (TwiBot-20) ---
    TWIBOT20_FILE = os.path.join(PROJECT_ROOT, "TwiBot-20_sample.json") 
    
    # --- Arquivos de Cache (TwiBot-20) ---
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data_tw20") 
    GRAPH_SAVE_FILE = os.path.join(PROCESSED_DIR, "tw20_graph.pkl")
    LABELS_SAVE_FILE = os.path.join(PROCESSED_DIR, "tw20_labels.json")
    BIAS_SCORES_FILE = os.path.join(PROCESSED_DIR, "tw20_bias_scores.json")
    
    # --- Modelo de Viés ---
    BIAS_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    BATCH_SIZE = 64
    MAX_LENGTH = 256
    
    # --- Hardware ---
    DEVICE = 0 if torch.cuda.is_available() else -1
    NUM_WORKERS = os.cpu_count()
    
    # --- Algoritmo ---
    ALPHA = 0.5
    RANDOM_STATE = 42

    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)