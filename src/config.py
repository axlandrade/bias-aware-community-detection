import os

class Config:
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TWIBOT_PATH = os.path.join(PROJECT_ROOT, "data", "TwiBot22")
    
    # Model settings
    BIAS_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    BATCH_SIZE = 32
    MAX_LENGTH = 128
    
    # Algorithm settings
    ALPHA = 0.5
    MAX_ITERATIONS = 20
    
    # Random state
    RANDOM_STATE = 42
    
    # File names
    GRAPH_SAVE_FILE = "processed_graph.pkl"
    BIAS_SCORES_FILE = "bias_scores.json"
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(os.path.dirname(cls.GRAPH_SAVE_FILE), exist_ok=True)