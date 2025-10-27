import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from tqdm import tqdm
import json
import os
from config import Config

class BiasCalculator:
    def __init__(self):
        self.config = Config()
        self.device = 0 if torch.cuda.is_available() else -1
        self._setup_model()
    
    def _setup_model(self):
        """Configura o modelo de análise de sentimento"""
        print("🤖 Carregando modelo de análise de viés...")
        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.config.BIAS_MODEL,
                tokenizer=self.config.BIAS_MODEL,
                device=self.device,
                max_length=self.config.MAX_LENGTH
            )
            print(f"✅ Modelo carregado (device: {'GPU' if self.device == 0 else 'CPU'})")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            self.pipeline = None
    
    def calculate_bias_from_tweets(self, user_ids):
        """Calcula scores de viés baseados em tweets"""
        if self.pipeline is None:
            print("🔄 Usando scores sintéticos (modelo não disponível)")
            return self._generate_synthetic_bias(user_ids)
        
        print("📝 Calculando scores de viés reais...")
        bias_scores = {}
        
        try:
            # Coletar tweets para cada usuário
            user_tweets = self._collect_user_tweets(user_ids)
            
            # Calcular viés em batches
            for user_id in tqdm(user_ids):
                tweets = user_tweets.get(user_id, [])
                if tweets:
                    # Usar últimos 10 tweets para eficiência
                    recent_tweets = tweets[:10]
                    bias_score = self._analyze_tweets_batch(recent_tweets)
                else:
                    bias_score = 0.0  # Neutro se não há tweets
                
                bias_scores[user_id] = bias_score
            
            print(f"✅ Scores calculados para {len(bias_scores)} usuários")
            return bias_scores
            
        except Exception as e:
            print(f"❌ Erro no cálculo de viés: {e}")
            return self._generate_synthetic_bias(user_ids)
    
    def _collect_user_tweets(self, user_ids):
        """Coleta tweets dos usuários (versão simplificada)"""
        user_tweets = {}
        tweet_files = [f for f in os.listdir(self.config.TWIBOT_PATH) 
                      if f.startswith('tweet_') and f.endswith('.json')]
        
        if not tweet_files:
            return {}
        
        print("📖 Coletando tweets...")
        for tweet_file in tqdm(tweet_files[:2]):  # Limitar para teste
            try:
                with open(f"{self.config.TWIBOT_PATH}/{tweet_file}", 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            tweet = json.loads(line)
                            user_id = tweet.get('author_id', '')
                            if user_id in user_ids:
                                text = tweet.get('text', '')
                                if text and len(text) > 10:  # Filtrar tweets muito curtos
                                    if user_id not in user_tweets:
                                        user_tweets[user_id] = []
                                    user_tweets[user_id].append(text)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                continue
                
        return user_tweets
    
    def _analyze_tweets_batch(self, tweets):
        """Analisa um batch de tweets e retorna score médio"""
        if not tweets:
            return 0.0
        
        try:
            results = self.pipeline(tweets, batch_size=self.config.BATCH_SIZE)
            scores = []
            
            for result in results:
                # Converter para escala -1 a 1
                if result['label'] == 'positive':
                    scores.append(result['score'])
                elif result['label'] == 'negative':
                    scores.append(-result['score'])
                else:  # neutral
                    scores.append(0.0)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            print(f"⚠️ Erro na análise de tweets: {e}")
            return 0.0
    
    def _generate_synthetic_bias(self, user_ids):
        """Gera scores de viés sintéticos para teste"""
        np.random.seed(self.config.RANDOM_STATE)
        return {user_id: np.random.uniform(-1, 1) for user_id in user_ids}