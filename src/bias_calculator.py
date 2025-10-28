# src/bias_calculator.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from tqdm import tqdm
import json
import gc
import os
import psutil 
import multiprocessing as mp
from collections import defaultdict
import time
from .config import Config # Import relativo
from typing import Dict, List, Any, Set

# --- Factory (para corrigir erro de multiprocessing) ---
def _defaultdict_factory():
    return {'score_sum': 0.0, 'tweet_count': 0}

# --- Inicializador do Worker (Carrega o LLM) ---
worker_pipeline = None

def init_worker_pipeline():
    global worker_pipeline
    if worker_pipeline is None:
        print(f"   [Worker {os.getpid()}] Carregando modelo LLM...")
        try:
            cfg = Config()
            worker_pipeline = pipeline(
                "sentiment-analysis",
                model=cfg.BIAS_MODEL,
                tokenizer=cfg.BIAS_MODEL,
                device=cfg.DEVICE,
                max_length=cfg.MAX_LENGTH,
                truncation=True
            )
        except Exception as e:
            print(f"   [Worker {os.getpid()}] Erro ao carregar modelo: {e}")
            worker_pipeline = "ERROR"

def process_user_tweets(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processa os tweets de um ÚNICO usuário e retorna seu score de viés agregado.
    Esta é a nova função do worker.
    """
    global worker_pipeline
    if worker_pipeline is None or worker_pipeline == "ERROR":
        return None # Ignorar se o modelo não carregou

    user_id = user_data.get('ID')
    tweet_texts = user_data.get('tweet', [])
    
    if not user_id or not tweet_texts:
        return None

    # Limpar textos (alguns podem ser 'None' ou vazios)
    valid_texts = [str(t) for t in tweet_texts if str(t).strip()]
    
    if not valid_texts:
        return None

    # Processar tweets em batch (caso um usuário tenha milhares de tweets)
    cfg = Config() # Pegar BATCH_SIZE
    user_score_sum = 0.0
    user_tweet_count = 0
    
    try:
        # Passar todos os tweets do usuário de uma vez para o pipeline
        results = worker_pipeline(valid_texts, batch_size=cfg.BATCH_SIZE)
        
        for res in results:
            if res['label'] == 'positive' or res['label'] == 'LABEL_2':
                user_score_sum += res['score']
            elif res['label'] == 'negative' or res['label'] == 'LABEL_0':
                user_score_sum -= res['score']
            user_tweet_count += 1
            
    except Exception as e:
        print(f"   [Worker {os.getpid()}] Erro ao processar tweets para usuário {user_id}: {e}")
        return None

    if user_tweet_count > 0:
        return {'id': user_id, 'score_sum': user_score_sum, 'tweet_count': user_tweet_count}
    else:
        return None

# --- Classe Principal ---
class BiasCalculator:
    def __init__(self):
        self.config = Config()

    def get_or_calculate_bias_scores(self, G_nodes_set: Set[str]) -> Dict[str, float]:
        """
        Carrega scores de viés do cache ou calcula lendo o JSONL do TwiBot-20.
        """
        print("\n🧠 Fase 2: Calculando/Carregando Scores de Viés (TwiBot-20)...")
        bias_file = self.config.BIAS_SCORES_FILE

        # --- 1. Tentar Carregar do Cache ---
        if os.path.exists(bias_file):
            print(f"   Arquivo de cache '{bias_file}' encontrado! Carregando...")
            try:
                with open(bias_file, 'r') as f: bias_scores_final = json.load(f)
                # Garantir que todos os nós do grafo tenham um score
                missing = 0
                for node in G_nodes_set:
                    if node not in bias_scores_final:
                        bias_scores_final[node] = 0.0
                        missing += 1
                if missing > 0: print(f"   ⚠️ {missing} nós do grafo sem score de viés. Atribuindo 0.0.")
                print(f"   ✅ Scores carregados para {len(bias_scores_final)} usuários.")
                return bias_scores_final
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar cache: {e}. Reconstruindo...")

        # --- 2. Calcular do Zero ---
        print("\n   Cache não encontrado. Calculando do zero...")
        
        # --- Carregar Dados Brutos (com lógica robusta) ---
        all_users_data = []
        print(f"   Lendo {self.config.TWIBOT20_FILE}...")
        with open(self.config.TWIBOT20_FILE, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    print("   Formato detectado: Lista JSON.")
                    all_users_data = data
            except json.JSONDecodeError:
                print("   Formato detectado: JSON-Lines. (Lendo...)")
                f.seek(0)
                for line in tqdm(f, desc="   Lendo arquivo JSONL"):
                    try:
                        all_users_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        # --- Fim da Carga ---

        if not all_users_data:
             raise ValueError("Nenhum dado lido do arquivo fonte.")

        # Filtrar apenas aqueles que estão no nosso grafo final (G_nodes_set)
        users_to_process = []
        print(f"   Filtrando usuários que estão no grafo final...")
        for user in all_users_data:
             if user.get('ID') in G_nodes_set and user.get('tweet'):
                 users_to_process.append(user)
        
        if not users_to_process:
            print("⚠️ AVISO: Nenhum usuário com tweets encontrado no grafo. Retornando scores nulos.")
            return {node: 0.0 for node in G_nodes_set}
            
        print(f"   {len(users_to_process):,} usuários (com tweets) a serem processados.")
        
        # --- 3. Processamento Paralelo ---
        num_workers = self.config.NUM_WORKERS
        print(f"--- Iniciando processamento paralelo ({num_workers} workers) ---")
        start_pass = time.time()
        
        user_bias_data_final = defaultdict(_defaultdict_factory)

        try:
            with mp.Pool(processes=num_workers, initializer=init_worker_pipeline) as pool:
                
                results_iterator = pool.imap(process_user_tweets, users_to_process)
                
                for result in tqdm(results_iterator, total=len(users_to_process), desc="   Progresso (Usuários)"):
                    if result:
                        user_id = result['id']
                        user_bias_data_final[user_id]['score_sum'] += result['score_sum']
                        user_bias_data_final[user_id]['tweet_count'] += result['tweet_count']
                        
        except Exception as e:
            print(f"⚠️ ERRO GERAL durante o processamento paralelo: {e}"); raise
        
        end_pass = time.time()
        print(f"\n📊 Processamento paralelo concluído em {end_pass - start_pass:.2f} segundos.")

        # --- 4. Agregar e Salvar ---
        print("\n⚙️ Calculando scores médios e salvando...")
        bias_scores_final = {}
        for user_id, data in user_bias_data_final.items():
            if data['tweet_count'] > 0:
                bias_scores_final[user_id] = data['score_sum'] / data['tweet_count']
            else:
                bias_scores_final[user_id] = 0.0
        
        # Garantir que todos os nós do grafo original tenham um score
        missing = 0
        for node in G_nodes_set:
            if node not in bias_scores_final:
                bias_scores_final[node] = 0.0
                missing += 1
        if missing > 0: print(f"   ↳ {missing:,} nós sem tweets receberam score 0.0.")

        print(f"\n💾 Salvando scores finais em '{bias_file}'..."); 
        try:
            with open(bias_file, 'w', encoding='utf-8') as f: json.dump(bias_scores_final, f)
            print("   ✅ Scores finais salvos.");
        except Exception as e: print(f"   ⚠️ Erro ao salvar: {e}")
        
        return bias_scores_final