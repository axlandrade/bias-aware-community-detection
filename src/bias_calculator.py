# src/bias_calculator.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from tqdm import tqdm
import json
import glob
import gc
import csv
import os
import psutil 
import pandas as pd 
import multiprocessing as mp
from collections import defaultdict # <--- Importação necessária
from tqdm import tqdm
import time
from .config import Config # Import relativo
from typing import Dict, List, Optional, Any, Set

# --- INÍCIO DA MODIFICAÇÃO (1/3) ---
# Substitui a 'lambda' para ser 'picklable' (serializável) pelo multiprocessing
def _defaultdict_factory():
    """Função auxiliar para criar o dicionário padrão."""
    return {'score_sum': 0.0, 'tweet_count': 0}
# --- FIM DA MODIFICAÇÃO ---


# --- Função Auxiliar (Definida no nível superior para multiprocessing) ---
def print_memory_usage(label=""):
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"   {label} RAM Usada: {mem_info.rss / (1024 * 1024):,.1f} MB")
    except Exception: pass

# --- Função do Worker (Passagem Única Paralela) ---
# Esta função precisa do pipeline (modelo) carregado
# Para evitar recarregar o modelo em cada worker, o inicializamos globalmente por worker
worker_pipeline = None

def init_worker_pipeline():
    """Inicializador para o pool de multiprocessing: carrega o modelo LLM uma vez por worker."""
    global worker_pipeline
    if worker_pipeline is None:
        print(f"   [Worker {os.getpid()}] Carregando modelo LLM...")
        try:
            cfg = Config() # Carregar config no worker
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
            worker_pipeline = "ERROR" # Sinalizar erro

def process_tweets_aggregate_bias(args_tuple):
    """Lê arquivos, filtra, calcula score (batch) e retorna agregado parcial."""
    global worker_pipeline
    list_of_tweet_files, worker_num, nodes_valid_set, twibot_path, batch_size = args_tuple
    
    # --- Limite (removido para execução completa) ---
    # TWEET_LIMIT_PER_WORKER = 3000 
    # ---

    if worker_pipeline is None or worker_pipeline == "ERROR":
        print(f"   [Worker {worker_num}] Modelo não carregado. Usando placeholder.")
        # Fallback para placeholder se o modelo falhou ao carregar
        local_pipeline = None
    else:
        local_pipeline = worker_pipeline

    # --- INÍCIO DA MODIFICAÇÃO (2/3) ---
    # Substituída a lambda pela função factory global
    partial_bias_data = defaultdict(_defaultdict_factory)
    # --- FIM DA MODIFICAÇÃO ---
    
    tweets_found_count = 0 
    stop_processing_flag = False
    
    print(f"   [Worker {worker_num}] Iniciando {len(list_of_tweet_files)} arquivo(s)...")
    
    current_batch_texts = []
    current_batch_users = []

    def process_batch(texts, users, pipeline):
        """Processa o batch atual com o LLM e atualiza o dict agregado."""
        if not texts: return 0
        
        scores = []
        if pipeline:
            try:
                results = pipeline(texts, batch_size=batch_size)
                for res in results:
                    if res['label'] == 'positive' or res['label'] == 'LABEL_2': scores.append(res['score'])
                    elif res['label'] == 'negative' or res['label'] == 'LABEL_0': scores.append(-res['score'])
                    else: scores.append(0.0)
            except Exception as e_infer:
                 print(f"   [Worker {worker_num}] Erro inferência: {e_infer}. Usando placeholder p/ batch.")
                 scores = [np.tanh((hash(txt) % 1000 - 500) / 250) for txt in texts] # Placeholder
        else: # Placeholder
            scores = [np.tanh((hash(txt) % 1000 - 500) / 250) for txt in texts]

        # Agregar resultados
        for user_id_str, score in zip(users, scores):
            partial_bias_data[user_id_str]['score_sum'] += score
            partial_bias_data[user_id_str]['tweet_count'] += 1
        
        return len(texts)

    # Iterar pelos arquivos
    for tweet_file_path in list_of_tweet_files:
        if stop_processing_flag: # Se o limite foi atingido, pare de abrir novos arquivos
            break
            
        try:
            with open(os.path.join(twibot_path, tweet_file_path), 'r', encoding='utf-8') as infile:
                for line in infile:
                    
                    # --- Lógica do Limite (desativada) ---
                    # if 'TWEET_LIMIT_PER_WORKER' in locals() and TWEET_LIMIT_PER_WORKER > 0:
                    #     if tweets_found_count >= TWEET_LIMIT_PER_WORKER:
                    #         stop_processing_flag = True
                    #         break 
                    # ---

                    try:
                        tweet_data = json.loads(line)
                        user_id_str = tweet_data.get('author_id')
                        if user_id_str and user_id_str in nodes_valid_set:
                            tweet_text = tweet_data.get('text', '').replace('\n', ' ').replace('\t', ' ')
                            if tweet_text:
                                current_batch_texts.append(tweet_text)
                                current_batch_users.append(user_id_str)
                                
                                tweets_found_count += 1 # Incrementar o contador aqui
                                
                                # Processar se o batch estiver cheio
                                if len(current_batch_texts) >= batch_size:
                                    process_batch(current_batch_texts, current_batch_users, local_pipeline)
                                    current_batch_texts = []
                                    current_batch_users = []
                                    
                    except (json.JSONDecodeError, AttributeError): continue
                    finally: del tweet_data
        except Exception as e_file: print(f"   [Worker {worker_num}] Erro no arquivo {os.path.basename(tweet_file_path)}: {e_file}")
    
    # Processar o último batch restante
    if current_batch_texts:
        process_batch(current_batch_texts, current_batch_users, local_pipeline)
        
    # ATUALIZAR MENSAGEM DE SAÍDA
    print(f"   [Worker {worker_num}] Concluído ({tweets_found_count:,} tweets processados)")
    return partial_bias_data # Retorna o dicionário agregado

# --- Classe Principal ---
class BiasCalculator:
    def __init__(self):
        self.config = Config()
        self.pipeline = None 
        
    def get_or_calculate_bias_scores(self, G_nodes_set: set) -> Dict[str, float]:
        """
        Carrega scores de viés do arquivo ou executa o processo completo
        de passagem única paralela.
        """
        print("\n🧠 Fase 2: Calculando/Carregando Scores de Viés...")
        print_memory_usage("Início Viés")

        bias_file = self.config.BIAS_SCORES_FILE
        
        # --- 1. Tentar Carregar Resultado Final ---
        print(f"💾 Verificando se o arquivo final '{bias_file}' já existe...")
        calculation_needed = True 
        bias_scores_real = None   
        if os.path.exists(bias_file):
            print(f"   Arquivo final encontrado! Carregando...")
            try:
                with open(bias_file, 'r', encoding='utf-8') as f: bias_scores_real = json.load(f)
                if isinstance(bias_scores_real, dict) and bias_scores_real:
                    print(f"   ✅ Scores carregados para {len(bias_scores_real)} usuários.")
                    missing = [n for n in G_nodes_set if n not in bias_scores_real]
                    if missing: print(f"   ⚠️ {len(missing)} nós do grafo sem score. Atribuindo 0.0.");
                    for node in missing: bias_scores_real[node] = 0.0
                    calculation_needed = False
                else: calculation_needed = True; bias_scores_real = None 
            except Exception as e: calculation_needed = True; bias_scores_real = None 
        else: print(f"   Arquivo final não encontrado. Calculando...")

        # --- Executar Cálculo Apenas se Necessário ---
        if calculation_needed:
            print("\n--- Iniciando cálculo de scores de viés (Single-Pass Paralelo) ---")
            
            # --- Lógica de I/O Local ---
            # (Assume que /tmp/tweet_data existe, como no notebook)
            LOCAL_TWEET_PATH = "/tmp/tweet_data" 
            
            # AVISO: Se rodar localmente no Windows, você DEVE ter criado C:\tmp\tweet_data
            if os.name == 'nt' and not os.path.exists("C:\\tmp\\tweet_data"):
                 print(f"--- ATENÇÃO WINDOWS: Lendo do caminho original {self.config.TWIBOT_PATH} ---")
                 print(f"   (Caminho C:\\tmp\\tweet_data não encontrado)")
                 LOCAL_TWEET_PATH = self.config.TWIBOT_PATH
            elif not os.path.exists(LOCAL_TWEET_PATH):
                 print(f"--- ATENÇÃO: Lendo do caminho original {self.config.TWIBOT_PATH} ---")
                 print(f"   (Caminho {LOCAL_TWEET_PATH} não encontrado)")
                 LOCAL_TWEET_PATH = self.config.TWIBOT_PATH
            else:
                 print(f"--- ATENÇÃO: Lendo tweets do disco local: {LOCAL_TWEET_PATH} ---")
            
            tweet_files = sorted([f for f in os.listdir(LOCAL_TWEET_PATH) 
                                  if f.startswith('tweet_') and f.endswith('.json')])
            # --- Fim da lógica de I/O ---
            
            if not tweet_files: 
                print(f"⚠️ AVISO: Nenhum arquivo tweet_*.json encontrado em '{LOCAL_TWEET_PATH}'.")
                bias_scores_real = {} # Vazio
            else:
                num_workers = self.config.NUM_WORKERS
                print(f"\n--- Passagem Única: Processando {len(tweet_files)} arquivos em paralelo ({num_workers} workers) ---")
                start_pass1 = time.time()
                
                # Dividir arquivos entre workers
                files_per_worker = [[] for _ in range(num_workers)]
                for i, f in enumerate(tweet_files): files_per_worker[i % num_workers].append(f)
                
                pool_args = [(files_per_worker[i], i, G_nodes_set, LOCAL_TWEET_PATH, self.config.BATCH_SIZE) 
                             for i in range(num_workers) if files_per_worker[i]]
                
                partial_results = []
                try:
                    # initializer=init_worker_pipeline carrega o LLM em cada worker
                    with mp.Pool(processes=len(pool_args), initializer=init_worker_pipeline) as pool:
                        
                        print(f"   Iniciando {len(pool_args)} workers. Acompanhe o progresso:")
                        
                        results_iterator = pool.imap(process_tweets_aggregate_bias, pool_args)
                        
                        for partial_dict in tqdm(results_iterator, total=len(pool_args), desc="   Progresso Workers"):
                            if partial_dict is not None:
                                partial_results.append(partial_dict)
                        
                except Exception as e:
                    print(f"⚠️ ERRO GERAL durante a Passagem 1 paralela: {e}"); raise
                
                end_pass1 = time.time()
                print(f"\n📊 Processamento paralelo concluído em {end_pass1 - start_pass1:.2f} segundos.")
                print_memory_usage("Após processamento paralelo:")

                # --- Agregar Resultados Parciais ---
                print("\n⚙️ Agregando resultados dos workers...")
                start_agg = time.time()
                
                # --- INÍCIO DA MODIFICAÇÃO (3/3) ---
                # Substituída a lambda pela função factory global
                user_bias_data_final = defaultdict(_defaultdict_factory)
                # --- FIM DA MODIFICAÇÃO ---
                
                total_processed_tweets = 0
                for partial_dict in partial_results:
                    for user_id, data in partial_dict.items():
                        user_bias_data_final[user_id]['score_sum'] += data['score_sum']
                        user_bias_data_final[user_id]['tweet_count'] += data['tweet_count']
                        total_processed_tweets += data['tweet_count']
                
                del partial_results; gc.collect() 
                end_agg = time.time()
                print(f"   ↳ Total de tweets processados: {total_processed_tweets:,}")
                print(f"   ✅ Agregação concluída em {end_agg - start_agg:.2f}s para {len(user_bias_data_final):,} usuários.")
                print_memory_usage("Após agregação:")

                # --- Calcular o Score Final (Média) ---
                bias_scores_real = {} 
                print("\n⚙️ Calculando scores médios de viés por usuário...")
                for user_id, data in user_bias_data_final.items():
                    bias_scores_real[user_id] = data['score_sum'] / data['tweet_count'] if data['tweet_count'] > 0 else 0.0
                del user_bias_data_final; gc.collect()

            # --- Garantir Scores e Salvar ---
            print("\n⚙️ Garantindo scores..."); missing=0
            try:
                for name in G_nodes_set: # Usar o set G_nodes_set original
                     if name not in bias_scores_real: bias_scores_real[name]=0.0; missing+=1
                if missing > 0: print(f"   ↳ {missing:,} nós sem tweets receberam score 0.0.")
            except Exception as e: print(f"   ⚠️ Erro: {e}")
            
            print(f"\n💾 Salvando scores finais em '{bias_file}'..."); 
            try:
                with open(bias_file, 'w', encoding='utf-8') as f: json.dump(bias_scores_real, f)
                print("   ✅ Scores finais salvos.");
            except Exception as e: print(f"   ⚠️ Erro ao salvar: {e}")

            print("\n✅ Cálculo de viés (Completo) concluído."); print_memory_usage("Final:")
            gc.collect()

        # --- Fim do Bloco if calculation_needed ---
        
        if bias_scores_real is None:
             if os.path.exists(bias_file):
                 try:
                     with open(bias_file, 'r', encoding='utf-8') as f: bias_scores_real = json.load(f)
                 except: pass 
                 
        if 'bias_scores_real' not in locals() or not isinstance(bias_scores_real, dict):
             raise RuntimeError("ERRO CRÍTICO: 'bias_scores_real' não definida/carregada.")
        elif not bias_scores_real and calculation_needed: 
             print("\n⚠️ AVISO FINAL: 'bias_scores_real' vazio após cálculo.")
        
        return bias_scores_real # Retorna o dicionário