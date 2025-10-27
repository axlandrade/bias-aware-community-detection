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
import time
from .config import Config # Import relativo

# --- ADICIONAR ESTA LINHA ---
from typing import Dict, List, Optional, Any, Set
# -----------------------------

# --- Função Auxiliar (Definida no nível superior para multiprocessing) ---
def print_memory_usage(label=""):
    """Imprime o uso atual de memória RAM do processo."""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"   {label} RAM Usada: {mem_info.rss / (1024 * 1024):,.1f} MB")
    except Exception as e_mem: 
        print(f"   {label} Aviso: Não foi possível obter RAM: {e_mem}")

# --- Função do Worker (Passagem 1) - Definida no nível superior ---
def process_and_save_texts(args_tuple):
    """Lê arquivos, filtra por usuário e salva ID (int) e Texto."""
    # Desempacotar argumentos
    list_of_tweet_files, worker_num, nodes_valid_set, user_str_to_int_map, output_file_base, twibot_path = args_tuple
    output_filename = f"{output_file_base}_{worker_num}.tsv"
    count_saved = 0
    count_lines = 0
    
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            for tweet_file_path in list_of_tweet_files:
                try:
                    with open(os.path.join(twibot_path, tweet_file_path), 'r', encoding='utf-8') as infile:
                        for line in infile:
                            count_lines += 1
                            try:
                                tweet_data = json.loads(line)
                                user_id_str = tweet_data.get('author_id')
                                # Filtrar rápido
                                if user_id_str and user_id_str in nodes_valid_set:
                                    tweet_text = tweet_data.get('text', '').replace('\n', ' ').replace('\t', ' ')
                                    if tweet_text:
                                        user_id_int = user_str_to_int_map.get(user_id_str)
                                        if user_id_int is not None:
                                            writer.writerow([user_id_int, tweet_text])
                                            count_saved += 1
                            except (json.JSONDecodeError, AttributeError): 
                                continue
                            finally: 
                                if 'tweet_data' in locals(): 
                                    del tweet_data
                except Exception as e_file: 
                    print(f"   [Worker {worker_num}] Erro {os.path.basename(tweet_file_path)}: {e_file}")
        
        print(f"   [Worker {worker_num}] Concluído ({count_saved:,} textos salvos de {count_lines:,} linhas)")
        return output_filename, count_saved
        
    except Exception as e_worker: 
        print(f"   [Worker {worker_num}] ERRO FATAL: {e_worker}")
        return None, 0

# --- Classe Principal ---
class BiasCalculator:
    def __init__(self):
        self.config = Config()
        self.pipeline = None 
        self._load_id_maps() # Carregar mapas na inicialização

    def _setup_model(self):
        """Configura o modelo de análise de sentimento (LLM)"""
        if self.pipeline: 
            return
        print("🤖 Carregando modelo LLM (pode levar tempo)...")
        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.config.BIAS_MODEL,
                tokenizer=self.config.BIAS_MODEL,
                device=self.config.DEVICE,
                max_length=self.config.MAX_LENGTH,
                truncation=True
            )
            print(f"✅ Modelo LLM carregado (device: {self.config.DEVICE_NAME})")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}. Usando placeholder.")
            self.pipeline = None

    def _load_id_maps(self):
        """Carrega os mapeamentos de ID do arquivo salvo pela CÉLULA 1."""
        self.user_id_map = None
        self.user_id_rev_map = None
        id_map_file = self.config.ID_MAP_SAVE_FILE
        if os.path.exists(id_map_file):
            try:
                with open(id_map_file, 'r', encoding='utf-8') as f:
                    id_maps = json.load(f)
                self.user_id_map = id_maps.get('user_id_map') # str -> int
                self.user_id_rev_map = {int(k): v for k, v in id_maps.get('user_id_rev_map', {}).items()} # int -> str
                if not self.user_id_map or not self.user_id_rev_map:
                     print("   ⚠️ Mapeamentos de ID inválidos no arquivo.")
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar mapeamentos de ID: {e}")
            
    # Este é o método principal que o notebook deve chamar
    def get_or_calculate_bias_scores(self, G_nodes_set: Set[str]) -> Dict[str, float]:
        """
        Carrega scores de viés do arquivo ou executa o processo completo
        de duas passagens (paralelo, ordenação, LLM).
        
        Args:
            G_nodes_set (set): Conjunto de IDs de nó (string) do grafo final.
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
                with open(bias_file, 'r', encoding='utf-8') as f: 
                    bias_scores_real = json.load(f)
                if isinstance(bias_scores_real, dict) and bias_scores_real:
                    print(f"   ✅ Scores carregados para {len(bias_scores_real)} usuários.")
                    missing = [n for n in G_nodes_set if n not in bias_scores_real]
                    if missing: 
                        print(f"   ⚠️ {len(missing)} nós do grafo sem score. Atribuindo 0.0.")
                    for node in missing: 
                        bias_scores_real[node] = 0.0
                    calculation_needed = False
                else: 
                    calculation_needed = True
                    bias_scores_real = None 
            except Exception as e: 
                calculation_needed = True
                bias_scores_real = None 
        else: 
            print(f"   Arquivo final não encontrado. Calculando...")

        # --- Executar Cálculo Apenas se Necessário ---
        if calculation_needed:
            print("\n--- Iniciando cálculo completo de scores de viés (Duas Passagens) ---")
            
            if not self.user_id_map or not self.user_id_rev_map:
                print("⚠️ ERRO: Mapeamentos de ID não carregados.")
                raise RuntimeError("Mapeamentos de ID não encontrados.")
            
            # --- 3. Passagem 1: Extração Paralela de Textos ---
            intermediate_base = self.config.INTERMEDIATE_TEXT_BASE
            intermediate_combined = self.config.INTERMEDIATE_TEXT_COMBINED
            
            tweet_files = sorted([f for f in os.listdir(self.config.TWIBOT_PATH) 
                                  if f.startswith('tweet_') and f.endswith('.json')])
            
            partial_files_info = []
            processed_tweets_pass1 = 0
            
            if not tweet_files: 
                print(f"⚠️ AVISO: Nenhum arquivo tweet_*.json encontrado.")
            else:
                print(f"\n--- Passagem 1: Extraindo {len(tweet_files)} arquivos em paralelo ({self.config.NUM_WORKERS} workers) ---")
                start_pass1 = time.time()
                files_per_worker = [[] for _ in range(self.config.NUM_WORKERS)]
                for i, f in enumerate(tweet_files): 
                    files_per_worker[i % self.config.NUM_WORKERS].append(f)
                pool_args = [(files_per_worker[i], i, G_nodes_set, self.user_id_map, intermediate_base, self.config.TWIBOT_PATH) 
                             for i in range(self.config.NUM_WORKERS) if files_per_worker[i]]
                
                print("   Limpando arquivos parciais antigos...")
                for f_old in glob.glob(f"{intermediate_base}_*.tsv"): 
                    try:
                        if os.path.exists(f_old): 
                            os.remove(f_old)
                    except Exception as e_remove: 
                        print(f"      Aviso: Não foi possível remover {f_old}: {e_remove}")
                
                try:
                    with mp.Pool(processes=len(pool_args)) as pool: 
                        results = pool.map(process_and_save_texts, pool_args)
                    for filename, count in results:
                        if filename: 
                            partial_files_info.append(filename)
                            processed_tweets_pass1 += count
                except Exception as e:
                    print(f"⚠️ ERRO GERAL Passagem 1: {e}")
                    print("   Limpando arquivos parciais devido ao erro...")
                    for f_part in glob.glob(f"{intermediate_base}_*.tsv"): 
                         if os.path.exists(f_part):
                            try: 
                                os.remove(f_part)
                            except Exception as e_remove: 
                                print(f"      Aviso: Não foi possível remover {f_part}: {e_remove}")
                    raise
                
                end_pass1 = time.time()
                if not partial_files_info: 
                    print(f"\n⚠️ Nenhum arquivo parcial gerado.")
                else: 
                    print(f"\n📊 Passagem 1 concluída em {end_pass1 - start_pass1:.2f}s ({processed_tweets_pass1:,} textos).")

            del G_nodes_set
            gc.collect() 
            print_memory_usage("Após Passagem 1:")

            # --- 4. Concatenar Arquivos Parciais ---
            intermediate_file_to_sort = None
            if not partial_files_info: 
                print("\n⚠️ Pulando concatenação/ordenação.")
            else:
                print(f"\n--- Concatenando {len(partial_files_info)} arquivos -> '{intermediate_combined}' ---")
                start_concat = time.time()
                try:
                    if os.path.exists(intermediate_combined): 
                        os.remove(intermediate_combined)
                    with open(intermediate_combined, 'wb') as outfile: 
                         outfile.write("user_id_int\ttweet_text\n".encode('utf-8')) 
                         for fname in partial_files_info:
                             try:
                                 with open(fname, 'rb') as infile: 
                                     outfile.write(infile.read())
                                 os.remove(fname) 
                             except Exception as e_concat_file: 
                                 print(f"      Erro ao concatenar {fname}: {e_concat_file}")
                    end_concat = time.time()
                    print(f"\n📊 Concatenação concluída em {end_concat - start_concat:.2f}s.")
                    intermediate_file_to_sort = intermediate_combined
                except Exception as e: 
                    print(f"⚠️ ERRO concatenação: {e}")
                    raise
            
            # --- 5. Ordenar Arquivo Concatenado ---
            sort_method = "N/A"
            sorted_file = self.config.SORTED_INTERMEDIATE_TEXT_FILE
            if intermediate_file_to_sort and os.path.exists(intermediate_file_to_sort):
                print(f"\n--- Ordenando '{intermediate_file_to_sort}' -> '{sorted_file}' ---")
                start_sort = time.time()
                try: # Tentar com Pandas
                    if os.path.exists(sorted_file): 
                        os.remove(sorted_file)
                    print("   Tentando ordenar com Pandas...")
                    reader = pd.read_csv(intermediate_file_to_sort, delimiter='\t', 
                                         chunksize=self.config.SORT_READ_CHUNK_SIZE, 
                                         dtype={0: np.int64, 1: str}, low_memory=False, 
                                         quoting=csv.QUOTE_NONE, escapechar='\\')
                    all_chunks = []
                    print(f"   Lendo chunks...")
                    for i, chunk in enumerate(reader): 
                        print(f"      Chunk {i+1}")
                        all_chunks.append(chunk)
                        gc.collect()
                    if not all_chunks: 
                        print("   ⚠️ Arquivo vazio.")
                        sort_method = "Pulado (vazio)"
                    else:
                        print("   Concatenando/Ordenando...")
                        full_df_temp = pd.concat(all_chunks, ignore_index=True)
                        del all_chunks
                        gc.collect()
                        full_df_temp.sort_values(by='user_id_int', inplace=True, kind='mergesort')
                        print(f"   Escrevendo '{sorted_file}'...")
                        full_df_temp.to_csv(sorted_file, sep='\t', index=False, header=True, chunksize=1000000, quoting=csv.QUOTE_NONE, escapechar='\\')
                        del full_df_temp
                        gc.collect()
                        sort_method = "Pandas"
                except MemoryError as me: # Fallback
                    print(f"\n   ⚠️ ERRO DE MEMÓRIA com Pandas. Usando fallback: ordenação externa.")
                    sort_command = f"(head -n 1 {intermediate_file_to_sort} && tail -n +2 {intermediate_file_to_sort} | sort -t$'\\t' -k1,1n -T .) > {sorted_file}"
                    print(f"      Comando: {sort_command}")
                    print("\n      >>> PAUSADO. Execute o comando acima no terminal <<<")
                    input("      >>> Aperte Enter APÓS terminar. <<<")
                    if not os.path.exists(sorted_file) or os.path.getsize(sorted_file) < 10: 
                        raise RuntimeError("Arquivo ordenado externo falhou.")
                    sort_method = "Externo (OS sort)"
                except Exception as e: 
                    print(f"   ⚠️ ERRO na ordenação: {e}")
                    raise
                end_sort = time.time()
                print(f"\n📊 Ordenação ({sort_method}) concluída em {end_sort - start_sort:.2f}s.")
                if os.path.exists(intermediate_file_to_sort): 
                    os.remove(intermediate_file_to_sort)
            
            # --- 6. Passagem 2: Ler Ordenado, Calcular Viés (LLM Batch), Agregar ---
            if sort_method != "Pulado (vazio)":
                bias_scores_real = {}
                current_user_id_int = -1
                current_score_sum = 0.0
                current_tweet_count = 0
                print(f"\n--- Passagem 2: Lendo '{sorted_file}', usando LLM em batches ---")
                start_pass2 = time.time()
                processed_lines_pass2 = 0
                
                self._setup_model() 
                if self.pipeline is None: 
                    print("   ⚠️ Modelo LLM não carregado, usando placeholder.")

                batch_size = self.config.BATCH_SIZE
                current_batch_texts = []
                
                try:
                    buffer_size = 10 * 1024 * 1024
                    with open(sorted_file, 'r', newline='', encoding='utf-8', buffering=buffer_size) as infile:
                        reader = csv.reader(infile, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\') 
                        header = next(reader)
                        
                        for row_num, row in enumerate(tqdm(reader, desc="Processando tweets ordenados", unit="tweet")):
                            processed_lines_pass2 += 1
                            if len(row) == 2:
                                try:
                                    user_id_int_cr = int(row[0])
                                    tweet_text = row[1]
                                    if (user_id_int_cr != current_user_id_int and current_batch_texts) or len(current_batch_texts) >= batch_size:
                                        if current_batch_texts:
                                            batch_scores = self._analyze_tweets_batch_internal(current_batch_texts)
                                            current_score_sum += sum(batch_scores)
                                        current_batch_texts = []
                                    if user_id_int_cr != current_user_id_int:
                                        if current_user_id_int != -1 and current_tweet_count > 0:
                                            avg = current_score_sum / current_tweet_count
                                            user_str = self.user_id_rev_map.get(current_user_id_int)
                                            if user_str: 
                                                bias_scores_real[user_str] = avg
                                        current_user_id_int = user_id_int_cr
                                        current_score_sum = 0.0
                                        current_tweet_count = 0
                                    current_batch_texts.append(tweet_text)
                                    current_tweet_count += 1 
                                except (ValueError, KeyError): 
                                    continue
                    
                        # Processar último batch e último usuário
                        if current_batch_texts:
                            batch_scores = self._analyze_tweets_batch_internal(current_batch_texts)
                            current_score_sum += sum(batch_scores)
                        if current_user_id_int != -1 and current_tweet_count > 0:
                             avg = current_score_sum / current_tweet_count
                             user_str = self.user_id_rev_map.get(current_user_id_int)
                             if user_str: 
                                 bias_scores_real[user_str] = avg
                        
                    end_pass2 = time.time()
                    print(f"\n📊 Passagem 2 (LLM) concluída em {end_pass2 - start_pass2:.2f}s.")
                    print(f"   ↳ Scores para {len(bias_scores_real):,} usuários de {processed_lines_pass2:,} tweets.")
                except Exception as e: 
                    print(f"⚠️ ERRO GERAL Passagem 2: {e}")
                    raise
                finally:
                    if os.path.exists(sorted_file): 
                        os.remove(sorted_file)
            
            else: 
                bias_scores_real = {} 
        
        else: 
            bias_scores_real = {} 

        # --- 7. Garantir Scores e Salvar ---
        print("\n⚙️ Garantindo scores...")
        missing = 0
        try:
            # Usar o G_nodes_set original passado como argumento
            for name in G_nodes_set: 
                 if name not in bias_scores_real: 
                     bias_scores_real[name] = 0.0
                     missing += 1
            if missing > 0: 
                print(f"   ↳ {missing:,} nós sem tweets receberam score 0.0.")
        except Exception as e: 
            print(f"   ⚠️ Erro: {e}")
        
        print(f"\n💾 Salvando scores finais em '{bias_file}'...")
        try:
            with open(bias_file, 'w', encoding='utf-8') as f: 
                json.dump(bias_scores_real, f)
            print("   ✅ Scores finais salvos.")
        except Exception as e: 
            print(f"   ⚠️ Erro ao salvar: {e}")

        print("\n✅ Cálculo de viés (Completo) concluído.")
        print_memory_usage("Final:")
        gc.collect()

        # --- Fim do Bloco if calculation_needed ---
        
        if bias_scores_real is None:
            # Tentar carregar novamente se o cálculo foi pulado mas a variável é None
            if os.path.exists(bias_file):
                try:
                    with open(bias_file, 'r', encoding='utf-8') as f: 
                        bias_scores_real = json.load(f)
                except: 
                    pass 
                
        if not isinstance(bias_scores_real, dict):
            raise RuntimeError("ERRO CRÍTICO: 'bias_scores_real' não definida/carregada.")
        elif not bias_scores_real and calculation_needed: 
            print("\n⚠️ AVISO FINAL: 'bias_scores_real' vazio após cálculo.")
        
        # Retorna o dicionário (carregado ou calculado)
        return bias_scores_real  # AGORA CORRETAMENTE DENTRO DA FUNÇÃO

    # --- Funções Auxiliares da Classe ---
    
    def _analyze_tweets_batch_internal(self, tweets: List[str]) -> List[float]:
        """Função auxiliar interna para processar um batch com o pipeline."""
        if not tweets: 
            return []
        
        if self.pipeline:
            try:
                # Truncar textos longos ANTES de enviar ao pipeline
                truncated_tweets = [t[:self.config.MAX_LENGTH*4] for t in tweets] 
                
                results = self.pipeline(truncated_tweets, batch_size=self.config.BATCH_SIZE) 
                scores = []
                
                for res in results:
                    # Mapear output do cardiffnlp/twitter-roberta-base-sentiment-latest
                    if res['label'] == 'positive' or res['label'] == 'LABEL_2':
                        scores.append(res['score'])
                    elif res['label'] == 'negative' or res['label'] == 'LABEL_0':
                        scores.append(-res['score'])
                    else:  # neutral ou LABEL_1
                        scores.append(0.0)
                return scores
            except Exception as e_infer:
                 # Silenciar o erro de inferência para não poluir o log, apenas retornar neutro
                 return [0.0] * len(tweets)
        else: # Placeholder
            return [np.tanh((hash(txt) % 1000 - 500) / 250) for txt in tweets]

    def _generate_synthetic_bias(self, user_ids):
        """Gera scores de viés sintéticos para teste"""
        print("🔄 Gerando scores sintéticos (placeholder)...")
        np.random.seed(self.config.RANDOM_STATE)
        return {user_id: np.random.uniform(-1, 1) for user_id in user_ids}