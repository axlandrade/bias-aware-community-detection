# src/bias_calculator.py - VERSÃO CORRIGIDA (MODELO POLÍTICO 'POLITICS')
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
import json
import gc
import os
import multiprocessing as mp
from collections import defaultdict
import time
from .config import Config  # Import relativo
from typing import Dict, Any, Set
import traceback
import sys

# --- CORREÇÃO CRÍTICA: garante que 'src.bias_calculator' aponte para o módulo atual ---
sys.modules["src.bias_calculator"] = sys.modules[__name__]

# --- Factory ---
def _defaultdict_factory():
    return {"score_sum": 0.0, "tweet_count": 0}


# --- Variáveis globais do worker ---
worker_model = None
worker_tokenizer = None


# --- Inicializador do Worker (Carrega o MODELO 'POLITICS') ---
def init_worker_pipeline():
    print(f"--- [Worker {os.getpid()}] init_worker_pipeline CHAMADO (POLITICS Model) ---")
    global worker_model, worker_tokenizer

    if worker_model is not None and worker_model != "ERROR":
        print(f"   [Worker {os.getpid()}] Modelo já carregado.")
        return

    try:
        cfg = Config()
        model_name = "matous-volf/political-leaning-politics"
        tokenizer_name = "launch/POLITICS"

        print(f"   [Worker {os.getpid()}] Carregando tokenizer: {tokenizer_name}")
        worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        print(f"   [Worker {os.getpid()}] Carregando modelo: {model_name}")
        worker_model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if cfg.DEVICE >= 0 and torch.cuda.is_available():
            print(f"   [Worker {os.getpid()}] Movendo modelo para GPU...")
            worker_model.to("cuda")
            print(f"   [Worker {os.getpid()}] Modelo movido para GPU.")
        else:
            print(f"   [Worker {os.getpid()}] Modelo rodará na CPU.")

        worker_model.eval()
        print(f"   [Worker {os.getpid()}] Modelo carregado e em modo de avaliação.")

    except Exception as e:
        print(f"   [Worker {os.getpid()}] ########## ERRO FATAL NO INIT (POLITICS) ##########")
        print(f"   [Worker {os.getpid()}] Erro: {e}")
        print(traceback.format_exc())
        print(f"   [Worker {os.getpid()}] ############################################")
        worker_model = "ERROR"
        worker_tokenizer = "ERROR"


# --- Processamento de tweets de um único usuário ---
def process_user_tweets(user_data: Dict[str, Any]) -> Dict[str, Any]:
    global worker_model, worker_tokenizer
    user_id = user_data.get("ID", "ID_DESCONHECIDO")
    tweet_texts = user_data.get("tweet", [])

    if worker_model in (None, "ERROR") or worker_tokenizer in (None, "ERROR"):
        print(f"   [Worker {os.getpid()}] Modelo/tokenizer não carregados. Pulando usuário {user_id}.")
        return None

    if not tweet_texts:
        return None

    valid_texts = [str(t) for t in tweet_texts if str(t).strip()]
    if not valid_texts:
        return None

    cfg = Config()
    user_score_sum = 0.0
    user_tweet_count = 0

    try:
        with torch.no_grad():
            for i in range(0, len(valid_texts), cfg.BATCH_SIZE):
                batch_texts = valid_texts[i : i + cfg.BATCH_SIZE]
                inputs = worker_tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.MAX_LENGTH
                )

                if cfg.DEVICE >= 0 and torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                outputs = worker_model(**inputs)
                probs = outputs.logits.softmax(dim=-1).cpu().numpy()

                for p in probs:
                    score = (p[2] * 1.0) + (p[0] * -1.0)  # Right - Left
                    user_score_sum += score
                    user_tweet_count += 1

    except Exception as e:
        print(f"   [Worker {os.getpid()}] ERRO INFERÊNCIA (User {user_id}): {e}")
        print(traceback.format_exc())
        return None

    if user_tweet_count == 0:
        return None

    return {"id": user_id, "score_sum": user_score_sum, "tweet_count": user_tweet_count}


# --- Classe Principal ---
class BiasCalculator:
    def __init__(self):
        self.config = Config()

    def _load_users_from_file(self, filepath):
        users = []
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError:
                pass
            f.seek(0)
            for line in tqdm(f, desc="   Lendo arquivo JSONL"):
                try:
                    users.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return users

    def get_or_calculate_bias_scores(self, G_nodes_set: Set[str]) -> Dict[str, float]:
        print("\n🧠 Fase 2: Calculando/Carregando Scores de Viés (MODELO POLITICS)...")
        bias_file = self.config.BIAS_SCORES_FILE

        # --- Tentar Carregar Cache ---
        if os.path.exists(bias_file):
            print(f"   Arquivo de cache '{bias_file}' encontrado! Carregando...")
            try:
                with open(bias_file, "r") as f:
                    bias_scores_final = json.load(f)
                for node in G_nodes_set:
                    bias_scores_final.setdefault(node, 0.0)
                print(f"   ✅ Scores carregados para {len(bias_scores_final)} usuários.")
                return bias_scores_final
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar cache: {e}. Reconstruindo...")

        # --- Calcular do zero ---
        print("\n   Cache não encontrado. Calculando do zero...")
        all_users_data = self._load_users_from_file(self.config.TWIBOT20_FILE)

        users_to_process = [
            u for u in all_users_data if u.get("ID") in G_nodes_set and u.get("tweet")
        ]
        print(f"   {len(users_to_process)} usuários (com tweets) a serem processados.")

        if not users_to_process:
            return {node: 0.0 for node in G_nodes_set}

        user_bias_data_final = defaultdict(_defaultdict_factory)

        # --- Inicializar pipeline no processo principal ---
        print("\n🔧 Testando init_worker_pipeline() no processo principal...")
        init_worker_pipeline()

        if worker_model in (None, "ERROR") or worker_tokenizer in (None, "ERROR"):
            print("⚠️ init_worker_pipeline falhou — executando fallback SEQUENCIAL.")
            iterator = tqdm(users_to_process, desc="   Processando (sequencial)")
            for user in iterator:
                result = process_user_tweets(user)
                if result:
                    uid = result["id"]
                    user_bias_data_final[uid]["score_sum"] += result["score_sum"]
                    user_bias_data_final[uid]["tweet_count"] += result["tweet_count"]

        else:
            print(f"✅ init_worker_pipeline OK — iniciando multiprocessing com {self.config.NUM_WORKERS} workers...")
            try:
                context = mp.get_context("spawn")
                with context.Pool(processes=self.config.NUM_WORKERS, initializer=init_worker_pipeline) as pool:
                    for result in tqdm(pool.imap_unordered(process_user_tweets, users_to_process),
                                       total=len(users_to_process),
                                       desc="   Progresso (Usuários)"):
                        if result:
                            uid = result["id"]
                            user_bias_data_final[uid]["score_sum"] += result["score_sum"]
                            user_bias_data_final[uid]["tweet_count"] += result["tweet_count"]
            except Exception as e:
                print(f"⚠️ ERRO no multiprocessing: {e}")
                print("   ⚠️ Fallback: processando sequencialmente.")
                for user in tqdm(users_to_process, desc="   Processando (fallback seq)"):
                    result = process_user_tweets(user)
                    if result:
                        uid = result["id"]
                        user_bias_data_final[uid]["score_sum"] += result["score_sum"]
                        user_bias_data_final[uid]["tweet_count"] += result["tweet_count"]

        # --- Calcular médias ---
        print("\n⚙️ Calculando scores médios e salvando...")
        bias_scores_final = {}
        for uid, data in user_bias_data_final.items():
            if data["tweet_count"] > 0:
                bias_scores_final[uid] = data["score_sum"] / data["tweet_count"]
            else:
                bias_scores_final[uid] = 0.0

        # Adicionar nós faltantes
        for node in G_nodes_set:
            bias_scores_final.setdefault(node, 0.0)

        print(f"   ✅ Scores calculados para {len(bias_scores_final)} usuários.")

        print(f"\n💾 Salvando scores finais em '{bias_file}'...")
        os.makedirs(os.path.dirname(bias_file), exist_ok=True)
        bias_scores_final = {str(k): float(v) for k, v in bias_scores_final.items()}

        with open(bias_file, "w", encoding="utf-8") as f:
            json.dump(bias_scores_final, f, ensure_ascii=False, indent=2)
        print("   ✅ Scores finais salvos.")

        return bias_scores_final
