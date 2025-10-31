# src/bias_calculator.py — Versão revisada para TwiBot-20 e TwiBot-22
import os, sys, json, time, traceback, gc
import numpy as np
import torch
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, Set
from .config import Config

# 🔧 Garantir namespace correto (corrige import/multiprocess)
sys.modules["src.bias_calculator"] = sys.modules[__name__]

# --- Helpers ---
def _defaultdict_factory():
    return {"score_sum": 0.0, "tweet_count": 0}


# --- Variáveis globais dos workers ---
worker_model = None
worker_tokenizer = None


# --- Inicializador dos workers ---
def init_worker_pipeline():
    global worker_model, worker_tokenizer
    print(f"[Worker {os.getpid()}] Inicializando modelo POLITICS...")
    if worker_model not in (None, "ERROR"):
        print(f"[Worker {os.getpid()}] Modelo já carregado.")
        return

    try:
        cfg = Config()
        model_name = "matous-volf/political-leaning-politics"
        tokenizer_name = "launch/POLITICS"

        worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        worker_model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if cfg.DEVICE >= 0 and torch.cuda.is_available():
            worker_model.to("cuda")
            print(f"[Worker {os.getpid()}] Modelo movido para GPU.")
        else:
            print(f"[Worker {os.getpid()}] Modelo rodando na CPU.")

        worker_model.eval()
        print(f"[Worker {os.getpid()}] Modelo pronto e em modo de avaliação.")

    except Exception as e:
        print(f"[Worker {os.getpid()}] ❌ Erro ao carregar modelo: {e}")
        print(traceback.format_exc())
        worker_model, worker_tokenizer = "ERROR", "ERROR"


# --- Processar tweets de um único usuário (TwiBot-20) ---
def process_user_tweets(user_data: Dict[str, Any]) -> Dict[str, Any]:
    global worker_model, worker_tokenizer
    if worker_model in (None, "ERROR") or worker_tokenizer in (None, "ERROR"):
        return None

    user_id = str(user_data.get("ID", "")).strip()
    tweet_texts = [str(t).strip() for t in user_data.get("tweet", []) if str(t).strip()]
    if not tweet_texts:
        return None

    cfg = Config()
    user_score_sum, user_tweet_count = 0.0, 0

    try:
        with torch.no_grad():
            for i in range(0, len(tweet_texts), cfg.BATCH_SIZE):
                batch = tweet_texts[i:i + cfg.BATCH_SIZE]
                inputs = worker_tokenizer(batch, return_tensors="pt", padding=True,
                                          truncation=True, max_length=cfg.MAX_LENGTH)
                if cfg.DEVICE >= 0 and torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                outputs = worker_model(**inputs)
                probs = outputs.logits.softmax(dim=-1).cpu().numpy()
                for p in probs:
                    score = (p[2] * 1.0) + (p[0] * -1.0)
                    user_score_sum += score
                    user_tweet_count += 1
    except Exception as e:
        print(f"[Worker {os.getpid()}] Erro ao processar usuário {user_id}: {e}")
        return None

    if user_tweet_count == 0:
        return None
    return {"id": user_id, "score_sum": user_score_sum, "tweet_count": user_tweet_count}


# --- Processar tweets agregados (TwiBot-22) ---
def process_tweets_aggregate_bias_TW22(job_data: Dict[str, Any]) -> Dict[str, Any]:
    global worker_model, worker_tokenizer
    if worker_model in (None, "ERROR") or worker_tokenizer in (None, "ERROR"):
        return None

    tweet_file_path, nodes_valid_set = job_data["file_path"], job_data["nodes_valid_set"]
    valid_tweets = []

    # Detectar formato (lista JSON vs JSONL)
    try:
        with open(tweet_file_path, "r", encoding="utf-8") as infile:
            first_char = infile.read(1)
            infile.seek(0)
            if first_char == "[":
                tweets = json.load(infile)
            else:
                tweets = [json.loads(line) for line in infile if line.strip()]
    except Exception as e:
        print(f"⚠️ Erro ao abrir {tweet_file_path}: {e}")
        return None

    for t in tweets:
        try:
            uid_raw = str(t.get("author_id", "")).strip()
            uid = f"u{uid_raw}" if not uid_raw.startswith("u") else uid_raw

            if uid and uid in nodes_valid_set:
              txt = str(t.get("text", "")).replace("\n", " ").replace("\t", " ").strip()
              if txt:
                valid_tweets.append((uid, txt))

        except Exception:
            continue
    
    # Log informativo sobre o matching
    total_tweets = len(tweets)
    valid_count = len(valid_tweets)
    perc = (valid_count / total_tweets * 100) if total_tweets else 0
    print(f"[Worker {os.getpid()}] {os.path.basename(tweet_file_path)} → {valid_count}/{total_tweets} "
          f"tweets válidos ({perc:.2f}%) com author_id presente no grafo.")

    if not valid_tweets:
        print(f"[Worker {os.getpid()}] Nenhum tweet válido em {os.path.basename(tweet_file_path)}.")
        return None

    cfg = Config()
    user_bias_data = defaultdict(_defaultdict_factory)
    total = len(valid_tweets)
    print(f"[Worker {os.getpid()}] {os.path.basename(tweet_file_path)} → {total} tweets válidos.")

    try:
        with torch.no_grad():
            for i in range(0, total, cfg.BATCH_SIZE):
                batch = valid_tweets[i:i + cfg.BATCH_SIZE]
                texts = [t[1] for t in batch]
                inputs = worker_tokenizer(texts, return_tensors="pt", padding=True,
                                          truncation=True, max_length=cfg.MAX_LENGTH)
                if cfg.DEVICE >= 0 and torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                outputs = worker_model(**inputs)
                probs = outputs.logits.softmax(dim=-1).cpu().numpy()
                for (uid, _), p in zip(batch, probs):
                    score = (p[2] * 1.0) + (p[0] * -1.0)
                    user_bias_data[uid]["score_sum"] += score
                    user_bias_data[uid]["tweet_count"] += 1
    except Exception as e:
        print(f"[Worker {os.getpid()}] Erro em {tweet_file_path}: {e}")
        print(traceback.format_exc())

    return user_bias_data


# --- Classe principal ---
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
                f.seek(0)
                for line in tqdm(f, desc="   Lendo JSONL"):
                    try:
                        users.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return users

    def get_or_calculate_bias_scores(self, G_nodes_set: Set[str]) -> Dict[str, float]:
        cfg = self.config
        print(f"\n🧠 Calculando/Carregando Scores de Viés (Modo: {cfg.DATASET_MODE})...")
        bias_file = cfg.BIAS_SCORES_FILE

        # --- Tentar carregar cache ---
        if os.path.exists(bias_file):
            print(f"   Cache encontrado em '{bias_file}', carregando...")
            with open(bias_file, "r") as f:
                return json.load(f)

        print("\n   Cache não encontrado. Calculando do zero...")

        # --- TWIBOT-20 ---
        if cfg.DATASET_MODE == "TWIBOT_20":
            all_users = self._load_users_from_file(cfg.TWIBOT20_FILE)
            users_to_process = [u for u in all_users if u.get("ID") in G_nodes_set and u.get("tweet")]
            print(f"   {len(users_to_process)} usuários com tweets a processar.")

            user_bias_data = defaultdict(_defaultdict_factory)
            init_worker_pipeline()

            if worker_model in (None, "ERROR"):
                print("⚠️ Modelo não carregado — fallback sequencial.")
                iterator = tqdm(users_to_process, desc="Processando seq.")
                for user in iterator:
                    r = process_user_tweets(user)
                    if r:
                        uid = r["id"]
                        user_bias_data[uid]["score_sum"] += r["score_sum"]
                        user_bias_data[uid]["tweet_count"] += r["tweet_count"]
            else:
                ctx = mp.get_context("spawn")
                with ctx.Pool(cfg.NUM_WORKERS, initializer=init_worker_pipeline) as pool:
                    for r in tqdm(pool.imap_unordered(process_user_tweets, users_to_process),
                                  total=len(users_to_process), desc="Paralelo"):
                        if r:
                            uid = r["id"]
                            user_bias_data[uid]["score_sum"] += r["score_sum"]
                            user_bias_data[uid]["tweet_count"] += r["tweet_count"]

        # --- TWIBOT-22 ---
        elif cfg.DATASET_MODE == "TWIBOT_22":
            tweet_files = sorted([
                os.path.join(cfg.TWEET_DATA_PATH, f)
                for f in os.listdir(cfg.TWEET_DATA_PATH)
                if f.startswith("tweet_") and f.endswith(".json")
            ])
            print(f"   {len(tweet_files)} arquivos tweet_*.json encontrados.")

            user_bias_data = defaultdict(_defaultdict_factory)
            jobs = [{"file_path": f, "nodes_valid_set": G_nodes_set} for f in tweet_files]
            print(f"   {len(jobs)} 'jobs' (arquivos) a processar.")

            init_worker_pipeline()

            if worker_model in (None, "ERROR"):
                print("⚠️ Modelo não carregado — fallback sequencial.")
                for job in tqdm(jobs, desc="Processando seq."):
                    r = process_tweets_aggregate_bias_TW22(job)
                    if r:
                        for uid, data in r.items():
                            user_bias_data[uid]["score_sum"] += data["score_sum"]
                            user_bias_data[uid]["tweet_count"] += data["tweet_count"]
            else:
                ctx = mp.get_context("spawn")
                with ctx.Pool(cfg.NUM_WORKERS, initializer=init_worker_pipeline) as pool:
                    for r in tqdm(pool.imap_unordered(process_tweets_aggregate_bias_TW22, jobs),
                                  total=len(jobs), desc="   Progresso (Usuários/Lotes)"):
                        if r:
                            for uid, data in r.items():
                                user_bias_data[uid]["score_sum"] += data["score_sum"]
                                user_bias_data[uid]["tweet_count"] += data["tweet_count"]

        else:
            raise ValueError(f"Dataset mode '{cfg.DATASET_MODE}' não suportado.")

        # --- Calcular médias ---
        print("\n⚙️ Calculando scores médios e salvando...")
        bias_scores_final = {}
        for uid, d in user_bias_data.items():
            bias_scores_final[uid] = float(d["score_sum"] / d["tweet_count"]) if d["tweet_count"] > 0 else 0.0

        for node in G_nodes_set:
            bias_scores_final.setdefault(str(node).strip(), 0.0)

        print(f"   ✅ Scores médios calculados para {sum(1 for v in bias_scores_final.values() if v != 0.0)} usuários.")
        os.makedirs(os.path.dirname(bias_file), exist_ok=True)
        with open(bias_file, "w", encoding="utf-8") as f:
            json.dump(bias_scores_final, f, ensure_ascii=False, indent=2)
        print(f"   ✅ Scores finais salvos em '{bias_file}'.")

        return bias_scores_final
