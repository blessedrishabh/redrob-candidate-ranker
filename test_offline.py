import importlib.util
import os
import numpy as np
import sys

BASE_DIR = r"D:\projects\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\redrob_hack"
os.chdir(BASE_DIR)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod1 = load_module("parse", "offline/01_parse_and_validate.py")
mod2 = load_module("honeypots", "offline/02_detect_honeypots.py")
mod4 = load_module("embed", "offline/04_embed_candidates.py")
mod5 = load_module("matrix", "offline/05_build_feature_matrix.py")

print("Loading candidates...")
candidates = mod1.load_candidates("data/candidates.jsonl.gz")

print("Detecting honeypots...")
honeypot_results = mod2.detect_all_honeypots(candidates)

print("Loading model and creating JD embedding...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
JD_TEXT = """
Senior AI Engineer role at a Series A product company.
REQUIRED: Production deployment of embeddings-based retrieval systems.
REQUIRED: Production experience with vector databases (Pinecone, Weaviate, Qdrant, FAISS, Milvus, OpenSearch, Elasticsearch).
REQUIRED: Strong Python engineering. Code quality matters. Shipping to real users.
REQUIRED: Evaluation frameworks for ranking systems — NDCG, MRR, MAP, A/B testing.
PREFERRED: LLM fine-tuning, learning-to-rank, recommendation systems, search systems.
EXPERIENCE: 5-9 years, of which 4-5 in applied ML at product companies (NOT consulting, NOT pure research).
ROLE TYPE: Founding team engineer who writes code, ships systems, thinks about architecture.
NOT WANTED: Title-chasers, pure researchers, LangChain-only developers, consulting-only backgrounds,
CV/speech/robotics specialists, people who haven't written production code in 18 months.
LOCATION: Pune or Noida preferred. Also Hyderabad, Mumbai, Delhi NCR.
"""
os.makedirs("artifacts", exist_ok=True)
jd_emb = model.encode(JD_TEXT, normalize_embeddings=True)
np.save("artifacts/jd_embedding.npy", jd_emb)

print("Testing offline pipeline on 100 candidates to verify logic...")
candidates_subset = candidates[:100]

embs = mod4.embed_all_candidates(candidates_subset, model)

# We need to construct a subset of honeypot results
honeypot_subset_results = {c["candidate_id"]: honeypot_results[c["candidate_id"]] for c in candidates_subset}

mod5.build_feature_matrix(candidates_subset, embs, honeypot_subset_results, jd_emb)
print("Offline pipeline validation successful! No errors found.")
