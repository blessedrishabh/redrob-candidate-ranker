import importlib.util
import os
import numpy as np
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

print("--- 1. Loading candidates... ---")
candidates = mod1.load_candidates("data/candidates.jsonl")

print("\n--- 2. Detecting honeypots... ---")
honeypot_results = mod2.detect_all_honeypots(candidates)
# Save honeypot IDs required by rank.py
os.makedirs("artifacts", exist_ok=True)
honeypot_ids = [cid for cid, res in honeypot_results.items() if res["is_honeypot"]]
with open("artifacts/honeypot_ids.txt", "w") as f:
    f.write("\n".join(honeypot_ids))

print("\n--- 3. Loading model and creating JD embedding... ---")
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
jd_emb = model.encode(JD_TEXT, normalize_embeddings=True)
np.save("artifacts/jd_embedding.npy", jd_emb)

print("\n--- 4. Generating candidate embeddings (100k) ---")
print("NOTE: This will take ~8 hours on CPU, or ~10 minutes on a T4 GPU.")
embs = mod4.embed_all_candidates(candidates, model)

print("\n--- 5. Building feature matrix... ---")
mod5.build_feature_matrix(candidates, embs, honeypot_results, jd_emb)

print("\n[SUCCESS] Full offline pipeline completed! You can now run rank.py")
