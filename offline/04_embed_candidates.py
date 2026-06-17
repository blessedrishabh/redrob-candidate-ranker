from sentence_transformers import SentenceTransformer
import numpy as np
import os
from tqdm import tqdm

def build_candidate_text(candidate: dict) -> str:
    parts = []
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "")
    yoe = profile.get("years_of_experience", 0)
    parts.append(f"{title} with {yoe:.1f} years of experience.")
    
    loc = profile.get("location", "")
    parts.append(f"Located in {loc}.")
    
    summary = profile.get("summary", "")
    if summary:
        parts.append(summary)
    
    for job in candidate.get("career_history", [])[:5]:
        company = job.get("company", "")
        job_title = job.get("title", "")
        duration = job.get("duration_months", 0)
        description = job.get("description", "")
        parts.append(
            f"At {company} as {job_title} ({duration} months): {description}"
        )
    
    skills_raw = candidate.get("skills", [])
    skills = [s.get("name", "") for s in skills_raw] if skills_raw and isinstance(skills_raw[0], dict) else skills_raw
    if skills:
        parts.append(f"Skills: {', '.join(skills[:20])}")
    
    for edu in candidate.get("education", [])[:2]:
        parts.append(f"Education: {edu.get('degree', '')} from {edu.get('institution', '')}")
    
    gh = candidate.get("redrob_signals", {}).get("github_activity_score", -1)
    if gh > 0:
        parts.append(f"GitHub activity score: {gh}/100")
    
    return " ".join(filter(None, parts))

def embed_all_candidates(candidates: list[dict], model, batch_size: int = 512) -> np.ndarray:
    texts = [build_candidate_text(c) for c in tqdm(candidates, desc="Building texts")]
    
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embs)
    
    embeddings = np.vstack(all_embeddings)
    print(f"Embedding matrix shape: {embeddings.shape}")
    return embeddings

if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
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
    
    jd_embedding = model.encode(JD_TEXT, normalize_embeddings=True)
    np.save("artifacts/jd_embedding.npy", jd_embedding)
    print(f"JD embedding shape: {jd_embedding.shape}")
    
    # Needs candidates loaded here
    # embeddings = embed_all_candidates(candidates, model)
    # np.save("artifacts/candidate_embeddings.npy", embeddings)
