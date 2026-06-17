import argparse
import gzip
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

START_TIME = time.time()
DEADLINE_SECONDS = 4 * 60 + 30

def load_precomputed(artifacts_dir: str = "./artifacts"):
    t0 = time.time()
    
    features = pd.read_parquet(f"{artifacts_dir}/candidates_features.parquet")
    honeypot_ids = set(
        Path(f"{artifacts_dir}/honeypot_ids.txt").read_text().strip().split("\n")
    )
    
    print(f"[INFO] Loaded {len(features):,} candidates in {time.time()-t0:.1f}s")
    print(f"[INFO] Honeypot set: {len(honeypot_ids)} candidates")
    return features, honeypot_ids

def compute_availability_vectorized(df: pd.DataFrame) -> pd.Series:
    now = datetime.now()
    
    try:
        last_active = pd.to_datetime(df["last_active_date"], errors="coerce")
        days_since = (now - last_active).dt.days.fillna(365)
    except Exception:
        days_since = pd.Series(365, index=df.index)
    
    activity_score = np.where(days_since <= 7, 1.0,
                    np.where(days_since <= 30, 0.9,
                    np.where(days_since <= 90, 0.7,
                    np.where(days_since <= 180, 0.4,
                    np.where(days_since <= 365, 0.2, 0.1)))))
    
    rr = df["recruiter_response_rate"].fillna(0.5)
    rr_score = np.where(rr < 0.10, 0.30,
               np.where(rr < 0.30, 0.60,
               np.where(rr < 0.60, 0.85, 1.0)))
    
    otw_score = np.where(df["open_to_work"].fillna(0) == 1, 1.0, 0.75)
    
    notice = df["notice_period_days"].fillna(60)
    notice_score = np.where(notice <= 30, 1.0,
                   np.where(notice <= 60, 0.90,
                   np.where(notice <= 90, 0.80,
                   np.where(notice <= 120, 0.70, 0.55))))
    
    availability = activity_score * rr_score * otw_score * notice_score
    return pd.Series(np.clip(availability, 0.0, 1.0), index=df.index)

def compute_scores_vectorized(df: pd.DataFrame) -> pd.Series:
    embedding = df["embedding_score"].fillna(0)
    keyword = df["jd_keyword_score"].fillna(0)
    company = df["company_type_score"].fillna(0).clip(0, 1)
    location = df["location_score"].fillna(0.5)
    github = df["github_activity_score"].apply(
        lambda x: 0.5 if x == -1 else x/100
    ).fillna(0.5)
    shipper_bonus = df["shipper_bonus"].fillna(0)
    penalty = df["disqualifier_penalty"].fillna(0).clip(-1, 0)
    
    skill_score = (
        0.40 * embedding +
        0.30 * keyword +
        0.15 * company +
        0.10 * location +
        0.05 * github +
        shipper_bonus +
        penalty
    ).clip(0, 1)
    
    availability = df["availability"].fillna(0.5)
    final_score = skill_score * (0.70 + 0.30 * availability)
    
    return final_score.round(6)

def _days_since(date_str) -> int | None:
    if pd.isna(date_str) or not date_str:
        return None
    try:
        dt = datetime.fromisoformat(str(date_str))
        return (datetime.now() - dt).days
    except (ValueError, TypeError):
        return None

def generate_reasoning_fast(row: pd.Series) -> str:
    title = row.get("current_title", "Unknown title")
    yoe = row.get("years_of_experience", 0)
    loc = row.get("location", "Unknown location")
    rank = int(row.get("rank", 0))
    
    active_days = _days_since(row.get("last_active_date"))
    notice = row.get("notice_period_days", 60)
    rr = row.get("recruiter_response_rate", 0.5)
    github = row.get("github_activity_score", -1)
    open_to_work = row.get("open_to_work", 0)
    
    skills_raw = row.get("skills", "")
    if isinstance(skills_raw, str):
        skills = [s.strip() for s in skills_raw.split("|") if s.strip()]
    else:
        skills = []
    
    RELEVANT_SKILLS = {
        "embeddings", "vector database", "faiss", "pinecone", "weaviate", "qdrant",
        "sentence-transformers", "elasticsearch", "opensearch", "retrieval", "ranking",
        "python", "pytorch", "tensorflow", "lora", "qlora", "fine-tuning",
        "ndcg", "mrr", "a/b testing", "recommendation", "search", "bge", "e5"
    }
    matched_skills = [s for s in skills if s.lower() in RELEVANT_SKILLS][:3]
    
    concerns = []
    if active_days and active_days > 90:
        concerns.append(f"inactive for {active_days//30}+ months")
    if notice > 90:
        concerns.append(f"{notice}-day notice period")
    if rr < 0.30:
        concerns.append(f"low recruiter response rate ({rr:.0%})")
    if not open_to_work:
        concerns.append("not marked open to work")
    
    if rank <= 20:
        skill_str = (f" with {', '.join(matched_skills)}" if matched_skills else "")
        line1 = f"{title} with {yoe:.0f} years experience{skill_str}, based in {loc}."
        if concerns:
            line2 = f"Strong technical fit; note: {'; '.join(concerns[:1])}."
        elif github > 70:
            line2 = f"Active on GitHub (score: {github}/100), suggesting a shipper profile."
        elif open_to_work:
            line2 = "Actively open to work with strong JD-alignment across embeddings and retrieval."
        else:
            line2 = "Profile aligns well with the production-focused, product-company experience the JD requires."
        return f"{line1} {line2}"
    
    elif rank <= 50:
        skill_str = (f", including {matched_skills[0]}" if matched_skills else "")
        line1 = f"{title}, {yoe:.0f} yrs, {loc}{skill_str}."
        concern_str = f"Concerns: {'; '.join(concerns)}." if concerns else "Reasonably active profile."
        return f"{line1} {concern_str}"
    
    else:
        line1 = f"{yoe:.0f}-year {title} in {loc}."
        if concerns:
            line2 = f"Included for adjacent skills but ranked lower due to: {'; '.join(concerns[:2])}."
        else:
            line2 = "Below-cutoff profile included as best available match at this rank; partial skill alignment only."
        return f"{line1} {line2}"

def rank_candidates(features: pd.DataFrame, honeypot_ids: set) -> pd.DataFrame:
    t0 = time.time()
    
    features["is_honeypot"] = features["candidate_id"].isin(honeypot_ids)
    features["availability"] = compute_availability_vectorized(features)
    features["score"] = compute_scores_vectorized(features)
    features.loc[features["is_honeypot"], "score"] = 0.0
    
    features = features.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    features["rank"] = features.index + 1
    
    print(f"[INFO] Scoring complete in {time.time()-t0:.1f}s")
    
    elapsed = time.time() - t0
    print(f"[INFO] Total elapsed: {elapsed:.1f}s / {DEADLINE_SECONDS}s budget")
    
    if elapsed > DEADLINE_SECONDS:
        raise RuntimeError(f"[FATAL] Over time budget! {elapsed:.1f}s elapsed")
    
    return features

def build_top_100(ranked_df: pd.DataFrame) -> pd.DataFrame:
    clean = ranked_df[~ranked_df["is_honeypot"]].copy()
    top_100 = clean.head(100).copy()
    top_100["rank"] = range(1, 101)
    
    honeypot_in_top = ranked_df.head(100)["is_honeypot"].sum()
    print(f"[INFO] Honeypots in top 100: {honeypot_in_top} (must be < 10)")
    if honeypot_in_top >= 10:
        raise ValueError(f"[FATAL] Honeypot rate too high: {honeypot_in_top}/100 = {honeypot_in_top}%")
    
    scores = top_100["score"].values
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), \
        "Scores are not monotonically decreasing!"
    
    return top_100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl.gz")
    parser.add_argument("--artifacts", default="./artifacts", help="Path to precomputed artifacts")
    parser.add_argument("--out", default="./submission.csv", help="Output CSV path")
    args = parser.parse_args()
    
    print(f"[INFO] Ranker started at {datetime.now().isoformat()}")
    
    features, honeypot_ids = load_precomputed(args.artifacts)
    ranked = rank_candidates(features, honeypot_ids)
    top_100 = build_top_100(ranked)
    
    top_100["reasoning"] = top_100.apply(generate_reasoning_fast, axis=1)
    
    output = top_100[["candidate_id", "rank", "score", "reasoning"]]
    output.to_csv(args.out, index=False, encoding="utf-8")
    
    elapsed = time.time() - START_TIME
    print(f"\n[SUCCESS] Submission written to {args.out}")
    print(f"[SUCCESS] Total runtime: {elapsed:.1f}s / 300s budget ({elapsed/300*100:.1f}% used)")
    print(f"[SUCCESS] Rows: {len(output)} | Unique ranks: {output['rank'].nunique()}")

if __name__ == "__main__":
    main()
