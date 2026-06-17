import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import importlib.util
import os

# Load 04_embed_candidates dynamically
spec_embed = importlib.util.spec_from_file_location("embed_candidates", "offline/04_embed_candidates.py")
embed_candidates = importlib.util.module_from_spec(spec_embed)
spec_embed.loader.exec_module(embed_candidates)
build_candidate_text = embed_candidates.build_candidate_text

# Load 03_jd_scoring dynamically
spec_jd = importlib.util.spec_from_file_location("jd_scoring", "offline/03_jd_scoring.py")
jd_scoring = importlib.util.module_from_spec(spec_jd)
spec_jd.loader.exec_module(jd_scoring)

JD_ANALYSIS = jd_scoring.JD_ANALYSIS
LOCATION_WEIGHTS = jd_scoring.LOCATION_WEIGHTS
score_company_profile = jd_scoring.score_company_profile

def compute_jd_keyword_score(c):
    score = 0.0
    skills_raw = c.get("skills", [])
    skills_list = [s.get("name", "") for s in skills_raw] if skills_raw and isinstance(skills_raw[0], dict) else skills_raw
    profile = c.get("profile", {})
    full_text = " ".join([str(s).lower() for s in skills_list]) + " " + str(profile.get("current_title", "")).lower() + " " + " ".join([str(j).lower() for j in c.get("career_history", [])])
    for key, rule in JD_ANALYSIS["must_haves"].items():
        keywords = rule.get("keywords_hard", []) + rule.get("keywords_semantic", []) + rule.get("keywords", [])
        if any(kw.lower() in full_text for kw in keywords): score += rule.get("weight", 0)
    for key, rule in JD_ANALYSIS["nice_to_haves"].items():
        keywords = rule.get("keywords", [])
        if any(kw.lower() in full_text for kw in keywords): score += rule.get("weight", 0)
    return min(1.0, score)

def get_location_score(loc):
    loc_lower = str(loc).lower()
    for key, weight in LOCATION_WEIGHTS.items():
        if key in loc_lower: return weight
    return 0.5

def compute_shipper_bonus(c):
    bonus = 0.0
    work_history = " ".join([str(j).lower() for j in c.get("career_history", [])])
    for key, rule in JD_ANALYSIS["shipper_bonus"].items():
        if any(kw.lower() in work_history for kw in rule.get("keywords", [])): bonus += rule.get("bonus", 0)
    return min(0.20, bonus)

def compute_disqualifier_penalty(c):
    penalty = 0.0
    title = str(c.get("profile", {}).get("current_title", "")).lower()
    skills_raw = c.get("skills", [])
    skills_list = [s.get("name", "") for s in skills_raw] if skills_raw and isinstance(skills_raw[0], dict) else skills_raw
    skills = " ".join([str(s).lower() for s in skills_list])
    work_history = c.get("career_history", [])
    
    IRRELEVANT = ["hr", "marketing", "content", "graphic", "sales", "finance"]
    if any(t in title for t in IRRELEVANT) and any(k in skills for k in ["ai ", "ml ", "machine learning", "rag"]):
        penalty += JD_ANALYSIS["negative_signals"]["wrong_title_with_ai_skills"].get("penalty", -0.3)
        
    consulting_count = sum(1 for j in work_history if any(f in j.get("company", "").lower() for f in jd_scoring.COMPANY_TYPE_SIGNALS["consulting_disqualify"]))
    if len(work_history) > 0 and consulting_count == len(work_history):
        penalty -= 1.0
        
    research_titles = sum(1 for j in work_history if "research" in j.get("title", "").lower() or "scientist" in j.get("title", "").lower())
    eng_titles = sum(1 for j in work_history if "engineer" in j.get("title", "").lower() or "developer" in j.get("title", "").lower())
    if research_titles > 0 and eng_titles == 0:
        penalty -= 1.0
        
    return penalty

def build_feature_matrix(candidates, embeddings, honeypot_results, jd_embedding):
    records = []
    
    for i, c in enumerate(tqdm(candidates, desc="Building features")):
        cid = c["candidate_id"]
        sig = c.get("redrob_signals", {})
        
        # Embedding similarity to JD (cosine, since both normalized)
        emb_similarity = float(np.dot(embeddings[i], jd_embedding))
        
        # Honeypot flag
        is_honeypot = honeypot_results[cid]["is_honeypot"]
        
        skills_raw = c.get("skills", [])
        skills_list = [s.get("name", "") for s in skills_raw] if skills_raw and isinstance(skills_raw[0], dict) else skills_raw
        
        record = {
            "candidate_id": cid,
            "embedding_score": emb_similarity,
            "is_honeypot": is_honeypot,
            
            # Profile substance
            "current_title": c.get("profile", {}).get("current_title", ""),
            "years_of_experience": c.get("profile", {}).get("years_of_experience", 0),
            "location": c.get("profile", {}).get("location", ""),
            "skills": "|".join(skills_list),  # pipe-separated for parquet
            "num_skills": len(skills_list),
            
            # Behavioral signals (all 23)
            "profile_completeness": sig.get("profile_completeness_score", 0),
            "last_active_date": sig.get("last_active_date"),
            "open_to_work": int(sig.get("open_to_work_flag", False)),
            "profile_views_30d": sig.get("profile_views_received_30d", 0),
            "applications_30d": sig.get("applications_submitted_30d", 0),
            "recruiter_response_rate": sig.get("recruiter_response_rate", 0),
            "avg_response_time_hours": sig.get("avg_response_time_hours", 999),
            "notice_period_days": sig.get("notice_period_days", 90),
            "github_activity_score": sig.get("github_activity_score", -1),
            "saved_by_recruiters_30d": sig.get("saved_by_recruiters_30d", 0),
            "interview_completion_rate": sig.get("interview_completion_rate", 0),
            "offer_acceptance_rate": sig.get("offer_acceptance_rate", -1),
            "verified_email": int(sig.get("verified_email", False)),
            "verified_phone": int(sig.get("verified_phone", False)),
            "linkedin_connected": int(sig.get("linkedin_connected", False)),
            "expected_salary_min": sig.get("expected_salary_range_inr_lpa", {}).get("min", 0),
            "expected_salary_max": sig.get("expected_salary_range_inr_lpa", {}).get("max", 0),
            "preferred_work_mode": sig.get("preferred_work_mode", "flexible"),
            "willing_to_relocate": int(sig.get("willing_to_relocate", False)),
            
            # Pre-computed rule-based scores (computed offline)
            "jd_keyword_score": compute_jd_keyword_score(c),
            "company_type_score": score_company_profile(c.get("career_history", [])),
            "location_score": get_location_score(c.get("profile", {}).get("location", "")),
            "shipper_bonus": compute_shipper_bonus(c),
            "disqualifier_penalty": compute_disqualifier_penalty(c),
            
            # Store raw text for reasoning generation
            "candidate_text_snippet": build_candidate_text(c)[:500],
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    os.makedirs("artifacts", exist_ok=True)
    df.to_parquet("artifacts/candidates_features.parquet", index=False, compression="snappy")
    print(f"Feature matrix: {df.shape} saved to artifacts/candidates_features.parquet")
    print(f"File size: {os.path.getsize('artifacts/candidates_features.parquet') / 1e6:.1f} MB")
    return df
