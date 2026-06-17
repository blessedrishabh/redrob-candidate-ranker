from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import os

def compute_honeypot_flags(candidate: dict) -> dict:
    flags = {
        "critical": [],
        "soft": [],
    }
    
    now = datetime.now()
    
    # --- CRITICAL FLAGS ---
    for job in candidate.get("career_history", []):
        company_founded = job.get("company_founded_year")
        job_start = int(job.get("start_date")[:4]) if job.get("start_date") else None
        if company_founded and job_start:
            if int(job_start) < int(company_founded):
                flags["critical"].append(f"started_at_{job['company']}_before_founded")
    
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    work_history = candidate.get("career_history", [])
    if work_history:
        try:
            earliest_start = min(
                int(j.get("start_date")[:4]) 
                for j in work_history 
                if j.get("start_date")
            )
            max_possible_yoe = now.year - earliest_start
            if yoe > max_possible_yoe + 2:
                flags["critical"].append(f"yoe_exceeds_timeline: claimed={yoe}, possible={max_possible_yoe}")
        except (ValueError, TypeError):
            pass
    
    skills = candidate.get("skills", [])
    skill_years = {s["name"]: s.get("duration_months", 0)/12.0 for s in skills if isinstance(s, dict)}
    for skill, years in skill_years.items():
        if years == 0 and skill in ["Python", "ML", "PyTorch", "TensorFlow"]:
            flags["critical"].append(f"expert_with_zero_years:{skill}")
    
    last_active = candidate.get("redrob_signals", {}).get("last_active_date")
    if last_active:
        try:
            active_dt = datetime.fromisoformat(str(last_active))
            if active_dt > now:
                flags["critical"].append("last_active_date_in_future")
        except (ValueError, TypeError):
            pass
    
    signup = candidate.get("redrob_signals", {}).get("signup_date")
    if last_active and signup:
        try:
            s_dt = datetime.fromisoformat(str(signup))
            a_dt = datetime.fromisoformat(str(last_active))
            if s_dt > a_dt:
                flags["critical"].append("signup_after_last_active")
        except (ValueError, TypeError):
            pass
    
    completeness = candidate.get("redrob_signals", {}).get("profile_completeness_score", 0)
    if completeness == 100:
        missing = [f for f in ["skills", "career_history", "education"] 
                   if not candidate.get(f)]
        if missing:
            flags["critical"].append(f"completeness_100_but_missing:{missing}")
    
    if len(skills) > 20:
        assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
        if not assessments or all(v == 0 for v in assessments.values()):
            flags["soft"].append("20+_skills_zero_assessments")
    
    # --- SOFT FLAGS ---
    if yoe < 3 and len(skills) > 15:
        flags["soft"].append("too_many_skills_for_tenure")
    
    signals = candidate.get("redrob_signals", {})
    rr = signals.get("recruiter_response_rate", 0)
    rt = signals.get("avg_response_time_hours", -1)
    if rr == 1.0 and rt == 0:
        flags["soft"].append("perfect_response_rate_zero_time")
    
    booleans = ["open_to_work_flag", "verified_email", "verified_phone", "linkedin_connected"]
    bool_values = [signals.get(b, False) for b in booleans]
    if all(bool_values) and signals.get("github_activity_score", -1) == 100:
        flags["soft"].append("all_signals_maxed")
    
    is_honeypot = len(flags["critical"]) > 0 or len(flags["soft"]) >= 2
    
    return {
        "is_honeypot": is_honeypot,
        "critical_flags": flags["critical"],
        "soft_flags": flags["soft"],
    }

def detect_all_honeypots(candidates: list[dict]) -> dict:
    results = {}
    for c in tqdm(candidates, desc="Honeypot detection"):
        cid = c["candidate_id"]
        results[cid] = compute_honeypot_flags(c)
    
    honeypot_ids = [cid for cid, r in results.items() if r["is_honeypot"]]
    print(f"[INFO] Detected {len(honeypot_ids)} honeypots (expected ~80)")
    print(f"[INFO] Honeypot rate in full set: {len(honeypot_ids)/len(candidates)*100:.2f}%")
    
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/honeypot_ids.txt", "w") as f:
        f.write("\n".join(honeypot_ids))
    
    return results

def spot_check_honeypots(candidates, honeypot_results, n=10):
    flagged = [c for c in candidates 
               if honeypot_results[c["candidate_id"]]["is_honeypot"]][:n]
    for c in flagged:
        print(f"\n--- {c['candidate_id']} ---")
        print(f"  Title: {c.get('profile', {}).get('current_title')}")
        print(f"  YoE: {c.get('profile', {}).get('years_of_experience')}")
        print(f"  Skills: {c.get('skills', [])[:5]}")
        print(f"  Flags: {honeypot_results[c['candidate_id']]}")
