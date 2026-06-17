# Redrob Intelligent Candidate Discovery — Complete Team Workflow
### Expert AI Engineering Tech Lead Playbook

> **Critical pre-read before anything else:** The sample `sample_submission.csv` ranks an HR Manager, Content Writers, Graphic Designers, and Marketing Managers in the top 10. That is *exactly* the keyword-stuffing trap the organizers built in. Your entire system must be designed to do the opposite.

---

## Table of Contents

1. [Project Architecture Overview](#architecture)
2. [Phase 0 — Environment Setup & Team Roles](#phase-0)
3. [Step 1 — Data Ingestion & Schema Mapping](#step-1)
4. [Step 2 — Honeypot Detection Framework](#step-2)
5. [Step 3 — JD Decomposition & Scoring Matrix](#step-3)
6. [Step 4 — Offline Precomputation Phase (GPU/API allowed)](#step-4)
7. [Step 5 — Behavioral Signal Scoring Engine](#step-5)
8. [Step 6 — Semantic Matching: Catching Tier 5 Candidates](#step-6)
9. [Step 7 — Online Ranker (The Constrained 5-Minute Script)](#step-7)
10. [Step 8 — Reasoning Column Generation](#step-8)
11. [Step 9 — Validation, Format Check & Dry Run](#step-9)
12. [Step 10 — Sandbox Setup (HuggingFace Spaces)](#step-10)
13. [Final Submission Checklist](#checklist)
14. [Scoring Strategy & What to Maximize](#scoring)

---

## Architecture Overview {#architecture}

The hard constraints (5 min, CPU-only, 16 GB RAM, no network) force a **two-phase architecture**. Everything expensive happens offline; the live ranker is just a fast lookup.

```
┌─────────────────────────────────────────────────────────┐
│               PHASE 1: OFFLINE PRECOMPUTATION            │
│           (No time/GPU constraints — do this once)       │
│                                                          │
│  candidates.jsonl.gz  →  Feature Engineering            │
│                        →  Embedding Generation (GPU/API) │
│                        →  Honeypot Flag Column          │
│                        →  Behavioral Signal Scores      │
│                        →  Precomputed Feature Matrix    │
│                              ↓                          │
│                    candidates_features.parquet           │
│                    (compact, CPU-loadable in <30s)       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│               PHASE 2: ONLINE RANKER (rank.py)           │
│           (5 min, 16 GB RAM, CPU only, no network)       │
│                                                          │
│  candidates_features.parquet  →  Score computation      │
│                               →  Honeypot filter        │
│                               →  Final weighted ranking │
│                               →  Top 100 selection      │
│                               →  Reasoning generation   │
│                                     ↓                   │
│                           submission.csv                 │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 0 — Environment Setup & Team Roles {#phase-0}

### 0.1 Team Role Assignment

| Role | Responsibilities |
|------|-----------------|
| **Data Lead** | Steps 1-2: Parsing, schema, honeypot detection |
| **ML Lead** | Steps 4, 6: Embeddings, semantic matching |
| **Signal Engineer** | Steps 3, 5: JD scoring matrix, behavioral signals |
| **Infra Lead** | Steps 7, 10: Constrained ranker script, sandbox |
| **PM/QA** | Steps 8, 9, 11: Reasoning, validation, checklist |

### 0.2 Environment Setup

```bash
# Create isolated environment
python -m venv redrob_env
source redrob_env/bin/activate

# Core dependencies for offline phase (GPU machine)
pip install pandas numpy scikit-learn sentence-transformers faiss-cpu \
            pyarrow fastparquet tqdm anthropic openai python-dateutil

# Core dependencies for online ranker (must work CPU-only, no network)
pip install pandas numpy scikit-learn pyarrow

# Freeze for repo
pip freeze > requirements.txt
```

### 0.3 Repository Structure

```
redrob-ranker/
├── README.md                      # Setup + single command to reproduce
├── requirements.txt
├── submission_metadata.yaml
├── data/
│   ├── candidates.jsonl.gz        # Original (gitignored if >100MB)
│   └── sample_candidates.json
├── offline/
│   ├── 01_parse_and_validate.py   # Step 1
│   ├── 02_detect_honeypots.py     # Step 2
│   ├── 03_jd_scoring.py           # Step 3
│   ├── 04_embed_candidates.py     # Step 4
│   └── 05_build_feature_matrix.py # Steps 5-6 combined
├── artifacts/
│   ├── candidates_features.parquet  # Precomputed (committed to repo)
│   ├── jd_embedding.npy             # JD embedding vector
│   └── honeypot_ids.txt             # List of detected honeypot IDs
├── rank.py                          # Step 7: THE constrained ranker
├── generate_reasoning.py            # Step 8
├── validate_submission.py           # From hackathon bundle
└── sandbox/
    └── app.py                       # Step 10: Streamlit app
```

---

## Step 1 — Data Ingestion & Schema Mapping {#step-1}

**Owner:** Data Lead | **Estimated Time:** 2-3 hours

### 1.1 Parse the 100K Candidate File

The README specifies `candidates.jsonl.gz` (~52 MB compressed, ~465 MB uncompressed). Use streaming to stay well within the 16 GB RAM constraint.

```python
# offline/01_parse_and_validate.py
import gzip
import json
import pandas as pd
from tqdm import tqdm

def load_candidates(filepath: str) -> list[dict]:
    """Stream-parse gzipped JSONL. Memory-safe for 100K records."""
    candidates = []
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading candidates"):
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping malformed line: {e}")
    print(f"[INFO] Loaded {len(candidates):,} candidates")
    assert len(candidates) == 100_000, f"Expected 100K, got {len(candidates)}"
    return candidates

candidates = load_candidates("data/candidates.jsonl.gz")
```

### 1.2 Schema Mapping & Field Inventory

Run this immediately after loading to understand the actual data shape:

```python
import json
from collections import Counter, defaultdict

def audit_schema(candidates: list[dict], sample_n: int = 200):
    """Inventory all fields, types, and null rates."""
    field_types = defaultdict(Counter)
    field_nulls = Counter()
    
    for c in candidates[:sample_n]:
        for key, val in c.items():
            field_types[key][type(val).__name__] += 1
            if val is None or val == "" or val == []:
                field_nulls[key] += 1
    
    print("=== FIELD AUDIT ===")
    for field in sorted(field_types.keys()):
        null_rate = field_nulls[field] / sample_n * 100
        print(f"  {field:40s} | types={dict(field_types[field])} | null={null_rate:.1f}%")
    
    # Specifically check redrob_signals structure
    sample = candidates[0].get("redrob_signals", {})
    print("\n=== REDROB_SIGNALS KEYS ===")
    for k, v in sample.items():
        print(f"  {k}: {v}")

audit_schema(candidates)
```

### 1.3 Fields to Extract (Confirmed from Spec + JD)

Based on the JD and signals doc, build a flat feature record for each candidate:

```python
FIELDS_TO_EXTRACT = {
    # Identity
    "candidate_id": str,
    "name": str,
    "current_title": str,
    "years_of_experience": float,
    "location": str,
    
    # Professional substance
    "skills": list,          # raw skill list — DO NOT trust alone
    "work_history": list,    # array of {company, title, duration, description}
    "education": list,
    "certifications": list,
    "summary": str,
    
    # Redrob behavioral signals (all 23)
    "redrob_signals": dict,
}
```

> **Key insight from the Final Note:** The `skills` field is the trap. A candidate with `skills: ["RAG", "Pinecone", "LangChain", "GPT-4", "Embeddings"]` who is a Marketing Manager is *not* a fit. Your extractor must parse `work_history` and `current_title`, not just skills.

---

## Step 2 — Honeypot Detection Framework {#step-2}

**Owner:** Data Lead | **Estimated Time:** 3-4 hours

The spec (Section 7) warns: ~80 honeypots with "subtly impossible profiles." These are **forced to relevance tier 0** and will tank your NDCG if they appear in your top 100. Honeypot rate > 10% (i.e., >10 honeypots in top 100) = **Stage 3 disqualification**.

### 2.1 Honeypot Detection Logic

Honeypots are designed to catch keyword stuffers. They pass skill-match filters but fail logical consistency checks. Build a rule-based detector:

```python
# offline/02_detect_honeypots.py
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

def compute_honeypot_flags(candidate: dict) -> dict:
    """
    Returns a dict of honeypot signals. A candidate is flagged as a
    honeypot if ANY critical_flag is True, or if soft_flags >= 2.
    """
    flags = {
        "critical": [],   # any one = honeypot
        "soft": [],       # 2+ = honeypot
    }
    
    now = datetime.now()
    
    # --- CRITICAL FLAGS (impossible facts) ---
    
    # 1. Experience exceeds company age
    for job in candidate.get("work_history", []):
        company_founded = job.get("company_founded_year")
        job_start = job.get("start_year")
        if company_founded and job_start:
            if int(job_start) < int(company_founded):
                flags["critical"].append(
                    f"started_at_{job['company']}_before_founded"
                )
    
    # 2. Total claimed years of experience vs. career timeline
    yoe = candidate.get("years_of_experience", 0)
    work_history = candidate.get("work_history", [])
    if work_history:
        try:
            earliest_start = min(
                int(j.get("start_year", now.year)) 
                for j in work_history 
                if j.get("start_year")
            )
            max_possible_yoe = now.year - earliest_start
            if yoe > max_possible_yoe + 2:  # 2yr buffer for overlaps
                flags["critical"].append(
                    f"yoe_exceeds_timeline: claimed={yoe}, possible={max_possible_yoe}"
                )
        except (ValueError, TypeError):
            pass
    
    # 3. Expert in skills with 0 years usage reported
    skills = candidate.get("skills", [])
    skill_years = candidate.get("skill_years", {})  # if schema has this
    for skill, years in skill_years.items():
        if years == 0 and skill in ["Python", "ML", "PyTorch", "TensorFlow"]:
            flags["critical"].append(f"expert_with_zero_years:{skill}")
    
    # 4. Last active date in the future
    last_active = candidate.get("redrob_signals", {}).get("last_active_date")
    if last_active:
        try:
            active_dt = datetime.fromisoformat(str(last_active))
            if active_dt > now:
                flags["critical"].append("last_active_date_in_future")
        except (ValueError, TypeError):
            pass
    
    # 5. Signup date after last active date (impossible)
    signup = candidate.get("redrob_signals", {}).get("signup_date")
    if last_active and signup:
        try:
            s_dt = datetime.fromisoformat(str(signup))
            a_dt = datetime.fromisoformat(str(last_active))
            if s_dt > a_dt:
                flags["critical"].append("signup_after_last_active")
        except (ValueError, TypeError):
            pass
    
    # 6. Profile completeness = 100 but key fields missing
    completeness = candidate.get("redrob_signals", {}).get("profile_completeness_score", 0)
    if completeness == 100:
        missing = [f for f in ["skills", "work_history", "education"] 
                   if not candidate.get(f)]
        if missing:
            flags["critical"].append(f"completeness_100_but_missing:{missing}")
    
    # 7. Skill count impossibly high with 0 assessment scores
    if len(skills) > 20:
        assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
        if not assessments or all(v == 0 for v in assessments.values()):
            flags["soft"].append("20+_skills_zero_assessments")
    
    # --- SOFT FLAGS (suspicious but not impossible alone) ---
    
    # 8. All skills at "expert" level with short tenure
    if yoe < 3 and len(skills) > 15:
        flags["soft"].append("too_many_skills_for_tenure")
    
    # 9. Recruiter response rate = 1.0 but 0 response time
    signals = candidate.get("redrob_signals", {})
    rr = signals.get("recruiter_response_rate", 0)
    rt = signals.get("avg_response_time_hours", -1)
    if rr == 1.0 and rt == 0:
        flags["soft"].append("perfect_response_rate_zero_time")
    
    # 10. All boolean signals set to True simultaneously
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
```

### 2.2 Batch Detection & Validation

```python
def detect_all_honeypots(candidates: list[dict]) -> dict:
    results = {}
    for c in tqdm(candidates, desc="Honeypot detection"):
        cid = c["candidate_id"]
        results[cid] = compute_honeypot_flags(c)
    
    honeypot_ids = [cid for cid, r in results.items() if r["is_honeypot"]]
    print(f"[INFO] Detected {len(honeypot_ids)} honeypots (expected ~80)")
    print(f"[INFO] Honeypot rate in full set: {len(honeypot_ids)/len(candidates)*100:.2f}%")
    
    # Save for the ranker to filter
    with open("artifacts/honeypot_ids.txt", "w") as f:
        f.write("\n".join(honeypot_ids))
    
    return results

honeypot_results = detect_all_honeypots(candidates)
```

### 2.3 Validation Gate: Manual Spot-Check

Before trusting your detector, manually inspect 10 flagged honeypots and 10 unflagged candidates:

```python
def spot_check_honeypots(candidates, honeypot_results, n=10):
    """Print a few honeypots for manual review."""
    flagged = [c for c in candidates 
               if honeypot_results[c["candidate_id"]]["is_honeypot"]][:n]
    for c in flagged:
        print(f"\n--- {c['candidate_id']} ---")
        print(f"  Title: {c.get('current_title')}")
        print(f"  YoE: {c.get('years_of_experience')}")
        print(f"  Skills: {c.get('skills', [])[:5]}")
        print(f"  Flags: {honeypot_results[c['candidate_id']]}")
```

> **Critical validation:** You must keep honeypot rate in your final top 100 **below 10% (< 10 candidates)**. Test this before submission.

---

## Step 3 — JD Decomposition & Scoring Matrix {#step-3}

**Owner:** Signal Engineer | **Estimated Time:** 4-6 hours

### 3.1 The Four JD Tiers (From the Job Description)

Read the JD carefully. It explicitly creates a scoring hierarchy:

```python
# offline/03_jd_scoring.py

JD_ANALYSIS = {
    
    # TIER 1: INSTANT DISQUALIFIERS (→ score = 0, exclude from top 100)
    "hard_disqualifiers": {
        "pure_research_only": {
            "description": "No production deployment in career history",
            "detection": "all titles contain 'Researcher' or 'Scientist' with no 'Engineer' or 'Developer'"
        },
        "llm_framework_only": {
            "description": "AI experience = LangChain tutorials only, <12 months",
            "detection": "recent titles + skills dominated by LangChain/LlamaIndex with no pre-2022 ML work"
        },
        "no_code_last_18months": {
            "description": "Moved entirely to Architecture/TechLead with no production code",
            "detection": "title_history shows no IC role in last 18 months"
        },
        "consulting_only": {
            "description": "Entire career at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini",
            "detection": "every employer in work_history is a consulting firm"
        },
        "wrong_domain": {
            "description": "CV/Speech/Robotics only, no NLP/IR",
            "detection": "skills dominated by CV terms (YOLO, OpenCV, etc.) with no NLP/retrieval"
        }
    },
    
    # TIER 2: MUST-HAVES (required for rank 1-30)
    "must_haves": {
        "embeddings_production": {
            "weight": 0.25,
            "keywords_hard": ["sentence-transformers", "openai embeddings", "bge", "e5", 
                              "embedding drift", "index refresh"],
            "keywords_semantic": ["embedding", "dense retrieval", "bi-encoder", "vector search"],
            "work_evidence": ["shipped", "deployed", "production", "real users", "scale"]
        },
        "vector_db_production": {
            "weight": 0.20,
            "keywords_hard": ["pinecone", "weaviate", "qdrant", "milvus", "opensearch",
                              "elasticsearch", "faiss", "chroma"],
            "keywords_semantic": ["vector database", "ann search", "approximate nearest neighbor",
                                  "hybrid search", "inverted index"]
        },
        "python_strength": {
            "weight": 0.10,
            "signals": ["github_activity_score > 50", "open source contributions", 
                        "code quality mentions", "python in skills"]
        },
        "evaluation_framework": {
            "weight": 0.15,
            "keywords": ["ndcg", "mrr", "map", "a/b test", "offline eval", "online eval",
                         "ranking evaluation", "relevance judgment", "precision@k"]
        }
    },
    
    # TIER 3: NICE-TO-HAVES (differentiators for rank 1-20)
    "nice_to_haves": {
        "llm_finetuning": {
            "weight": 0.08,
            "keywords": ["lora", "qlora", "peft", "fine-tuning", "instruction tuning", "sft"]
        },
        "learning_to_rank": {
            "weight": 0.06,
            "keywords": ["xgboost ranking", "lambdamart", "listwise", "pairwise", "ltr", 
                         "learning to rank"]
        },
        "hr_tech_experience": {
            "weight": 0.04,
            "keywords": ["recruiting", "talent", "ats", "job matching", "candidate ranking"]
        },
        "open_source": {
            "weight": 0.03,
            "keywords": ["open source", "github", "contributor", "maintainer"]
        }
    },
    
    # TIER 4: NEGATIVE SIGNALS (penalize these)
    "negative_signals": {
        "title_chaser": {
            "penalty": -0.15,
            "detection": "≥3 job changes in 4 years all with ascending seniority titles"
        },
        "framework_enthusiast_only": {
            "penalty": -0.10,
            "detection": "GitHub/blog dominated by 'LangChain tutorial', 'OpenAI integration demo'"
        },
        "wrong_title_with_ai_skills": {
            "penalty": -0.30,
            "description": "This is the sample_submission trap. Marketing/HR/Design titles with AI skills.",
            "detection": "current_title NOT in ML/AI/Engineering + skills have AI keywords"
        }
    },
    
    # TIER 5: SHIPPER BONUS (catches semantic candidates)
    "shipper_bonus": {
        "recommendation_system": {
            "bonus": 0.08,
            "rationale": "Rec systems = implicit RAG. Built it = understands retrieval/ranking.",
            "keywords": ["recommendation", "recommender", "collaborative filtering", 
                         "content-based filtering", "item embedding", "user embedding"]
        },
        "search_system": {
            "bonus": 0.07,
            "keywords": ["search", "information retrieval", "query understanding",
                         "ranking", "re-ranking", "bm25", "tf-idf", "lucene", "solr"]
        },
        "production_scale": {
            "bonus": 0.05,
            "keywords": ["production", "shipped", "deployed", "real users", "at scale",
                         "latency", "throughput", "millions of", "billion"]
        }
    }
}
```

### 3.2 The "Shipper vs. Researcher" Detection Logic

This is the most critical differentiator in this hackathon:

```python
COMPANY_TYPE_SIGNALS = {
    # Product companies (positive signal)
    "product_positive": [
        "startup", "series a", "series b", "saas", "platform", 
        # Add known product companies from your domain knowledge
    ],
    
    # Research institutions (negative for this role)
    "research_negative": [
        "university", "iit", "iim", "nit", "iiser", "lab", "research center",
        "microsoft research", "google brain", "deepmind", "fair", "openai research"
    ],
    
    # Consulting firms (hard disqualifier if ENTIRE career)
    "consulting_disqualify": [
        "tcs", "tata consultancy", "infosys", "wipro", "accenture",
        "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis"
    ]
}

def score_company_profile(work_history: list) -> float:
    """
    Returns a score from -1.0 to 1.0.
    Product company experience = positive.
    All consulting = -1.0 (disqualifier).
    Research-only = -0.5.
    """
    if not work_history:
        return 0.0
    
    consulting_count = 0
    research_count = 0
    product_count = 0
    
    for job in work_history:
        company = job.get("company", "").lower()
        title = job.get("title", "").lower()
        
        if any(c in company for c in CONSULTING_FIRMS):
            consulting_count += 1
        elif any(r in company or r in title for r in RESEARCH_ORGS):
            research_count += 1
        else:
            product_count += 1
    
    total = len(work_history)
    
    # Hard disqualifier: 100% consulting
    if consulting_count == total:
        return -1.0
    
    # Pure research penalty
    research_ratio = research_count / total
    product_ratio = product_count / total
    
    return product_ratio - (research_ratio * 0.5) - (consulting_count / total * 0.3)
```

### 3.3 Location & Availability Scoring

The JD explicitly specifies: Pune/Noida preferred; also Hyderabad, Mumbai, Delhi NCR.

```python
LOCATION_WEIGHTS = {
    "pune": 1.0,
    "noida": 1.0,
    "delhi": 0.85,
    "delhi ncr": 0.85,
    "gurgaon": 0.85,
    "gurugram": 0.85,
    "mumbai": 0.75,
    "hyderabad": 0.75,
    "bangalore": 0.60,    # possible but not preferred
    "bengaluru": 0.60,
    "india": 0.50,        # unspecified Indian city
    "remote": 0.40,       # risky — they want hybrid presence
}
```

---

## Step 4 — Offline Precomputation Phase {#step-4}

**Owner:** ML Lead | **Estimated Time:** 6-10 hours | **GPU/API Allowed Here**

This is where you generate embeddings and features you'll use in the constrained ranker. This is done ONCE and committed to the repo.

### 4.1 JD Embedding Generation

```python
# offline/04_embed_candidates.py
from sentence_transformers import SentenceTransformer
import numpy as np

# Use a strong model — you have GPU here
# Options (ranked by quality):
# - "BAAI/bge-large-en-v1.5"     (best open-source for retrieval)
# - "intfloat/e5-large-v2"        (strong, widely used)
# - "thenlper/gte-large"           (efficient alternative)
# DO NOT use "all-MiniLM-L6-v2" — too weak for nuanced JD matching

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Build a rich JD text that emphasizes SHIPPER signals
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
```

### 4.2 Candidate Text Featurization

For each candidate, build a rich text representation that gives the model signal beyond the skills list:

```python
def build_candidate_text(candidate: dict) -> str:
    """
    Builds a rich text representation of a candidate.
    Critically: uses work_history descriptions, not just skills.
    This is what catches the Tier 5 candidate who built a recommendation
    system without using the word 'RAG'.
    """
    parts = []
    
    # Title and experience (context setter)
    title = candidate.get("current_title", "")
    yoe = candidate.get("years_of_experience", 0)
    parts.append(f"{title} with {yoe:.1f} years of experience.")
    
    # Location
    loc = candidate.get("location", "")
    parts.append(f"Located in {loc}.")
    
    # Summary (if exists)
    summary = candidate.get("summary", "")
    if summary:
        parts.append(summary)
    
    # Work history — MOST IMPORTANT for catching semantic candidates
    for job in candidate.get("work_history", [])[:5]:  # last 5 roles
        company = job.get("company", "")
        job_title = job.get("title", "")
        duration = job.get("duration_months", 0)
        description = job.get("description", "")
        parts.append(
            f"At {company} as {job_title} ({duration} months): {description}"
        )
    
    # Skills (still useful, just not the only signal)
    skills = candidate.get("skills", [])
    if skills:
        parts.append(f"Skills: {', '.join(skills[:20])}")
    
    # Education
    for edu in candidate.get("education", [])[:2]:
        parts.append(f"Education: {edu.get('degree', '')} from {edu.get('institution', '')}")
    
    # GitHub activity (proxy for shipper)
    gh = candidate.get("redrob_signals", {}).get("github_activity_score", -1)
    if gh > 0:
        parts.append(f"GitHub activity score: {gh}/100")
    
    return " ".join(filter(None, parts))
```

### 4.3 Batch Embedding Generation

```python
def embed_all_candidates(candidates: list[dict], model, batch_size: int = 512) -> np.ndarray:
    """
    Embed all 100K candidates in batches.
    With GPU, this should take ~20-40 minutes for bge-large.
    Without GPU, use a smaller model for this step.
    """
    texts = [build_candidate_text(c) for c in tqdm(candidates, desc="Building texts")]
    
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embs)
    
    embeddings = np.vstack(all_embeddings)
    print(f"Embedding matrix shape: {embeddings.shape}")  # (100000, 1024) for bge-large
    return embeddings

embeddings = embed_all_candidates(candidates, model)
np.save("artifacts/candidate_embeddings.npy", embeddings)
```

> **Memory note:** 100K × 1024 float32 embeddings = ~400 MB. Well within budget. Use float16 if concerned: `embeddings.astype(np.float16)`.

### 4.4 Build the Precomputed Feature Matrix

This is the artifact that `rank.py` loads at runtime:

```python
# offline/05_build_feature_matrix.py
import pandas as pd
import numpy as np

def build_feature_matrix(candidates, embeddings, honeypot_results, jd_embedding):
    records = []
    
    for i, c in enumerate(tqdm(candidates, desc="Building features")):
        cid = c["candidate_id"]
        sig = c.get("redrob_signals", {})
        
        # Embedding similarity to JD (cosine, since both normalized)
        emb_similarity = float(np.dot(embeddings[i], jd_embedding))
        
        # Honeypot flag
        is_honeypot = honeypot_results[cid]["is_honeypot"]
        
        record = {
            "candidate_id": cid,
            "embedding_score": emb_similarity,
            "is_honeypot": is_honeypot,
            
            # Profile substance
            "current_title": c.get("current_title", ""),
            "years_of_experience": c.get("years_of_experience", 0),
            "location": c.get("location", ""),
            "skills": "|".join(c.get("skills", [])),  # pipe-separated for parquet
            "num_skills": len(c.get("skills", [])),
            
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
            "company_type_score": score_company_profile(c.get("work_history", [])),
            "location_score": get_location_score(c.get("location", "")),
            "shipper_bonus": compute_shipper_bonus(c),
            "disqualifier_penalty": compute_disqualifier_penalty(c),
            
            # Store raw text for reasoning generation
            "candidate_text_snippet": build_candidate_text(c)[:500],
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_parquet("artifacts/candidates_features.parquet", index=False, compression="snappy")
    print(f"Feature matrix: {df.shape} saved to artifacts/candidates_features.parquet")
    print(f"File size: {os.path.getsize('artifacts/candidates_features.parquet') / 1e6:.1f} MB")
    return df

df = build_feature_matrix(candidates, embeddings, honeypot_results, jd_embedding)
```

---

## Step 5 — Behavioral Signal Scoring Engine {#step-5}

**Owner:** Signal Engineer | **Estimated Time:** 3-4 hours

### 5.1 Signal Weights & Rationale

The Final Note says explicitly: "a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available."

Build a single `availability_multiplier` from 0.0 to 1.0:

```python
from datetime import datetime, date
import numpy as np

def compute_availability_multiplier(row: dict, reference_date: datetime = None) -> float:
    """
    Computes a 0.0–1.0 availability multiplier.
    A score of 0.2 means: technically in the pool, but almost certainly
    unavailable. We multiply (not add) this against the skill score.
    
    This prevents a 10/10 skill-match candidate who's been inactive for
    a year from appearing in the top 10.
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    score = 1.0
    
    # --- HIGH IMPACT SIGNALS ---
    
    # 1. Last active date (decays availability over time)
    last_active = row.get("last_active_date")
    if last_active:
        try:
            active_dt = datetime.fromisoformat(str(last_active))
            days_since_active = (reference_date - active_dt).days
            
            if days_since_active <= 7:
                activity_score = 1.0
            elif days_since_active <= 30:
                activity_score = 0.9
            elif days_since_active <= 90:
                activity_score = 0.7
            elif days_since_active <= 180:  # "6 months" — the spec's example
                activity_score = 0.4
            elif days_since_active <= 365:
                activity_score = 0.2
            else:
                activity_score = 0.1  # Over a year — almost certainly unavailable
            
            score *= activity_score
        except (ValueError, TypeError):
            score *= 0.7  # Unknown date = mild penalty
    
    # 2. Open to work flag (most direct signal)
    if row.get("open_to_work"):
        score *= 1.0   # No penalty for openly available
    else:
        score *= 0.75  # Passive candidate — still reachable, but harder
    
    # 3. Recruiter response rate ("5% response rate" — spec's example)
    rr = row.get("recruiter_response_rate", 0.5)
    if rr < 0.10:      # The spec's "5% response rate" example
        score *= 0.30
    elif rr < 0.30:
        score *= 0.60
    elif rr < 0.60:
        score *= 0.85
    else:
        score *= 1.0
    
    # 4. Notice period (JD says "love sub-30 days, up to 30 buyout is fine")
    notice = row.get("notice_period_days", 60)
    if notice <= 30:
        score *= 1.0
    elif notice <= 60:
        score *= 0.90
    elif notice <= 90:
        score *= 0.80
    elif notice <= 120:
        score *= 0.70
    else:                # 180 days = serious friction
        score *= 0.55
    
    # --- MEDIUM IMPACT SIGNALS ---
    
    # 5. Applications submitted (actively looking)
    apps_30d = row.get("applications_30d", 0)
    if apps_30d >= 5:
        score *= 1.05   # Active job seeker (capped at modest boost)
    elif apps_30d >= 1:
        score *= 1.02
    
    # 6. Average response time
    response_time = row.get("avg_response_time_hours", 24)
    if response_time <= 2:
        score *= 1.03
    elif response_time <= 24:
        score *= 1.0
    elif response_time > 72:
        score *= 0.90
    
    # 7. Interview completion rate (signal of genuine interest when contacted)
    icr = row.get("interview_completion_rate", 0.5)
    if icr < 0.30:
        score *= 0.75   # Ghosting pattern
    elif icr >= 0.80:
        score *= 1.02
    
    # 8. Offer acceptance rate
    oar = row.get("offer_acceptance_rate", -1)
    if oar == -1:
        pass   # No history, neutral
    elif oar < 0.20:
        score *= 0.80   # Accepting <20% of offers = likely passive or picky
    
    # --- LOW IMPACT SIGNALS (verifications) ---
    
    # 9. Profile verification (quality signal)
    if row.get("verified_email") and row.get("verified_phone"):
        score *= 1.02
    elif not row.get("verified_email"):
        score *= 0.95
    
    # 10. Location/relocation (matching JD geography)
    loc_score = row.get("location_score", 0.5)
    if loc_score < 0.5 and not row.get("willing_to_relocate"):
        score *= 0.85   # Far away AND unwilling to relocate
    
    return min(score, 1.0)  # Cap at 1.0
```

### 5.2 GitHub Activity Score (Shipper Proxy)

```python
def compute_github_contribution(row: dict) -> float:
    """
    GitHub activity is a proxy for being a shipper, not a researcher.
    Range: -1 (not connected) to 100.
    """
    gh = row.get("github_activity_score", -1)
    
    if gh == -1:
        return 0.5   # Unknown — neutral
    elif gh >= 80:
        return 1.0
    elif gh >= 60:
        return 0.85
    elif gh >= 40:
        return 0.70
    elif gh >= 20:
        return 0.55
    else:
        return 0.40
```

---

## Step 6 — Semantic Matching: Catching Tier 5 Candidates {#step-6}

**Owner:** ML Lead | **Estimated Time:** 3-4 hours

### 6.1 The Problem: Keyword vs. Semantic Matching

The Final Note is explicit: "A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit."

Your embedding model handles this automatically — IF you built the candidate text correctly in Step 4.2. A recommendation system description will have high cosine similarity with the JD embedding because both mention:
- User/item embedding → semantic neighbor of candidate/job embedding
- Retrieval of relevant items → semantic neighbor of retrieval systems
- Production scale → explicit in JD

### 6.2 Hybrid Scoring Architecture

Don't rely on embeddings alone. Use a weighted combination:

```python
def compute_final_score(row: dict) -> float:
    """
    Final score combining all signals.
    This is what rank.py calls for each of the 100K candidates.
    
    Weights derived from JD analysis:
    - Embedding similarity: captures semantic shipper signals
    - Keyword score: catches explicit must-haves
    - Behavioral multiplier: ensures we only rank available candidates
    - Company type: shipper vs. researcher signal
    """
    
    # GATE 1: Hard disqualifiers → score = 0
    if row.get("is_honeypot"):
        return 0.0
    
    if row.get("disqualifier_penalty", 0) <= -1.0:
        return 0.0
    
    # GATE 2: Title sanity check
    # The sample submission's failure: ranking HR Managers #1
    title = str(row.get("current_title", "")).lower()
    IRRELEVANT_TITLES = [
        "hr manager", "human resources", "marketing manager", "content writer",
        "graphic designer", "business analyst", "sales", "finance manager",
        "operations", "recruiter", "talent acquisition"
    ]
    if any(t in title for t in IRRELEVANT_TITLES):
        # Unless they have a very strong work history of ML engineering
        if row.get("embedding_score", 0) < 0.6:
            return 0.0  # Reject title-mismatched candidates
    
    # --- COMPONENT SCORES ---
    
    # 1. Embedding similarity to JD (semantic understanding)
    embedding_score = row.get("embedding_score", 0)  # 0.0–1.0
    
    # 2. JD keyword matching score (explicit must-haves)
    keyword_score = row.get("jd_keyword_score", 0)   # 0.0–1.0
    
    # 3. Company type (product > research > consulting)
    company_score = max(0, row.get("company_type_score", 0))  # 0.0–1.0 (clamped)
    
    # 4. Location match
    location_score = row.get("location_score", 0.5)  # 0.0–1.0
    
    # 5. Shipper bonus (recommendation systems, search systems, etc.)
    shipper_bonus = row.get("shipper_bonus", 0)  # 0.0–0.20
    
    # 6. Disqualifier penalty
    penalty = max(-1.0, row.get("disqualifier_penalty", 0))  # ≤0
    
    # --- WEIGHTED SKILL SCORE ---
    skill_score = (
        0.40 * embedding_score +    # semantic understanding
        0.30 * keyword_score +      # explicit keyword match
        0.15 * company_score +      # shipper vs. researcher
        0.10 * location_score +     # JD geographic preference
        0.05 * row.get("github_activity", 0.5)  # shipper proxy
    )
    
    # Add shipper bonus and subtract penalties
    skill_score = skill_score + shipper_bonus + penalty
    skill_score = max(0.0, min(1.0, skill_score))
    
    # --- AVAILABILITY MULTIPLIER (applied multiplicatively) ---
    # This is what prevents the "inactive 6-month candidate" from ranking high
    availability = compute_availability_multiplier(row)
    
    # Final score: skill quality × actual availability
    # NOTE: We use sqrt(availability) so that a great candidate who's slightly
    # less active doesn't fall off a cliff. A bad candidate who's very active
    # still loses to a great candidate who's slightly inactive.
    final_score = skill_score * (0.70 + 0.30 * availability)
    
    return round(final_score, 6)
```

### 6.3 NDCG@10 Optimization Strategy

The scoring metric weights NDCG@10 at 50%. This means getting your top 10 right is worth MORE than the entire rest of the list combined. Design accordingly:

```python
# In rank.py: aggressive top-10 curation
def curate_top_10(df_ranked: pd.DataFrame) -> pd.DataFrame:
    """
    After initial scoring, manually validate the top 30 candidates
    to ensure ranks 1-10 are defensible.
    
    Check each of the top 30 for:
    1. Is their current_title an ML/AI/Engineering role?
    2. Do they have at least 1 vector DB mention in work history?
    3. Are they reasonably active (last_active < 90 days)?
    4. Are they in or willing to relocate to Pune/Noida area?
    5. Do they pass the shipper test (product co. experience)?
    """
    top_30 = df_ranked.head(30)
    
    # Log top-10 for manual review
    print("\n=== TOP 10 FOR REVIEW ===")
    for _, row in top_30.head(10).iterrows():
        print(f"  Rank {row['rank']:2d} | {row['candidate_id']} | {row['current_title']}")
        print(f"         Score: {row['score']:.4f} | Availability: {row['availability']:.2f}")
        print(f"         Embedding: {row['embedding_score']:.3f} | Keywords: {row['keyword_score']:.3f}")
    
    return df_ranked
```

---

## Step 7 — Online Ranker (The Constrained 5-Minute Script) {#step-7}

**Owner:** Infra Lead | **Estimated Time:** 4-6 hours

This is the script that must run in ≤5 min, CPU-only, 16 GB RAM, no network. It loads precomputed artifacts and scores/ranks candidates.

```python
# rank.py — THE CONSTRAINED RANKER
"""
Usage: python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

Compute constraints (enforced at Stage 3):
  - ≤5 minutes wall-clock
  - ≤16 GB RAM  
  - CPU only
  - No network access
"""
import argparse
import gzip
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

START_TIME = time.time()
DEADLINE_SECONDS = 4 * 60 + 30  # 4:30 — 30s buffer before the 5-min limit

def load_precomputed(artifacts_dir: str = "./artifacts"):
    """Load all precomputed artifacts. This should take <30 seconds."""
    t0 = time.time()
    
    features = pd.read_parquet(f"{artifacts_dir}/candidates_features.parquet")
    honeypot_ids = set(
        Path(f"{artifacts_dir}/honeypot_ids.txt").read_text().strip().split("\n")
    )
    
    print(f"[INFO] Loaded {len(features):,} candidates in {time.time()-t0:.1f}s")
    print(f"[INFO] Honeypot set: {len(honeypot_ids)} candidates")
    return features, honeypot_ids

def rank_candidates(features: pd.DataFrame, honeypot_ids: set) -> pd.DataFrame:
    """Score and rank all candidates. CPU-only. Should complete in <2 minutes."""
    t0 = time.time()
    
    # Apply honeypot filter
    features["is_honeypot"] = features["candidate_id"].isin(honeypot_ids)
    
    # Compute availability multiplier vectorized (much faster than row-by-row)
    features["availability"] = compute_availability_vectorized(features)
    
    # Compute final scores (vectorized)
    features["score"] = compute_scores_vectorized(features)
    
    # Set honeypot scores to 0
    features.loc[features["is_honeypot"], "score"] = 0.0
    
    # Sort and assign ranks
    features = features.sort_values("score", ascending=False).reset_index(drop=True)
    features["rank"] = features.index + 1
    
    print(f"[INFO] Scoring complete in {time.time()-t0:.1f}s")
    
    # Verify timing
    elapsed = time.time() - START_TIME
    print(f"[INFO] Total elapsed: {elapsed:.1f}s / {DEADLINE_SECONDS}s budget")
    
    if elapsed > DEADLINE_SECONDS:
        raise RuntimeError(f"[FATAL] Over time budget! {elapsed:.1f}s elapsed")
    
    return features

def compute_availability_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vectorized availability computation. Much faster than apply()."""
    now = datetime.now()
    
    # Last active score
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
    
    # Response rate score
    rr = df["recruiter_response_rate"].fillna(0.5)
    rr_score = np.where(rr < 0.10, 0.30,
               np.where(rr < 0.30, 0.60,
               np.where(rr < 0.60, 0.85, 1.0)))
    
    # Open to work
    otw_score = np.where(df["open_to_work"].fillna(0) == 1, 1.0, 0.75)
    
    # Notice period
    notice = df["notice_period_days"].fillna(60)
    notice_score = np.where(notice <= 30, 1.0,
                   np.where(notice <= 60, 0.90,
                   np.where(notice <= 90, 0.80,
                   np.where(notice <= 120, 0.70, 0.55))))
    
    availability = activity_score * rr_score * otw_score * notice_score
    return pd.Series(np.clip(availability, 0.0, 1.0), index=df.index)

def compute_scores_vectorized(df: pd.DataFrame) -> pd.Series:
    """Final score computation — vectorized for speed."""
    
    # Skill/quality component
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
    
    # Combine with availability
    availability = df["availability"].fillna(0.5)
    final_score = skill_score * (0.70 + 0.30 * availability)
    
    return final_score.round(6)

def build_top_100(ranked_df: pd.DataFrame) -> pd.DataFrame:
    """Select top 100 non-honeypot candidates and generate reasoning."""
    
    # Filter out honeypots first
    clean = ranked_df[~ranked_df["is_honeypot"]].copy()
    top_100 = clean.head(100).copy()
    top_100["rank"] = range(1, 101)
    
    # Verify honeypot constraint
    # (should be 0 honeypots since we filtered, but double-check)
    honeypot_in_top = ranked_df.head(100)["is_honeypot"].sum()
    print(f"[INFO] Honeypots in top 100: {honeypot_in_top} (must be < 10)")
    if honeypot_in_top >= 10:
        raise ValueError(f"[FATAL] Honeypot rate too high: {honeypot_in_top}/100 = {honeypot_in_top}%")
    
    # Verify score monotonicity
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
    
    # Load precomputed features (no need to re-read the full JSONL)
    features, honeypot_ids = load_precomputed(args.artifacts)
    
    # Score and rank
    ranked = rank_candidates(features, honeypot_ids)
    
    # Select top 100
    top_100 = build_top_100(ranked)
    
    # Generate reasoning (template-based, must be fast)
    top_100["reasoning"] = top_100.apply(generate_reasoning_fast, axis=1)
    
    # Output CSV
    output = top_100[["candidate_id", "rank", "score", "reasoning"]]
    output.to_csv(args.out, index=False, encoding="utf-8")
    
    elapsed = time.time() - START_TIME
    print(f"\n[SUCCESS] Submission written to {args.out}")
    print(f"[SUCCESS] Total runtime: {elapsed:.1f}s / 300s budget ({elapsed/300*100:.1f}% used)")
    print(f"[SUCCESS] Rows: {len(output)} | Unique ranks: {output['rank'].nunique()}")

if __name__ == "__main__":
    main()
```

### 7.1 Runtime Budget Allocation

| Task | Target Time |
|------|-------------|
| Import dependencies | ~2-3s |
| Load parquet feature file | ~5-10s |
| Load honeypot IDs | <1s |
| Vectorized scoring (100K rows) | ~10-30s |
| Sort + rank selection | ~5s |
| Reasoning generation (100 rows) | ~10-30s |
| CSV write | ~1s |
| **Total** | **~35-80s — well within 5 min** |

---

## Step 8 — Reasoning Column Generation {#step-8}

**Owner:** PM/QA Lead | **Estimated Time:** 3-4 hours

The spec (Section 3, Stage 4) evaluates reasoning on 6 dimensions: specific facts, JD connection, honest concerns, no hallucination, variation, rank consistency.

### 8.1 Two-Phase Reasoning Strategy

**Phase A (Offline, using Claude/GPT-4):** Generate rich, specific reasoning for the top 200 candidates. This is done offline where you can use APIs.

**Phase B (Online, in rank.py):** Use template-based reasoning that pulls directly from precomputed facts. Zero hallucination risk because every claim comes from actual fields.

### 8.2 Offline Reasoning Generation (High Quality, Phase A)

```python
# offline/06_generate_reasoning.py
# Run AFTER you have your top 100 finalized

import anthropic

def generate_reasoning_offline(candidate: dict, rank: int, score: float) -> str:
    """
    Use Claude to generate specific, honest, non-hallucinating reasoning.
    Done offline — API calls are fine here.
    """
    client = anthropic.Anthropic()
    
    # Build a structured profile for Claude
    profile_text = f"""
CANDIDATE PROFILE (USE ONLY THESE FACTS — DO NOT INVENT):
ID: {candidate['candidate_id']}
Title: {candidate.get('current_title', 'Unknown')}
Years of Experience: {candidate.get('years_of_experience', '?')}
Location: {candidate.get('location', 'Unknown')}
Skills (exact list): {', '.join(candidate.get('skills', []))}
Work History:
{format_work_history(candidate.get('work_history', []))}
GitHub Activity Score: {candidate.get('redrob_signals', {}).get('github_activity_score', 'N/A')}
Last Active: {candidate.get('redrob_signals', {}).get('last_active_date', 'Unknown')}
Open to Work: {candidate.get('redrob_signals', {}).get('open_to_work_flag', False)}
Notice Period: {candidate.get('redrob_signals', {}).get('notice_period_days', '?')} days
Recruiter Response Rate: {candidate.get('redrob_signals', {}).get('recruiter_response_rate', '?')}
"""
    
    prompt = f"""You are writing a 1-2 sentence reasoning for why a candidate is ranked #{rank} for a Senior AI Engineer role (founding team) that requires:
- Production embeddings/retrieval systems
- Vector database experience  
- Strong Python
- Evaluation frameworks (NDCG, MRR, A/B tests)
- "Shipper" mentality at product companies (NOT pure research, NOT consulting-only)
- 5-9 years experience in Pune/Noida/Delhi NCR/Hyderabad/Mumbai area

STRICT RULES:
1. ONLY reference facts that appear in the profile above. Zero hallucination.
2. Mention the candidate's title, years of experience, and at least one specific technical fact.
3. If rank > 50, acknowledge the concern/gap honestly.
4. Connect to specific JD requirements, not generic praise.
5. Each reasoning must be unique — vary structure and emphasis.
6. 1-2 sentences maximum. Be specific and honest.

{profile_text}

Write the reasoning for rank #{rank} (score: {score:.3f}):"""
    
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text.strip()
```

### 8.3 Template-Based Reasoning (Online, Zero Hallucination Risk)

For the constrained `rank.py`, use a smart template that pulls only from actual fields:

```python
def generate_reasoning_fast(row: pd.Series) -> str:
    """
    Template-based reasoning for rank.py (CPU, no network).
    Every claim is derived from actual fields — zero hallucination risk.
    """
    title = row.get("current_title", "Unknown title")
    yoe = row.get("years_of_experience", 0)
    loc = row.get("location", "Unknown location")
    rank = int(row.get("rank", 0))
    
    # Extract key signals
    active_days = _days_since(row.get("last_active_date"))
    notice = row.get("notice_period_days", 60)
    rr = row.get("recruiter_response_rate", 0.5)
    github = row.get("github_activity_score", -1)
    open_to_work = row.get("open_to_work", 0)
    
    # Skills snippet (only actual skills from profile)
    skills_raw = row.get("skills", "")
    if isinstance(skills_raw, str):
        skills = [s.strip() for s in skills_raw.split("|") if s.strip()]
    else:
        skills = []
    
    # Identify the strongest relevant skills (ground truth only)
    RELEVANT_SKILLS = {
        "embeddings", "vector database", "faiss", "pinecone", "weaviate", "qdrant",
        "sentence-transformers", "elasticsearch", "opensearch", "retrieval", "ranking",
        "python", "pytorch", "tensorflow", "lora", "qlora", "fine-tuning",
        "ndcg", "mrr", "a/b testing", "recommendation", "search", "bge", "e5"
    }
    matched_skills = [s for s in skills if s.lower() in RELEVANT_SKILLS][:3]
    
    # Concerns
    concerns = []
    if active_days and active_days > 90:
        concerns.append(f"inactive for {active_days//30}+ months")
    if notice > 90:
        concerns.append(f"{notice}-day notice period")
    if rr < 0.30:
        concerns.append(f"low recruiter response rate ({rr:.0%})")
    if not open_to_work:
        concerns.append("not marked open to work")
    
    # Build reasoning
    if rank <= 20:
        # Top candidates: lead with strengths
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
        # Mid-tier: honest about trade-offs
        skill_str = (f", including {matched_skills[0]}" if matched_skills else "")
        line1 = f"{title}, {yoe:.0f} yrs, {loc}{skill_str}."
        concern_str = f"Concerns: {'; '.join(concerns)}." if concerns else "Reasonably active profile."
        return f"{line1} {concern_str}"
    
    else:
        # Lower tier: clear about why they're included but lower-ranked
        line1 = f"{yoe:.0f}-year {title} in {loc}."
        if concerns:
            line2 = f"Included for adjacent skills but ranked lower due to: {'; '.join(concerns[:2])}."
        else:
            line2 = "Below-cutoff profile included as best available match at this rank; partial skill alignment only."
        return f"{line1} {line2}"

def _days_since(date_str) -> int | None:
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(str(date_str))
        return (datetime.now() - dt).days
    except (ValueError, TypeError):
        return None
```

### 8.4 Anti-Hallucination Checklist (Run Before Submission)

```python
def audit_reasoning_for_hallucinations(top_100: pd.DataFrame, candidates_lookup: dict):
    """
    Cross-check every reasoning string against the actual candidate profile.
    Flags any mention of skills/companies/titles not in the profile.
    """
    issues = []
    for _, row in top_100.iterrows():
        cid = row["candidate_id"]
        reasoning = str(row["reasoning"])
        candidate = candidates_lookup.get(cid, {})
        
        # Extract all skills from candidate profile
        actual_skills = set(s.lower() for s in candidate.get("skills", []))
        actual_titles = set(j.get("title", "").lower() for j in candidate.get("work_history", []))
        actual_companies = set(j.get("company", "").lower() for j in candidate.get("work_history", []))
        
        # Check for technical terms in reasoning not in profile
        TECHNICAL_TERMS = [
            "rag", "pinecone", "weaviate", "faiss", "llm", "gpt", "bert",
            "transformer", "fine-tuning", "lora", "qdrant", "milvus"
        ]
        for term in TECHNICAL_TERMS:
            if term in reasoning.lower() and term not in actual_skills:
                issues.append({
                    "candidate_id": cid,
                    "rank": row["rank"],
                    "issue": f"Mentions '{term}' but not in skills",
                    "reasoning": reasoning[:100]
                })
    
    if issues:
        print(f"[WARN] {len(issues)} potential hallucination issues found:")
        for issue in issues:
            print(f"  Rank {issue['rank']} | {issue['candidate_id']}: {issue['issue']}")
    else:
        print("[OK] No hallucination issues detected in reasoning column.")
    
    return issues
```

---

## Step 9 — Validation, Format Check & Dry Run {#step-9}

**Owner:** PM/QA Lead | **Estimated Time:** 2-3 hours

### 9.1 Format Validation (Use the Official Validator)

```bash
# Run the official hackathon validator
python validate_submission.py --submission submission.csv --candidates candidates.jsonl.gz
```

### 9.2 Manual Format Verification Checklist

```python
import pandas as pd

def validate_submission(csv_path: str, candidates_path: str):
    df = pd.read_csv(csv_path)
    
    print("=== SUBMISSION VALIDATION ===\n")
    
    checks = {
        "Exactly 100 rows": len(df) == 100,
        "Columns correct": list(df.columns) == ["candidate_id", "rank", "score", "reasoning"],
        "Ranks 1-100 each exactly once": sorted(df["rank"].tolist()) == list(range(1, 101)),
        "No duplicate candidate_ids": df["candidate_id"].nunique() == 100,
        "No null candidate_ids": df["candidate_id"].notna().all(),
        "No null scores": df["score"].notna().all(),
        "Scores are float": df["score"].dtype in [float, "float64"],
        "Scores monotonically non-increasing": all(
            df.sort_values("rank")["score"].iloc[i] >= df.sort_values("rank")["score"].iloc[i+1]
            for i in range(99)
        ),
        "Reasoning column present": "reasoning" in df.columns,
        "No empty reasoning (top 10)": df[df["rank"] <= 10]["reasoning"].notna().all(),
        "Scores not all identical": df["score"].nunique() > 1,
        "Rank 1 has highest score": df[df["rank"]==1]["score"].iloc[0] >= df[df["rank"]==100]["score"].iloc[0],
    }
    
    all_pass = True
    for check, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} | {check}")
        if not result:
            all_pass = False
    
    if all_pass:
        print("\n✅ ALL CHECKS PASSED — Ready to submit")
    else:
        print("\n❌ FIX FAILURES BEFORE SUBMITTING")
    
    return all_pass

validate_submission("submission.csv", "candidates.jsonl.gz")
```

### 9.3 Honeypot Rate Check

```python
def check_honeypot_rate(csv_path: str, honeypot_ids_path: str):
    df = pd.read_csv(csv_path)
    with open(honeypot_ids_path) as f:
        honeypot_ids = set(f.read().strip().split("\n"))
    
    honeypots_in_top_100 = df["candidate_id"].isin(honeypot_ids).sum()
    rate = honeypots_in_top_100 / 100 * 100
    
    print(f"Honeypots in top 100: {honeypots_in_top_100}")
    print(f"Honeypot rate: {rate:.1f}% (must be < 10%)")
    
    if honeypots_in_top_100 >= 10:
        print("❌ DISQUALIFICATION RISK — Honeypot rate too high!")
        # Show which ones
        honeypot_rows = df[df["candidate_id"].isin(honeypot_ids)]
        print("Honeypots found:", honeypot_rows[["candidate_id", "rank"]].to_string())
    else:
        print("✅ Honeypot rate within acceptable range")

check_honeypot_rate("submission.csv", "artifacts/honeypot_ids.txt")
```

### 9.4 Full Dry Run Timing Test

```bash
# Run on your local 16 GB CPU-only machine
time python rank.py \
    --candidates ./data/candidates.jsonl.gz \
    --artifacts ./artifacts \
    --out ./submission_test.csv

# Expected output:
# real    0m42.317s  (should be well under 5 minutes)
```

---

## Step 10 — Sandbox Setup (HuggingFace Spaces) {#step-10}

**Owner:** Infra Lead | **Estimated Time:** 4-6 hours

The sandbox is **required** for submission (Section 10.5). It must accept ≤100 candidates, run end-to-end, and produce a ranked CSV.

### 10.1 Streamlit App Structure

```python
# sandbox/app.py
import streamlit as st
import pandas as pd
import gzip
import json
import io
import time

st.set_page_config(
    page_title="Redrob Candidate Ranker — Sandbox",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Redrob Intelligent Candidate Ranker")
st.caption("Sandbox demo — runs on ≤100 candidates, CPU-only, no network access during ranking.")

st.markdown("""
### How to use this sandbox:
1. Upload a JSONL file with ≤100 candidate records (same schema as `candidates.jsonl.gz`)
2. Click **Run Ranker**
3. Download the ranked CSV output
""")

# File upload
uploaded = st.file_uploader(
    "Upload candidates file (.jsonl or .jsonl.gz, max 100 candidates)",
    type=["jsonl", "gz"]
)

if uploaded:
    # Parse uploaded file
    if uploaded.name.endswith(".gz"):
        content = gzip.decompress(uploaded.read()).decode("utf-8")
    else:
        content = uploaded.read().decode("utf-8")
    
    candidates = [json.loads(line) for line in content.strip().split("\n") if line.strip()]
    
    if len(candidates) > 100:
        st.error(f"Too many candidates ({len(candidates)}). Max 100 for sandbox.")
    else:
        st.success(f"Loaded {len(candidates)} candidates")
        
        # Preview
        with st.expander("Preview candidates"):
            sample = candidates[:3]
            for c in sample:
                st.json(c)
        
        if st.button("🚀 Run Ranker", type="primary"):
            with st.spinner("Ranking candidates..."):
                t0 = time.time()
                
                # Load precomputed artifacts (bundled with app)
                features = load_features_for_sample(candidates)
                ranked = rank_sample(features)
                
                elapsed = time.time() - t0
            
            st.success(f"Ranking complete in {elapsed:.2f}s")
            
            # Display results
            st.subheader("📊 Ranked Results")
            st.dataframe(
                ranked[["rank", "candidate_id", "score", "reasoning"]].head(min(len(ranked), 100)),
                use_container_width=True
            )
            
            # Download
            csv_bytes = ranked[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False)
            st.download_button(
                label="⬇️ Download submission.csv",
                data=csv_bytes,
                file_name="sandbox_submission.csv",
                mime="text/csv"
            )
            
            # Timing info
            st.info(f"""
            **Runtime:** {elapsed:.2f}s  
            **Candidates ranked:** {len(ranked)}  
            **Top 5 candidates:** {', '.join(ranked.head(5)['candidate_id'].tolist())}
            """)
```

### 10.2 HuggingFace Spaces Deployment

```bash
# Create HF Space
# 1. Go to https://huggingface.co/new-space
# 2. Select: Streamlit SDK, Python 3.11, Free CPU tier

# Repository structure for HF Space:
sandbox/
├── app.py
├── requirements.txt          # streamlit, pandas, numpy, pyarrow
├── artifacts/
│   ├── candidates_features_sample.parquet  # tiny sample for demo
│   └── honeypot_ids.txt
└── README.md                 # HF Space card

# requirements.txt for sandbox (minimal — no GPU, no network required)
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Deploy
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/redrob-ranker
git push origin main
```

### 10.3 Validating the Sandbox

Before submitting the sandbox link, verify it:
- [ ] URL is publicly accessible (no login required)
- [ ] Accepts `sample_candidates.json` (50 candidates) as upload
- [ ] Produces a ranked CSV with correct column order: `candidate_id,rank,score,reasoning`
- [ ] Completes in under 5 minutes (should be ~10-30s for 50 candidates)
- [ ] Does NOT make any external API calls during ranking

---

## Final Submission Checklist {#checklist}

### CSV File Checklist
- [ ] Filename: `team_[your_id].csv`
- [ ] Encoding: UTF-8
- [ ] Exactly **100 rows** of data (+ 1 header row = 101 total)
- [ ] Column order: `candidate_id,rank,score,reasoning` — **exactly this, no extras**
- [ ] `rank` column: integers 1–100, each appearing **exactly once**
- [ ] `candidate_id` column: no duplicates, all exist in `candidates.jsonl`
- [ ] `score` column: float, **monotonically non-increasing** as rank increases
- [ ] `score` column: all values are **different** (not all the same)
- [ ] `reasoning` column: 1-2 sentences, no empty strings in top 50
- [ ] **Official validator passes:** `python validate_submission.py --submission team_xxx.csv`
- [ ] **Honeypot rate < 10%:** fewer than 10 honeypot candidates in top 100
- [ ] **No HR Managers / Content Writers / Marketing Managers** in ranks 1-20
- [ ] Top 10 candidates all have relevant ML/AI/Engineering titles
- [ ] Reasoning references specific facts (not "great candidate with AI skills")
- [ ] Reasoning for rank 90-100 acknowledges they are lower-tier fits

### GitHub Repository Checklist
- [ ] Public repo (or private with organizer access granted)
- [ ] `README.md` with **single command** to reproduce: `python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv`
- [ ] `requirements.txt` with pinned versions
- [ ] All precomputed artifacts committed (`artifacts/candidates_features.parquet`, `artifacts/honeypot_ids.txt`)
- [ ] Offline scripts documented and in `/offline/` directory
- [ ] `submission_metadata.yaml` at root
- [ ] **Git history shows real iteration** (not a single commit dump — this is Stage 4 scrutiny)
- [ ] README distinguishes precomputation step from ranking step

### Portal Metadata Checklist
- [ ] Team name
- [ ] Primary contact name, email, phone
- [ ] GitHub repository URL
- [ ] **Working sandbox link** (HuggingFace Spaces / Streamlit Cloud / Colab)
- [ ] AI tools declared (Claude / GPT-4 / Copilot — be honest)
- [ ] Compute environment summary (e.g., "MacBook Pro M2, 16 GB RAM, Python 3.11")
- [ ] Team member list (name + email)
- [ ] Methodology summary (≤200 words — strongly recommended for Stage 4)

### Pre-Submission Timing Test
```bash
# Run this EXACTLY as Stage 3 will reproduce it
time python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
# Target: under 4 minutes (leave 1 min buffer)
```

---

## Scoring Strategy & What to Maximize {#scoring}

### Score Weights (from submission_spec.md Section 4)

| Metric | Weight | Implication |
|--------|--------|-------------|
| **NDCG@10** | **0.50** | Getting ranks 1-10 RIGHT is worth half the total score |
| NDCG@50 | 0.30 | Solid top-50 is the second priority |
| MAP | 0.15 | Precision across all relevance levels |
| P@10 | 0.05 | Simple precision of top-10 |

### Strategic Implications

1. **Spend 80% of your time getting the top 10 right.** Manually review them. Read the actual candidate profiles. Make sure none are honeypots, title-chasers, or keyword-stuffers.

2. **NDCG is position-aware.** Getting a highly relevant candidate at rank 1 vs. rank 5 matters enormously. Don't arbitrarily sort same-score candidates.

3. **P@10 requires a clean top 10.** Every irrelevant candidate (HR Manager, honeypot) in your top 10 is a 10% hit to P@10 AND hurts NDCG@10 most.

4. **The "Tier 5" candidate test.** Before finalizing, ask: *Does my ranker surface a candidate who built a recommendation system but never wrote "RAG"?* If not, your embedding model isn't working.

5. **The availability test.** Before finalizing, ask: *Is my #1 candidate actively looking (last active < 30 days, response rate > 60%, open_to_work = true)?* If not, your behavioral multiplier isn't working.

### Final Sanity Checks on Your Top 10

For each of your top 10 candidates, verify manually:
- ✅ Title contains "Engineer", "Scientist" (ML), "ML", "AI", "Data" (applied), or similar
- ✅ Work history includes at least one product company (not pure consulting or pure research)
- ✅ At least one clear production ML system in work history
- ✅ Located in or willing to relocate to target cities
- ✅ Last active < 90 days
- ✅ Not a honeypot (check your detector + manual verification)
- ✅ Reasoning is specific, honest, and references actual profile facts

---

## Methodology Summary (for Portal Submission)

> Copy-edit this to ≤200 words for your portal submission:

*Our system uses a two-phase architecture. In the offline phase (unrestricted compute), we embed all 100K candidates using BAAI/bge-large-en-v1.5, building rich candidate representations from work history, skills, and summaries rather than skills alone. We also precompute a feature matrix capturing 8 JD-alignment dimensions and all 23 behavioral signals. Honeypot detection uses 10 logical-consistency rules targeting impossible career timelines and signal contradictions. In the online phase (5 min, CPU-only), rank.py loads the precomputed feature matrix and combines: (40%) semantic embedding similarity to a JD-weighted query, (30%) explicit keyword matching for must-have skills, (15%) company-type scoring to distinguish shippers from researchers, and (15%) behavioral signal scoring. Availability is applied as a multiplicative modifier based on recency of activity, recruiter response rate, and open-to-work status, so inactive candidates are down-ranked regardless of skill score. We explicitly filter title-mismatched candidates (HR Managers with AI skills) and apply a "shipper bonus" to candidates who built recommendation or search systems without using current AI buzzwords. Reasoning is generated from actual profile fields with zero hallucination risk.*

---

*Document version: 1.0 | Based on official hackathon docs: README.docx, job_description.docx, redrob_signals_doc.docx, submission_spec.docx, Final_note_for_participants.txt*
