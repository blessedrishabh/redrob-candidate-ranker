JD_ANALYSIS = {
    # TIER 1: INSTANT DISQUALIFIERS
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
    # TIER 2: MUST-HAVES
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
    # TIER 3: NICE-TO-HAVES
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
    # TIER 4: NEGATIVE SIGNALS
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
    # TIER 5: SHIPPER BONUS
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

COMPANY_TYPE_SIGNALS = {
    "product_positive": [
        "startup", "series a", "series b", "saas", "platform", 
    ],
    "research_negative": [
        "university", "iit", "iim", "nit", "iiser", "lab", "research center",
        "microsoft research", "google brain", "deepmind", "fair", "openai research"
    ],
    "consulting_disqualify": [
        "tcs", "tata consultancy", "infosys", "wipro", "accenture",
        "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis"
    ]
}

def score_company_profile(work_history: list) -> float:
    if not work_history:
        return 0.0
    
    consulting_count = 0
    research_count = 0
    product_count = 0
    
    CONSULTING_FIRMS = COMPANY_TYPE_SIGNALS["consulting_disqualify"]
    RESEARCH_ORGS = COMPANY_TYPE_SIGNALS["research_negative"]
    
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
    
    if consulting_count == total:
        return -1.0
    
    research_ratio = research_count / total
    product_ratio = product_count / total
    
    return product_ratio - (research_ratio * 0.5) - (consulting_count / total * 0.3)

LOCATION_WEIGHTS = {
    "pune": 1.0,
    "noida": 1.0,
    "delhi": 0.85,
    "delhi ncr": 0.85,
    "gurgaon": 0.85,
    "gurugram": 0.85,
    "mumbai": 0.75,
    "hyderabad": 0.75,
    "bangalore": 0.60,    
    "bengaluru": 0.60,
    "india": 0.50,        
    "remote": 0.40,       
}
