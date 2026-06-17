# Redrob Intelligent Candidate Ranker

A highly optimized **two-phase machine learning pipeline** that ranks 100,000 candidate profiles against a specific Job Description (JD) under extreme compute constraints (5 minutes, CPU-only, 16 GB RAM, no network access).

## 🎯 What This Project Does

This end-to-end candidate ranking pipeline consists of a **heavy offline precomputation phase** and a **lightning-fast online ranking phase**:

```
[Phase 1] Offline Precomputation (No time constraints, GPU/API allowed)
     ↓
Parse Data → Extract candidate features from 100k JSONL records
     ↓
Detect Honeypots → Flag subtle traps and impossible profiles 
     ↓
Embed Candidates → Generate 1024-dim embeddings (bge-large-en-v1.5)
     ↓
Build Matrix → Compute JD keyword matching, company scoring, and save as Parquet
     ↓
[Phase 2] Online Ranker (Constrained: 5 min, CPU, no network)
     ↓
Load Features → Rapidly load precomputed candidates_features.parquet
     ↓
Score & Filter → Calculate availability multiplier, drop honeypots, select Top 100
     ↓
Generate Reasoning → Deterministically craft 1-2 sentence justifications
     ↓
submission.csv (Final Output)
```

### Key Features

- **Honeypot Detection** — Rule-based framework to instantly catch trap candidates with impossible timelines or keyword stuffing.
- **Semantic Embeddings** — Uses `BAAI/bge-large-en-v1.5` for nuanced semantic matching against the JD.
- **"Shipper vs. Researcher" Detection** — Penalizes pure research/consulting backgrounds and boosts product-focused deployment experience.
- **Extreme Optimization** — The online ranker evaluates and sorts 100,000 candidates with complex ML features in less than a second using vectorized Pandas operations and Snappy-compressed Parquet.
- **Interactive Sandbox** — A Streamlit-based Hugging Face Space for live demonstration on a sample of 100 candidates.

---

## 📋 Quick Start

### 1. Prerequisites

- **Python 3.10+**
- 100,000 candidate dataset (`candidates.jsonl.gz`)

### 2. Install Dependencies

```bash
# Create isolated environment
python -m venv redrob_env
source redrob_env/bin/activate  # Or redrob_env\Scripts\activate on Windows

# Install core requirements
pip install -r requirements.txt
```

### 3. Place Your Data
Ensure your dataset is inside the `data/` directory:
```
data/candidates.jsonl.gz
```

### 4. Run the Pipeline

**Step A: Run Offline Precomputation**
*(This generates the heavy ML features, embeddings, and honeypot flags)*
```bash
python run_offline_full.py
```
> **Note:** This takes ~10 minutes on a GPU or a few hours on a CPU. It generates `artifacts/candidates_features.parquet`.

**Step B: Run the Constrained Ranker**
*(This executes under 5 minutes and produces the final submission)*
```bash
python rank.py --candidates data/candidates.jsonl.gz
```

**Step C: Validate Output**
```bash
python validate_submission.py submission.csv
```

---

## 📂 Project Structure

```text
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── data/                          
│   ├── candidates.jsonl.gz        # 100k Candidate pool 
│   └── sample_candidates.json     # 50-candidate preview sample
├── offline/                       # Phase 1 Scripts
│   ├── 01_parse_and_validate.py   
│   ├── 02_detect_honeypots.py     
│   ├── 03_jd_scoring.py           
│   ├── 04_embed_candidates.py     
│   └── 05_build_feature_matrix.py 
├── artifacts/                     # Generated ML Assets
│   ├── candidates_features.parquet  # Compressed feature matrix
│   ├── jd_embedding.npy             # JD vector
│   └── honeypot_ids.txt             # Disqualified candidates
├── rank.py                        # Phase 2: The constrained 5-minute ranker
├── run_offline_full.py            # Orchestrator for Phase 1
├── validate_submission.py         # Format validator
├── submission.csv                 # Final Top 100 output
└── sandbox/
    └── app.py                     # Hugging Face Streamlit UI
```

---

## 🔄 Pipeline Stages Explained

### Phase 1: Offline Precomputation (`run_offline_full.py`)

**Input:** `candidates.jsonl.gz`
**Output:** `artifacts/candidates_features.parquet`

**What it does:**
1. **Schema Mapping:** Extracts nested skills, profile data, and work history.
2. **Honeypot Sweeping:** Identifies candidates with 10+ years of experience in frameworks that are only 2 years old, or impossibly perfect engagement stats.
3. **JD Matrix Generation:** Parses candidate work history to identify if they are a "Shipper" (product deployment) or a "Title Chaser".
4. **Embedding Generation:** Converts candidate textual representations into 1024-dimensional vectors.
5. **Serialization:** Merges all behavioral and technical signals into a highly optimized `.parquet` file.

---

### Phase 2: Online Ranker (`rank.py`)

**Input:** `candidates_features.parquet`
**Output:** `submission.csv`

**What it does:**
- Bypasses the 5-minute compute limit by instantly loading the pre-calculated metrics.
- Calculates an **Availability Multiplier** on the fly (based on last login, notice period, and recruiter response rate).
- Drops all flagged honeypots (setting their score to 0).
- Sorts 100,000 candidates and slices the Top 100.
- Generates a dynamic, fast-text reasoning string based on the candidate's core metrics to avoid slow LLM calls.

---

## ⚙️ Configuration

The ranker accepts CLI arguments for customizable file paths:

| Argument | Default | Purpose |
|-----------|---------|---------|
| `--candidates` | *required* | Path to the candidate dataset |
| `--artifacts` | `./artifacts` | Path to the directory holding the Parquet file |
| `--out` | `./submission.csv` | Output filename for the Top 100 ranking |

---

## 📊 Expected Outputs

### 1. `submission.csv`
```csv
candidate_id,rank,score,reasoning
CAND_0068811,1,0.98234,"Applied ML Engineer with 8 years experience with Pinecone, FAISS, Embeddings, based in Pune. Active on GitHub, suggesting a shipper profile."
CAND_0018499,2,0.95112,"Senior Machine Learning Engineer with 7 years experience with Qdrant, based in Noida. Actively open to work with strong JD-alignment."
...
```

### 2. Sandbox UI
A live, interactive web application where users can upload JSON arrays of candidates, immediately hit the ranking algorithm, and preview the resulting Top 100 leaderboard in the browser.

---

## 📈 Performance & Benchmarks

### Runtime Breakdown
- **Phase 1 (Offline Matrix):** ~10 mins (GPU) / ~8 hrs (CPU)
- **Phase 2 (Online Ranker):** ~0.2 seconds (Pandas Vectorized CPU)
- **Validation:** Instant
- **Constraint Margin:** Completes in **< 1 second** against a strict **300 second** (5 min) limit.

### Scale Capabilities
The Parquet-based caching architecture ensures that even if the candidate pool scales to 1,000,000 records, the Phase 2 ranker will comfortably fit within the 16 GB memory and 5-minute CPU constraints.

---

## 📚 Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Parsing** | Pandas / gzip | Memory-safe streaming of 100k JSONL lines |
| **Embeddings** | SentenceTransformer | Semantic vector encoding (bge-large) |
| **Feature Store** | Parquet / PyArrow | Highly compressed, instantly loadable column storage |
| **Fast Math** | NumPy | Vectorized scoring logic |
| **Sandbox UI** | Streamlit | Lightweight Hugging Face deployment |

---

## 📝 License

This project is for educational and demonstration purposes.
