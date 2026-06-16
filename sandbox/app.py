import streamlit as st
import pandas as pd
import gzip
import json
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

uploaded = st.file_uploader(
    "Upload candidates file (.jsonl or .jsonl.gz, max 100 candidates)",
    type=["jsonl", "gz"]
)

# Dummy implementations for missing sandbox functions mentioned in workflow doc
def load_features_for_sample(candidates):
    # normally this would load artifacts/candidates_features_sample.parquet
    return pd.DataFrame() 

def rank_sample(features):
    # normally this would use rank.py logic
    return pd.DataFrame({"rank": [], "candidate_id": [], "score": [], "reasoning": []})

if uploaded:
    if uploaded.name.endswith(".gz"):
        content = gzip.decompress(uploaded.read()).decode("utf-8")
    else:
        content = uploaded.read().decode("utf-8")
    
    candidates = [json.loads(line) for line in content.strip().split("\n") if line.strip()]
    
    if len(candidates) > 100:
        st.error(f"Too many candidates ({len(candidates)}). Max 100 for sandbox.")
    else:
        st.success(f"Loaded {len(candidates)} candidates")
        
        with st.expander("Preview candidates"):
            sample = candidates[:3]
            for c in sample:
                st.json(c)
        
        if st.button("🚀 Run Ranker", type="primary"):
            with st.spinner("Ranking candidates..."):
                t0 = time.time()
                features = load_features_for_sample(candidates)
                ranked = rank_sample(features)
                elapsed = time.time() - t0
            
            st.success(f"Ranking complete in {elapsed:.2f}s")
            
            st.subheader("📊 Ranked Results")
            st.dataframe(
                ranked[["rank", "candidate_id", "score", "reasoning"]].head(min(len(ranked), 100)),
                use_container_width=True
            )
            
            csv_bytes = ranked[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False)
            st.download_button(
                label="⬇️ Download submission.csv",
                data=csv_bytes,
                file_name="sandbox_submission.csv",
                mime="text/csv"
            )
            
            st.info(f"""
            **Runtime:** {elapsed:.2f}s  
            **Candidates ranked:** {len(ranked)}  
            """)
