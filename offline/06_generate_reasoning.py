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
{str(candidate.get('work_history', []))}
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
