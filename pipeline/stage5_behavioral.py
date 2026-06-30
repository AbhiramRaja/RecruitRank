"""
Stage 5 — Behavioral Signal Modifier

Runs: Inline in rank.py.
Design: Multiplicative modifier, not additive. Clips to [0.4, 1.2].
final_base_score = base_score * clamp(behavioral_modifier, 0.4, 1.2)
"""

import pandas as pd
from typing import List, Dict
import datetime

# Fixed date for reproducible "TODAY" calculations (matches dataset timeframe)
COMPETITION_DATE = datetime.date(2026, 6, 30)

def parse_date(date_str: str) -> datetime.date:
    """Parse YYYY-MM-DD date string to datetime.date object."""
    if not date_str:
        return COMPETITION_DATE
    try:
        parts = date_str.split('T')[0].split('-')
        if len(parts) == 3:
            return datetime.date(int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            return datetime.date(int(parts[0]), int(parts[1]), 1)
        elif len(parts) == 1:
            return datetime.date(int(parts[0]), 1, 1)
        return COMPETITION_DATE
    except Exception:
        return COMPETITION_DATE


def get_behavioral_modifier(signals: Dict) -> float:
    """
    Computes the behavioral modifier from redrob_signals.
    """
    score = 1.0  # neutral baseline

    # Availability (can they start soon?)
    last_active = parse_date(signals.get("last_active_date", ""))
    days_since_active = (COMPETITION_DATE - last_active).days
    
    if signals.get("open_to_work_flag", False):
        score += 0.08
    if signals.get("notice_period_days", 60) <= 30:
        score += 0.05
    if days_since_active <= 7:
        score += 0.05
    elif days_since_active <= 30:
        score += 0.02
    elif days_since_active > 90:
        score -= 0.10

    # Engagement quality
    recruiter_response_rate = signals.get("recruiter_response_rate", 0.5)
    if recruiter_response_rate >= 0.7:
        score += 0.06
    elif recruiter_response_rate <= 0.2:
        score -= 0.08
        
    if signals.get("avg_response_time_hours", 24) <= 4:
        score += 0.03

    # Recruiter interest (social proof)
    if signals.get("saved_by_recruiters_30d", 0) >= 5:
        score += 0.05
    if signals.get("search_appearance_30d", 0) >= 10:
        score += 0.03

    # Trust + reliability
    interview_completion_rate = signals.get("interview_completion_rate", 0.5)
    if interview_completion_rate >= 0.8:
        score += 0.06
    elif interview_completion_rate < 0.4:
        score -= 0.10
        
    if signals.get("verified_email", False) and signals.get("verified_phone", False):
        score += 0.04
        
    offer_acceptance_rate = signals.get("offer_acceptance_rate", -1)
    if offer_acceptance_rate > 0 and offer_acceptance_rate >= 0.8:
        score += 0.04

    # GitHub activity (engineering signal)
    github_activity_score = signals.get("github_activity_score", -1)
    if github_activity_score >= 70:
        score += 0.04
    elif github_activity_score == -1:
        pass  # no penalty for no GitHub

    # Hard clip between 0.4 and 1.2
    return max(0.4, min(1.2, score))


def apply_behavioral_modifier(scored_df: pd.DataFrame, candidates: List[Dict]) -> pd.DataFrame:
    """
    Applies the behavioral modifier to the base_score for each candidate.
    
    Args:
        scored_df: DataFrame with 'candidate_id' and 'base_score'.
        candidates: List of raw candidate dicts containing 'redrob_signals'.
        
    Returns:
        DataFrame with added 'behavioral_modifier' and 'final_base_score' columns.
    """
    # Create a mapping of candidate_id to their modifier
    modifiers = []
    for cand in candidates:
        cid = cand.get("candidate_id")
        signals = cand.get("redrob_signals", {})
        modifier = get_behavioral_modifier(signals)
        modifiers.append({
            "candidate_id": cid,
            "behavioral_modifier": modifier
        })
        
    mod_df = pd.DataFrame(modifiers)
    
    # Merge modifiers into scored_df
    df = scored_df.merge(mod_df, on="candidate_id", how="left")
    
    # Apply modifier (ensure missing values default to 1.0 just in case)
    df["behavioral_modifier"] = df["behavioral_modifier"].fillna(1.0)
    df["final_base_score"] = df["base_score"] * df["behavioral_modifier"]
    
    return df


# Simple standalone test
if __name__ == "__main__":
    import os
    import time
    import json
    
    print(f"[{time.strftime('%H:%M:%S')}] Stage 5: Behavioral Signal Modifier — start")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(project_root, "artifacts")
    sample_path = os.path.join(project_root, "sample_candidates.json")
    
    features_path = os.path.join(artifacts_dir, "features.parquet")
    career_path = os.path.join(artifacts_dir, "career_embeddings.npy")
    skills_path = os.path.join(artifacts_dir, "skills_embeddings.npy")
    jd_path = os.path.join(artifacts_dir, "jd_vector.npy")
    
    if os.path.exists(features_path) and os.path.exists(sample_path):
        import numpy as np
        from pipeline.stage4_scorer import compute_weighted_scores
        
        with open(sample_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
            
        features_df = pd.read_parquet(features_path)
        career_emb = np.load(career_path)
        skills_emb = np.load(skills_path)
        jd_vec = np.load(jd_path)
        
        with open(os.path.join(artifacts_dir, "candidate_ids.json"), "r") as f:
            cids = json.load(f)
            
        scored_df = compute_weighted_scores(features_df, career_emb, skills_emb, jd_vec, cids)
        
        final_df = apply_behavioral_modifier(scored_df, candidates)
        
        print("\nModifier Summary:")
        print(final_df[["base_score", "behavioral_modifier", "final_base_score"]].describe())
        
        print("\nTop 5 Candidates by Final Base Score:")
        top5 = final_df.sort_values("final_base_score", ascending=False).head(5)
        for _, row in top5.iterrows():
            print(f"  {row['candidate_id']}: base={row['base_score']:.4f} * mod={row['behavioral_modifier']:.4f} = final={row['final_base_score']:.4f}")
            
    else:
        print("Missing artifacts. Run Stages 1-4 first.")
        
    print(f"\n[{time.strftime('%H:%M:%S')}] Stage 5: Behavioral Signal Modifier — complete")
