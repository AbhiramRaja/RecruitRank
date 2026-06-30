"""
Stage 2 — Feature Extraction

Runs: Offline pre-computation. Output saved to artifacts/features.parquet.
Input: Filtered candidates (from Stage 1)
Output: Pandas DataFrame with candidate_id and computed feature scores.
"""

import json
import os
import pandas as pd
from typing import List, Dict

# Need this constant from Stage 1 or re-defined here
CONSULTING_INDUSTRIES = {"IT Services", "Consulting", "Outsourcing", "BPO", "KPO", "Staffing"}

def extract_features(candidates: List[Dict], jd_parsed: Dict) -> pd.DataFrame:
    """
    Compute feature scores for each candidate:
    - title_career_score
    - skills_score
    - experience_score
    - location_edu_score
    
    Returns a DataFrame containing these features.
    """
    
    required_skills = jd_parsed.get("required_skills", [])
    nice_to_have = jd_parsed.get("nice_to_have_skills", [])
    
    features = []
    
    for candidate in candidates:
        cid = candidate["candidate_id"]
        profile = candidate.get("profile", {})
        career_history = candidate.get("career_history", [])
        skills = candidate.get("skills", [])
        education = candidate.get("education", [])
        redrob_signals = candidate.get("redrob_signals", {})
        
        # ---------------------------------------------------------
        # 2A. Title + Career Score (range 0.0 - 1.0)
        # ---------------------------------------------------------
        tcs = 0.0
        TARGET_TITLES = [
            "machine learning engineer", "ml engineer", "ai engineer",
            "data scientist", "research engineer", "applied scientist",
            "software engineer", "backend engineer", "platform engineer",
            "mlops engineer", "llm engineer", "nlp engineer"
        ]
        title_lower = profile.get("current_title", "").lower()
        if any(t in title_lower for t in TARGET_TITLES):
            tcs += 0.4
            
        product_months = sum(
            entry.get("duration_months", 0)
            for entry in career_history
            if entry.get("company_size") in ("201-500", "501-1000", "1001-5000", "5001-10000", "10001+")
            and entry.get("industry") not in CONSULTING_INDUSTRIES
        )
        tcs += min(product_months / 60.0, 1.0) * 0.4
        
        SENIOR_KEYWORDS = ["senior", "lead", "principal", "staff", "head of", "director"]
        if any(k in title_lower for k in SENIOR_KEYWORDS):
            tcs += 0.2
            
        title_career_score = min(tcs, 1.0)
        
        # ---------------------------------------------------------
        # 2B. Skills Score (range 0.0 - 1.0)
        # ---------------------------------------------------------
        candidate_skill_names = {s.get("name", "").lower() for s in skills if s.get("name")}
        
        required_matched = sum(1 for rs in required_skills if rs.lower() in candidate_skill_names)
        required_score = required_matched / len(required_skills) if required_skills else 0.0
        
        nth_matched = sum(1 for nth in nice_to_have if nth.lower() in candidate_skill_names)
        nth_score = nth_matched / len(nice_to_have) if nice_to_have else 0.0
        
        if skills:
            avg_endorsements = sum(s.get("endorsements", 0) for s in skills) / len(skills)
        else:
            avg_endorsements = 0
            
        endorsement_trust = min(avg_endorsements / 20.0, 1.0)
        
        raw_skills_score = (3.0 * required_score + 1.0 * nth_score) / 4.0
        skills_score = raw_skills_score * (0.7 + 0.3 * endorsement_trust)
        
        # ---------------------------------------------------------
        # 2C. Experience Score (range 0.0 - 1.0)
        # ---------------------------------------------------------
        product_yoe = sum(
            entry.get("duration_months", 0) / 12.0
            for entry in career_history
            if entry.get("industry") not in CONSULTING_INDUSTRIES
            and entry.get("company_size") not in ("1-10", "11-50")
        )
        
        if product_yoe < 2:
            es = 0.1
        elif product_yoe < 4:
            es = 0.1 + (product_yoe - 2) / 2.0 * 0.4
        elif product_yoe <= 10:
            es = 0.5 + (product_yoe - 4) / 6.0 * 0.5
        else:
            es = 1.0 - min((product_yoe - 10) / 10.0, 0.2)
            
        experience_score = es
        
        # ---------------------------------------------------------
        # 2D. Location + Education Score (range 0.0 - 1.0)
        # ---------------------------------------------------------
        les = 0.0
        TARGET_CITIES = ["pune", "noida", "gurugram", "gurgaon", "delhi", "ncr",
                         "bangalore", "bengaluru", "hyderabad", "mumbai", "chennai"]
        loc_lower = profile.get("location", "").lower()
        
        if any(city in loc_lower for city in TARGET_CITIES):
            les += 0.5
        elif redrob_signals.get("willing_to_relocate", False):
            les += 0.3
            
        if education:
            tiers = [edu.get("tier", "unknown") for edu in education if edu.get("tier")]
            best_tier = min(tiers) if tiers else "unknown"
        else:
            best_tier = "unknown"
            
        if best_tier == "tier_1":
            les += 0.5
        elif best_tier == "tier_2":
            les += 0.3
        else:
            les += 0.1
            
        location_edu_score = min(les, 1.0)
        
        # Append candidate features
        features.append({
            "candidate_id": cid,
            "title_career_score": title_career_score,
            "skills_score": skills_score,
            "experience_score": experience_score,
            "location_edu_score": location_edu_score
        })
        
    return pd.DataFrame(features)


def save_features(df: pd.DataFrame, artifacts_dir: str = "artifacts") -> str:
    """Save the features DataFrame to a parquet file."""
    os.makedirs(artifacts_dir, exist_ok=True)
    output_path = os.path.join(artifacts_dir, "features.parquet")
    
    df.to_parquet(output_path, index=False)
    print(f"[Stage 2] Saved features.parquet -> {output_path} ({len(df)} rows)")
    return output_path

# ---------------------------------------------------------------------------
# Simple test if run standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from pipeline.stage1_filters import apply_hard_filters, apply_honeypot_gate
    
    print(f"[{time.strftime('%H:%M:%S')}] Stage 2: Feature Extraction — start")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jd_path = os.path.join(project_root, "artifacts", "jd_parsed.json")
    sample_path = os.path.join(project_root, "sample_candidates.json")
    
    if os.path.exists(jd_path) and os.path.exists(sample_path):
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_parsed = json.load(f)
            
        with open(sample_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
            
        # Run stage 1 to get filtered candidates
        filtered = apply_hard_filters(candidates, jd_parsed)
        # Stage 2 should run on surviving candidates (we can include honeypots for precomputation, but typically we'd just precompute for all surviving pass A)
        
        df = extract_features(filtered, jd_parsed)
        
        print("\nFeature Summary:")
        print(df.describe())
        
        save_features(df, artifacts_dir=os.path.join(project_root, "artifacts"))
        
    print(f"[{time.strftime('%H:%M:%S')}] Stage 2: Feature Extraction — complete")
