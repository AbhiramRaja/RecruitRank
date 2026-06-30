"""
Stage 1 — Hard Filters + Honeypot Gate

Runs: Inline at ranking time (first step in rank.py).
Input: List of candidate dictionaries (up to 100K).
Output: Filtered list of candidate dictionaries with honeypot_flag added.
"""

import datetime
import json
import math
from typing import List, Dict, Any

# Fixed date for reproducible "TODAY" calculations (matches dataset timeframe)
COMPETITION_DATE = datetime.date(2026, 6, 30)

CONSULTING_INDUSTRIES = {"IT Services", "Consulting", "Outsourcing", "BPO", "KPO", "Staffing"}

WRONG_DOMAIN_TITLES = [
    "marketing", "sales", "accountant", "civil engineer", "mechanical engineer",
    "graphic designer", "hr manager", "customer support", "content writer",
    "operations manager", "project manager (non-tech)"
]

FIELD_DEFAULTS = {
    "years_of_experience": 0,
    "duration_months": 0,
    "endorsements": 0,
    "interview_completion_rate": 0.5,
    "recruiter_response_rate": 0.5,
    "offer_acceptance_rate": -1,
    "github_activity_score": -1,
    "notice_period_days": 60,
}

def parse_date(date_str: str) -> datetime.date:
    """Parse YYYY-MM-DD date string to datetime.date object."""
    if not date_str:
        return COMPETITION_DATE
    try:
        # Handles YYYY-MM-DD
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

def get_field(obj: Dict, key: str, default: Any = None) -> Any:
    """Safely get a field, falling back to FIELD_DEFAULTS or provided default."""
    val = obj.get(key)
    if val is None:
        return FIELD_DEFAULTS.get(key, default)
    return val


def apply_hard_filters(candidates: List[Dict], jd_parsed: Dict) -> List[Dict]:
    """
    Pass A: Hard Filters (discard entirely)
    Remove candidate from consideration if ANY of the hard filters are true.
    """
    filtered_candidates = []
    
    # Combine hardcoded wrong titles with wrong_domains from jd_parsed
    all_wrong_domains = WRONG_DOMAIN_TITLES.copy()
    if "hard_disqualifiers" in jd_parsed and "wrong_domains" in jd_parsed["hard_disqualifiers"]:
        all_wrong_domains.extend([d.lower() for d in jd_parsed["hard_disqualifiers"]["wrong_domains"]])

    # Build a tech skills set based on jd_parsed for the tech skills check
    tech_skills_list = set(
        [s.lower() for s in jd_parsed.get("required_skills", [])] +
        [s.lower() for s in jd_parsed.get("nice_to_have_skills", [])]
    )

    for candidate in candidates:
        profile = candidate.get("profile", {})
        career_history = candidate.get("career_history", [])
        skills = candidate.get("skills", [])
        
        # 1. Wrong country
        country = profile.get("country", "")
        # Fallback to checking location if country is empty
        location = profile.get("location", "").lower()
        if country != "India" and not any(city in location for city in ["pune", "noida", "gurugram", "gurgaon", "delhi", "ncr", "bangalore", "bengaluru", "hyderabad", "mumbai", "chennai"]):
            if country != "India": # Strictly adhere to the spec: profile.country != "India"
                continue

        # 2. Consulting-only career
        if career_history:
            all_consulting = all(entry.get("industry", "") in CONSULTING_INDUSTRIES for entry in career_history)
            if all_consulting:
                continue
        
        # 3. Wrong domain (title clearly outside scope)
        current_title_lower = profile.get("current_title", "").lower()
        if any(wrong_title in current_title_lower for wrong_title in all_wrong_domains):
            continue
            
        # 4. Zero technical skills
        # We consider tech skills to be those listed in the JD (required + nice-to-have)
        # If candidate has NO skills that match the tech_skills_list, they are discarded.
        # However, to avoid discarding too aggressively if tech_skills_list is small,
        # we check if any of their skills are in the tech_skills_list. 
        # (Assuming the JD covers a broad enough base of tech skills).
        matched_tech_skills = [s for s in skills if s.get("name", "").lower() in tech_skills_list]
        if len(matched_tech_skills) == 0:
            continue

        filtered_candidates.append(candidate)
        
    return filtered_candidates


def apply_honeypot_gate(candidates: List[Dict]) -> List[Dict]:
    """
    Pass B: Honeypot Detection (zero the score, keep in dataset)
    Flag a candidate as a honeypot (set honeypot_flag = True, final_score = 0.0) if ANY rules are violated.
    """
    for candidate in candidates:
        candidate["honeypot_flag"] = False
        
        profile = candidate.get("profile", {})
        career_history = candidate.get("career_history", [])
        education = candidate.get("education", [])
        skills = candidate.get("skills", [])
        redrob_signals = candidate.get("redrob_signals", {})
        
        claimed_yoe = get_field(profile, "years_of_experience")
        
        # 1. Impossible experience: claimed YOE > actual career span
        if career_history:
            earliest_start = min([parse_date(entry.get("start_date")) for entry in career_history if entry.get("start_date")])
            actual_span_years = (COMPETITION_DATE - earliest_start).days / 365.25
            if claimed_yoe > actual_span_years + 2:
                candidate["honeypot_flag"] = True
                candidate["final_score"] = 0.0
                continue
                
        # 2. Impossible graduation: working before degree completed
        if education and career_history:
            grad_years = [edu.get("end_year") for edu in education if edu.get("end_year")]
            job_years = [parse_date(entry.get("start_date")).year for entry in career_history if entry.get("start_date")]
            
            if grad_years and job_years:
                grad_year = min(grad_years)
                earliest_job_year = min(job_years)
                if earliest_job_year < grad_year - 1:
                    candidate["honeypot_flag"] = True
                    candidate["final_score"] = 0.0
                    continue
                    
        # 3 & 4. Skill checks
        skill_flagged = False
        assessment_scores = redrob_signals.get("skill_assessment_scores", {})
        for skill in skills:
            proficiency = skill.get("proficiency", "").lower()
            duration_months = get_field(skill, "duration_months")
            
            # 3. Expert skill with 0-month duration
            if proficiency == "expert" and duration_months == 0:
                skill_flagged = True
                break
                
            # 4. Expert/advanced claim with very low assessment score
            skill_name = skill.get("name", "")
            assessment = assessment_scores.get(skill_name)
            if assessment is not None:
                if proficiency in ("expert", "advanced") and assessment < 25:
                    skill_flagged = True
                    break
                    
        if skill_flagged:
            candidate["honeypot_flag"] = True
            candidate["final_score"] = 0.0
            continue
            
        # 5. Claimed YOE exceeds biological maximum (working age check)
        if claimed_yoe > 40:
            candidate["honeypot_flag"] = True
            candidate["final_score"] = 0.0
            continue
            
    return candidates

# ---------------------------------------------------------------------------
# Simple test if run standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jd_path = os.path.join(project_root, "artifacts", "jd_parsed.json")
    sample_path = os.path.join(project_root, "sample_candidates.json")
    
    if os.path.exists(jd_path) and os.path.exists(sample_path):
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_parsed = json.load(f)
            
        with open(sample_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
            
        print(f"Total candidates before Pass A: {len(candidates)}")
        filtered = apply_hard_filters(candidates, jd_parsed)
        print(f"Candidates remaining after Pass A: {len(filtered)}")
        
        gated = apply_honeypot_gate(filtered)
        honeypot_count = sum(1 for c in gated if c.get("honeypot_flag", False))
        print(f"Honeypots detected in Pass B: {honeypot_count}")
        print(f"Valid candidates remaining: {len(gated) - honeypot_count}")
