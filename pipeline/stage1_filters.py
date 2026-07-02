"""
Stage 1 — Hard Filters + Honeypot Gate

Runs: Inline at ranking time (first step in rank.py).
Input: List of candidate dictionaries (up to 100K).
Output: Filtered list of candidate dictionaries with honeypot_flag added.

Pass A: Hard Filters — discard candidates entirely.
Pass B: Honeypot Gate — zero scores of suspicious candidates (kept in list).
"""

import json
from typing import Any, Dict, List

from pipeline.constants import (
    COMPETITION_DATE,
    CONSULTING_INDUSTRIES,
    FIELD_DEFAULTS,
    WRONG_DOMAIN_TITLES,
)
from pipeline.utils import get_field, parse_date


# ---------------------------------------------------------------------------
# Pass A — Hard Filters (discard entirely)
# ---------------------------------------------------------------------------

def apply_hard_filters(candidates: List[Dict], jd_parsed: Dict) -> List[Dict]:
    """
    Remove candidates from consideration if ANY hard filter fires.

    Filters (architecture.md Stage 1 Pass A):
      1. country != "India"
      2. All career_history entries are in CONSULTING_INDUSTRIES
      3. Current title substring-matches a wrong-domain title
      4. Zero technical skills (no overlap with JD required + nice-to-have)

    Returns the surviving candidate list (discarded ones are gone for good).

    Monitoring (Risk 4 fix): prints per-filter drop counts so operators can
    detect over-aggressive filtering on production data.
    """
    # Build combined wrong-domain list: spec titles + any extra from JD
    all_wrong_domains: List[str] = list(WRONG_DOMAIN_TITLES)
    jd_wrong = jd_parsed.get("hard_disqualifiers", {}).get("wrong_domains", [])
    all_wrong_domains.extend(d.lower() for d in jd_wrong)

    # Tech-skills set = all JD required + nice-to-have skill names (lowercased)
    tech_skills_set = {
        s.lower()
        for s in (
            jd_parsed.get("required_skills", [])
            + jd_parsed.get("nice_to_have_skills", [])
        )
    }

    filtered: List[Dict] = []
    # Per-filter drop counters for monitoring (Risk 4)
    drops: Dict[str, int] = {
        "country": 0,
        "consulting_only": 0,
        "wrong_domain_title": 0,
        "zero_tech_skills": 0,
    }
    total_in = len(candidates)

    for candidate in candidates:
        profile = candidate.get("profile", {})
        career_history = candidate.get("career_history", [])
        skills = candidate.get("skills", [])

        # ------------------------------------------------------------------
        # Filter 1: Wrong country
        # Spec: candidate.profile.country != "India"  → discard.
        # Strictly per spec — no fallback to location string.
        # ------------------------------------------------------------------
        if profile.get("country", "") != "India":
            drops["country"] += 1
            continue

        # ------------------------------------------------------------------
        # Filter 2: Consulting-only career
        # Spec: ALL career_history entries have industry in CONSULTING_INDUSTRIES
        # Only applies when career_history is non-empty.
        # ------------------------------------------------------------------
        if career_history and all(
            entry.get("industry", "") in CONSULTING_INDUSTRIES
            for entry in career_history
        ):
            drops["consulting_only"] += 1
            continue

        # ------------------------------------------------------------------
        # Filter 3: Wrong domain title
        # Spec: substring match on current_title.lower().
        # Ambiguous titles (e.g. "Project Manager - ML Platform") are kept.
        # ------------------------------------------------------------------
        current_title_lower = profile.get("current_title", "").lower()
        if any(wrong in current_title_lower for wrong in all_wrong_domains):
            drops["wrong_domain_title"] += 1
            continue

        # ------------------------------------------------------------------
        # Filter 4: Zero technical skills
        # Discard if candidate has no skills in the JD's tech-skills set.
        # ------------------------------------------------------------------
        matched_tech = [
            s for s in skills if s.get("name", "").lower() in tech_skills_set
        ]
        if len(matched_tech) == 0:
            drops["zero_tech_skills"] += 1
            continue

        filtered.append(candidate)

    total_out = len(filtered)
    total_dropped = total_in - total_out
    # --- Monitoring: print per-filter breakdown (Risk 4 fix) ---
    print(
        f"[Stage 1 Pass A] {total_in:,} in -> {total_out:,} out "
        f"({total_dropped:,} dropped, {total_out/max(total_in,1)*100:.1f}% pass rate)."
    )
    for fname, count in drops.items():
        pct = count / max(total_in, 1) * 100
        flag = " [HIGH]" if pct > 40 else ""
        print(f"  Filter '{fname}': removed {count:,} ({pct:.1f}%){flag}")
    if total_out < 100:
        print(
            f"  WARNING: Only {total_out} candidates survived hard filters. "
            "Honeypot gate + Stage 6 may not produce 100 valid candidates."
        )

    return filtered


# ---------------------------------------------------------------------------
# Pass B — Honeypot Gate (zero score, keep in dataset)
# ---------------------------------------------------------------------------

def apply_honeypot_gate(candidates: List[Dict]) -> List[Dict]:
    """
    Flag impossible / fraudulent profiles as honeypots.

    Sets honeypot_flag = True and final_score = 0.0 on any candidate that
    triggers at least one of the five honeypot rules (architecture.md
    Stage 1 Pass B). Flagged candidates stay in the list so rank.py can
    explicitly exclude them from the top-100.

    Per instructions.md Section 4.1: honeypotted candidates MUST have
    final_score = 0.0 — no partial scoring.
    Per instructions.md Section 4.4: all five checks must be implemented.
    """
    for candidate in candidates:
        candidate["honeypot_flag"] = False

        profile = candidate.get("profile", {})
        career_history = candidate.get("career_history", [])
        education = candidate.get("education", [])
        skills = candidate.get("skills", [])
        redrob_signals = candidate.get("redrob_signals", {})

        claimed_yoe = get_field(profile, "years_of_experience")

        # ------------------------------------------------------------------
        # Check 1: Claimed YOE > actual career span + 2-year tolerance
        # ------------------------------------------------------------------
        if career_history:
            start_dates = [
                parse_date(e.get("start_date", ""))
                for e in career_history
                if e.get("start_date")
            ]
            if start_dates:
                earliest_start = min(start_dates)
                actual_span_years = (COMPETITION_DATE - earliest_start).days / 365.25
                if claimed_yoe > actual_span_years + 2:
                    candidate["honeypot_flag"] = True
                    candidate["final_score"] = 0.0
                    continue

        # ------------------------------------------------------------------
        # Check 2: Working before graduation year (- 1 yr internship tolerance)
        # ------------------------------------------------------------------
        if education and career_history:
            grad_years = [
                edu.get("end_year")
                for edu in education
                if edu.get("end_year")
            ]
            job_years = [
                parse_date(e.get("start_date", "")).year
                for e in career_history
                if e.get("start_date")
            ]
            if grad_years and job_years:
                grad_year = min(grad_years)
                earliest_job_year = min(job_years)
                if earliest_job_year < grad_year - 1:
                    candidate["honeypot_flag"] = True
                    candidate["final_score"] = 0.0
                    continue

        # ------------------------------------------------------------------
        # Checks 3 & 4: Skill-level plausibility
        # ------------------------------------------------------------------
        assessment_scores = redrob_signals.get("skill_assessment_scores", {})
        skill_flagged = False

        for skill in skills:
            proficiency = skill.get("proficiency", "").lower()
            duration_months = get_field(skill, "duration_months")
            skill_name = skill.get("name", "")

            # Check 3: Expert skill claimed with zero months of practice
            if proficiency == "expert" and duration_months == 0:
                skill_flagged = True
                break

            # Check 4: Expert/advanced claim contradicted by assessment score < 25
            assessment = assessment_scores.get(skill_name)
            if assessment is not None:
                if proficiency in ("expert", "advanced") and assessment < 25:
                    skill_flagged = True
                    break

        if skill_flagged:
            candidate["honeypot_flag"] = True
            candidate["final_score"] = 0.0
            continue

        # ------------------------------------------------------------------
        # Check 5: Claimed YOE exceeds biological maximum (> 40 years)
        # ------------------------------------------------------------------
        if claimed_yoe > 40:
            candidate["honeypot_flag"] = True
            candidate["final_score"] = 0.0
            continue

    return candidates


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import time

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jd_path = os.path.join(project_root, "artifacts", "jd_parsed.json")
    sample_path = os.path.join(project_root, "data", "sample_candidates.json")

    if not os.path.exists(jd_path):
        print(f"ERROR: {jd_path} not found. Run Stage 0 first.")
        exit(1)
    if not os.path.exists(sample_path):
        print(f"ERROR: {sample_path} not found.")
        exit(1)

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_parsed = json.load(f)

    with open(sample_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    print(f"[{time.strftime('%H:%M:%S')}] Stage 1: Hard Filters + Honeypot Gate — start")
    print(f"  Total candidates before Pass A: {len(candidates)}")

    filtered = apply_hard_filters(candidates, jd_parsed)
    print(f"  Candidates remaining after Pass A: {len(filtered)}")

    gated = apply_honeypot_gate(filtered)
    honeypot_count = sum(1 for c in gated if c.get("honeypot_flag", False))
    print(f"  Honeypots detected in Pass B   : {honeypot_count}")
    print(f"  Valid candidates remaining     : {len(gated) - honeypot_count}")
    print(f"[{time.strftime('%H:%M:%S')}] Stage 1: Hard Filters + Honeypot Gate — complete")
