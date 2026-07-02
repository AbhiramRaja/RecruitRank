"""
Stage 2 — Feature Extraction

Runs: Offline pre-computation. Output saved to artifacts/features.parquet.
Input: Filtered candidates (post Stage 1 — hard filters applied, honeypots flagged)
Output: Pandas DataFrame with one row per candidate containing:
  - candidate_id
  - title_career_score    (2A)
  - skills_score          (2B)
  - experience_score      (2C)
  - location_edu_score    (2D)
  - honeypot_flag         (propagated from Stage 1 for rank.py join)
  - career_text           (raw text — needed by Stage 7 cross-encoder)
  - skill_text            (raw text — needed by Stage 3 embedding verification)

Per architecture.md Stage 2: "Runs offline, output saved to artifacts/features.parquet."
"""

import json
import os
import time
from typing import Dict, List

import pandas as pd

from pipeline.constants import (
    CONSULTING_INDUSTRIES,
    SENIOR_KEYWORDS,
    TARGET_CITIES,
    TARGET_TITLES,
)
from pipeline.utils import build_career_text, build_skills_text


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(candidates: List[Dict], jd_parsed: Dict) -> pd.DataFrame:
    """
    Compute all four feature scores for each candidate, plus store raw
    career/skill texts for downstream stages.

    Args:
        candidates: List of candidate dicts that have passed Stage 1
                    (honeypot_flag field is expected to be present).
        jd_parsed: Parsed JD dict from Stage 0.

    Returns:
        pd.DataFrame with columns:
          candidate_id, title_career_score, skills_score, experience_score,
          location_edu_score, honeypot_flag, career_text, skill_text
    """
    required_skills = jd_parsed.get("required_skills", [])
    nice_to_have = jd_parsed.get("nice_to_have_skills", [])
    required_skills_lower = {s.lower() for s in required_skills}

    rows: List[Dict] = []

    for candidate in candidates:
        cid = candidate["candidate_id"]
        profile = candidate.get("profile", {})
        career_history = candidate.get("career_history", [])
        skills = candidate.get("skills", [])
        education = candidate.get("education", [])
        redrob_signals = candidate.get("redrob_signals", {})
        honeypot_flag = candidate.get("honeypot_flag", False)

        # ==================================================================
        # 2A. Title + Career Score  (range 0.0 – 1.0)
        # architecture.md Stage 2, Section 2A
        # ==================================================================
        tcs = 0.0
        title_lower = profile.get("current_title", "").lower()

        # Base: current title in target scope?
        if any(t in title_lower for t in TARGET_TITLES):
            tcs += 0.4

        # Product company bonus: months at product companies
        product_months = sum(
            entry.get("duration_months", 0)
            for entry in career_history
            if entry.get("company_size") in (
                "201-500", "501-1000", "1001-5000", "5001-10000", "10001+"
            )
            and entry.get("industry") not in CONSULTING_INDUSTRIES
        )
        tcs += min(product_months / 60.0, 1.0) * 0.4   # caps at 5 yrs product exp

        # Seniority bonus
        if any(k in title_lower for k in SENIOR_KEYWORDS):
            tcs += 0.2

        title_career_score = min(tcs, 1.0)

        # ==================================================================
        # 2B. Skills Score  (range 0.0 – 1.0)
        # architecture.md Stage 2, Section 2B
        # ==================================================================
        candidate_skill_names = {
            s.get("name", "").lower() for s in skills if s.get("name")
        }

        # Required skill coverage (weighted 3× heavier per instructions.md §5.2)
        required_matched = sum(
            1 for rs in required_skills if rs.lower() in candidate_skill_names
        )
        required_score = required_matched / len(required_skills) if required_skills else 0.0

        # Nice-to-have coverage
        nth_matched = sum(
            1 for nth in nice_to_have if nth.lower() in candidate_skill_names
        )
        nth_score = nth_matched / len(nice_to_have) if nice_to_have else 0.0

        # Endorsement trust multiplier
        avg_endorsements = (
            sum(s.get("endorsements", 0) for s in skills) / len(skills)
            if skills else 0
        )
        endorsement_trust = min(avg_endorsements / 20.0, 1.0)   # caps at 20

        raw_skills_score = (3.0 * required_score + 1.0 * nth_score) / 4.0
        skills_score = raw_skills_score * (0.7 + 0.3 * endorsement_trust)

        # ==================================================================
        # 2C. Experience Score  (range 0.0 – 1.0)
        # architecture.md Stage 2, Section 2C
        # Piecewise — sweet spot is 4–10 yrs at product companies.
        # Per instructions.md §5.5: consulting time excluded.
        # ==================================================================
        product_yoe = sum(
            entry.get("duration_months", 0) / 12.0
            for entry in career_history
            if entry.get("industry") not in CONSULTING_INDUSTRIES
            and entry.get("company_size") not in ("1-10", "11-50")
        )

        if product_yoe < 2:
            es = 0.1
        elif product_yoe < 4:
            es = 0.1 + (product_yoe - 2) / 2.0 * 0.4      # 0.1 → 0.5 ramp
        elif product_yoe <= 10:
            es = 0.5 + (product_yoe - 4) / 6.0 * 0.5      # 0.5 → 1.0 ramp
        else:
            es = 1.0 - min((product_yoe - 10) / 10.0, 0.2) # soft cap, max -0.2

        experience_score = es

        # ==================================================================
        # 2D. Location + Education Score  (range 0.0 – 1.0)
        # architecture.md Stage 2, Section 2D
        # ==================================================================
        les = 0.0
        loc_lower = profile.get("location", "").lower()

        if any(city in loc_lower for city in TARGET_CITIES):
            les += 0.5
        elif redrob_signals.get("willing_to_relocate", False):
            les += 0.3

        # Education tier bonus — tier_1 < tier_2 < tier_3 lexicographically
        tiers = [edu.get("tier") for edu in education if edu.get("tier")]
        best_tier = min(tiers) if tiers else "unknown"

        if best_tier == "tier_1":
            les += 0.5
        elif best_tier == "tier_2":
            les += 0.3
        else:
            les += 0.1

        location_edu_score = min(les, 1.0)

        # ==================================================================
        # Raw texts — persisted for Stage 3 embedding and Stage 7 cross-encoder
        # ==================================================================
        c_text = build_career_text(candidate)
        s_text = build_skills_text(candidate, required_skills_lower)

        rows.append({
            "candidate_id": cid,
            "title_career_score": title_career_score,
            "skills_score": skills_score,
            "experience_score": experience_score,
            "location_edu_score": location_edu_score,
            "honeypot_flag": honeypot_flag,     # propagated for rank.py
            "career_text": c_text,              # needed by Stage 7 cross-encoder
            "skill_text": s_text,               # needed by Stage 3 consistency
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Artifact saver
# ---------------------------------------------------------------------------

def save_features(df: pd.DataFrame, artifacts_dir: str = "artifacts") -> str:
    """Save the features DataFrame to artifacts/features.parquet."""
    os.makedirs(artifacts_dir, exist_ok=True)
    output_path = os.path.join(artifacts_dir, "features.parquet")
    df.to_parquet(output_path, index=False)
    print(
        f"[Stage 2] Saved features.parquet -> {output_path} "
        f"({len(df)} rows, {list(df.columns)})"
    )
    return output_path


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pipeline.stage0_jd_parser import parse_jd
    from pipeline.stage1_filters import apply_hard_filters, apply_honeypot_gate

    print(f"[{time.strftime('%H:%M:%S')}] Stage 2: Feature Extraction — start")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_path = os.path.join(project_root, "data", "sample_candidates.json")
    artifacts_dir = os.path.join(project_root, "artifacts")

    if not os.path.exists(sample_path):
        print(f"ERROR: {sample_path} not found.")
        exit(1)

    jd_parsed = parse_jd()

    with open(sample_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    # Run Stage 1 to get filtered + honeypot-gated candidates
    filtered = apply_hard_filters(candidates, jd_parsed)
    gated = apply_honeypot_gate(filtered)

    df = extract_features(gated, jd_parsed)

    print("\nFeature Summary (numeric cols):")
    print(df[["title_career_score", "skills_score",
              "experience_score", "location_edu_score"]].describe())
    print(f"\nhoneypot_flag=True  : {df['honeypot_flag'].sum()}")
    print(f"career_text samples : {df['career_text'].str.len().describe()['mean']:.0f} chars avg")

    save_features(df, artifacts_dir=artifacts_dir)
    print(f"[{time.strftime('%H:%M:%S')}] Stage 2: Feature Extraction — complete")
