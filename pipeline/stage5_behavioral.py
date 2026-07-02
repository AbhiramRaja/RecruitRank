"""
Stage 5 — Behavioral Signal Modifier

Runs: Inline in rank.py.
Design: Multiplicative modifier, clipped to [0.4, 1.2].

    final_base_score = base_score × clamp(behavioral_modifier, 0.4, 1.2)

Per instructions.md §5.3: modifier is multiplicative, NOT additive.
Per instructions.md §5.4: modifier MUST be clipped to [0.4, 1.2].
"""

import time
from typing import Dict, List

import pandas as pd

from pipeline.constants import COMPETITION_DATE
from pipeline.utils import parse_date


# ---------------------------------------------------------------------------
# Behavioral modifier computation
# ---------------------------------------------------------------------------

def get_behavioral_modifier(signals: Dict) -> float:
    """
    Compute the behavioral modifier for a single candidate from their
    redrob_signals dict.

    Implements all signal checks from architecture.md Stage 5 in exact
    order with exact thresholds and deltas.

    Returns:
        float in [0.4, 1.2] — hard-clipped per instructions.md §5.4.
    """
    score = 1.0   # neutral baseline

    # ------------------------------------------------------------------
    # Availability — can they start soon?
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Engagement quality
    # ------------------------------------------------------------------
    recruiter_response_rate = signals.get("recruiter_response_rate", 0.5)
    if recruiter_response_rate >= 0.7:
        score += 0.06
    elif recruiter_response_rate <= 0.2:
        score -= 0.08

    if signals.get("avg_response_time_hours", 24) <= 4:
        score += 0.03

    # ------------------------------------------------------------------
    # Recruiter interest — social proof
    # ------------------------------------------------------------------
    if signals.get("saved_by_recruiters_30d", 0) >= 5:
        score += 0.05
    if signals.get("search_appearance_30d", 0) >= 10:
        score += 0.03

    # ------------------------------------------------------------------
    # Trust + reliability
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # GitHub activity — engineering signal
    # ------------------------------------------------------------------
    github_activity_score = signals.get("github_activity_score", -1)
    if github_activity_score >= 70:
        score += 0.04
    elif github_activity_score == -1:
        pass   # no penalty for no GitHub (architecture.md Stage 5)

    # Hard clip [0.4, 1.2] per instructions.md §5.4
    return max(0.4, min(1.2, score))


def apply_behavioral_modifier(
    scored_df: pd.DataFrame, candidates: List[Dict]
) -> pd.DataFrame:
    """
    Apply the behavioral modifier to each candidate's base_score.

    Args:
        scored_df: DataFrame with 'candidate_id' and 'base_score' columns
                   (output of Stage 4).
        candidates: List of raw candidate dicts containing 'redrob_signals'.

    Returns:
        DataFrame with added 'behavioral_modifier' and 'final_base_score'
        columns.
        final_base_score = base_score × behavioral_modifier
        (per instructions.md §5.3: multiplicative, not additive)
    """
    modifiers = [
        {
            "candidate_id": cand.get("candidate_id"),
            "behavioral_modifier": get_behavioral_modifier(
                cand.get("redrob_signals", {})
            ),
        }
        for cand in candidates
    ]
    mod_df = pd.DataFrame(modifiers)

    df = scored_df.merge(mod_df, on="candidate_id", how="left")

    # fillna(1.0): neutral modifier for any candidate not in the candidate
    # list (should not happen in production, but guards against edge cases).
    df["behavioral_modifier"] = df["behavioral_modifier"].fillna(1.0)

    # Multiplicative application per instructions.md §5.3
    df["final_base_score"] = df["base_score"] * df["behavioral_modifier"]

    return df


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import os

    import numpy as np

    from pipeline.stage4_scorer import compute_weighted_scores

    print(f"[{time.strftime('%H:%M:%S')}] Stage 5: Behavioral Signal Modifier — start")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(project_root, "artifacts")
    sample_path = os.path.join(project_root, "data", "sample_candidates.json")

    features_path = os.path.join(artifacts_dir, "features.parquet")
    career_path = os.path.join(artifacts_dir, "career_embeddings.npy")
    skills_path = os.path.join(artifacts_dir, "skills_embeddings.npy")
    jd_path = os.path.join(artifacts_dir, "jd_vector.npy")
    ids_path = os.path.join(artifacts_dir, "candidate_ids.json")

    required = [features_path, career_path, skills_path, jd_path,
                ids_path, sample_path]
    if not all(os.path.exists(p) for p in required):
        missing = [p for p in required if not os.path.exists(p)]
        print(f"Missing files: {missing}\nRun Stages 0–4 first.")
        exit(1)

    import pandas as pd

    with open(sample_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    features_df = pd.read_parquet(features_path)
    career_emb = np.load(career_path)
    skills_emb = np.load(skills_path)
    jd_vec = np.load(jd_path)

    with open(ids_path, "r") as f:
        cids = json.load(f)

    scored_df = compute_weighted_scores(
        features_df, career_emb, skills_emb, jd_vec, cids
    )

    # Only pass candidates whose IDs are in the scored DataFrame
    scored_ids = set(scored_df["candidate_id"])
    scored_candidates = [c for c in candidates if c["candidate_id"] in scored_ids]

    final_df = apply_behavioral_modifier(scored_df, scored_candidates)

    print("\nModifier Summary:")
    print(
        final_df[["base_score", "behavioral_modifier", "final_base_score"]].describe()
    )

    print("\nTop 5 Candidates by Final Base Score:")
    top5 = final_df.sort_values("final_base_score", ascending=False).head(5)
    for _, row in top5.iterrows():
        print(
            f"  {row['candidate_id']}: "
            f"base={row['base_score']:.4f} × "
            f"mod={row['behavioral_modifier']:.4f} = "
            f"final={row['final_base_score']:.4f}"
        )

    print(
        f"[{time.strftime('%H:%M:%S')}] Stage 5: Behavioral Signal Modifier — complete"
    )
