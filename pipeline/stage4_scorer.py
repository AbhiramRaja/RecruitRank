"""
Stage 4 — Weighted Score Combiner

Runs: Inline in rank.py — pure numpy/pandas, no ML.
Formula (architecture.md Stage 4, weights fixed per instructions.md §5.1):

base_score = (
    0.35 * title_career_score     # career depth signal
  + 0.30 * skills_score           # required vs nice-to-have, endorsement-weighted
  + 0.20 * experience_score       # piecewise, product co only
  + 0.10 * location_edu_score     # location match + institution tier
  + 0.05 * embedding_score        # dual-track SBERT similarity
)
"""

import time
from typing import List

import numpy as np
import pandas as pd


def compute_weighted_scores(
    features_df: pd.DataFrame,
    career_embeddings: np.ndarray,
    skills_embeddings: np.ndarray,
    jd_vector: np.ndarray,
    embedding_cids: List[str],
) -> pd.DataFrame:
    """
    Compute the combined embedding score and the final base_score.

    Args:
        features_df: DataFrame from Stage 2. Must contain columns:
                     candidate_id, title_career_score, skills_score,
                     experience_score, location_edu_score.
        career_embeddings: (N, 384) L2-normalised career vectors.
        skills_embeddings: (N, 384) L2-normalised skill vectors.
        jd_vector: (384,) L2-normalised JD vector.
        embedding_cids: List of N candidate_ids matching embedding rows
                        (from artifacts/candidate_ids.json).

    Returns:
        pd.DataFrame: features_df with added 'embedding_score' and
                      'base_score' columns.

    Raises:
        AssertionError: If the merge drops any rows (data mismatch between
                        features and embeddings), per instructions.md §12.3.
    """
    n_features = len(features_df)
    n_embeddings = len(embedding_cids)

    # --- Embedding similarity scores ---
    # Dot product of L2-normalised vectors == cosine similarity.
    career_sim = career_embeddings @ jd_vector   # (N,)
    skills_sim = skills_embeddings @ jd_vector   # (N,)

    # Per architecture.md Stage 3 + instructions.md §6.3: fixed formula.
    emb_scores = 0.65 * career_sim + 0.35 * skills_sim

    emb_df = pd.DataFrame(
        {"candidate_id": embedding_cids, "embedding_score": emb_scores}
    )

    # --- Merge features with embedding scores ---
    df = features_df.merge(emb_df, on="candidate_id", how="inner")

    # Fail loudly if the merge silently dropped rows (data mismatch).
    # Per instructions.md §12.3: "fail loudly, never silently".
    assert len(df) == n_features, (
        f"Stage 4 merge dropped rows: features had {n_features} rows, "
        f"embeddings had {n_embeddings} entries, merged to {len(df)}. "
        "Ensure precompute.py was run on the same candidate set as rank.py."
    )

    # --- Weighted base score (weights fixed per instructions.md §5.1) ---
    df["base_score"] = (
        0.35 * df["title_career_score"]
        + 0.30 * df["skills_score"]
        + 0.20 * df["experience_score"]
        + 0.10 * df["location_edu_score"]
        + 0.05 * df["embedding_score"]
    )

    return df


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import os

    print(f"[{time.strftime('%H:%M:%S')}] Stage 4: Weighted Score Combiner — start")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(project_root, "artifacts")

    features_path = os.path.join(artifacts_dir, "features.parquet")
    career_path = os.path.join(artifacts_dir, "career_embeddings.npy")
    skills_path = os.path.join(artifacts_dir, "skills_embeddings.npy")
    jd_path = os.path.join(artifacts_dir, "jd_vector.npy")
    ids_path = os.path.join(artifacts_dir, "candidate_ids.json")

    required = [features_path, career_path, skills_path, jd_path, ids_path]
    if not all(os.path.exists(p) for p in required):
        missing = [p for p in required if not os.path.exists(p)]
        print(f"Missing artifacts: {missing}\nRun Stages 0–3 first.")
        exit(1)

    features_df = pd.read_parquet(features_path)
    career_emb = np.load(career_path)
    skills_emb = np.load(skills_path)
    jd_vec = np.load(jd_path)

    with open(ids_path, "r") as f:
        cids = json.load(f)

    scored_df = compute_weighted_scores(
        features_df, career_emb, skills_emb, jd_vec, cids
    )

    print("\nScore Summary:")
    print(scored_df[["base_score", "embedding_score"]].describe())

    print("\nTop 5 Candidates by Base Score:")
    top5 = scored_df.sort_values("base_score", ascending=False).head(5)
    for _, row in top5.iterrows():
        print(f"  {row['candidate_id']}: base={row['base_score']:.4f}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Stage 4: Weighted Score Combiner — complete")
