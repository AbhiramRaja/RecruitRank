"""
Stage 3 — Dual-Track SBERT Embedding

Runs: Offline pre-computation. Output saved to artifacts/.
Model: sentence-transformers/all-MiniLM-L6-v2 (CPU, ~40MB, 384-dim)

Two embedding tracks:
  Track A: Career Description Embedding — concatenated career history
           "{title} at {company}: {description}" per entry.
  Track B: Skills Embedding — weighted skill text where required skills
           from the JD are repeated 3× to boost their semantic weight.

Also pre-computes the JD vector from jd_parsed["jd_embedding_text"].

All embeddings are L2-normalized (normalize_embeddings=True) so that
cosine similarity = dot product at ranking time.

Output files:
  artifacts/career_embeddings.npy   shape: (N, 384)  float32
  artifacts/skills_embeddings.npy   shape: (N, 384)  float32
  artifacts/jd_vector.npy           shape: (384,)    float32
  artifacts/candidate_ids.json      ordered list of candidate_ids matching
                                    rows in the embedding arrays
  artifacts/career_texts.json       {candidate_id: career_text} — required
                                    by Stage 7 cross-encoder at ranking time
"""

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from pipeline.constants import EMBEDDING_BATCH_SIZE, EMBEDDING_DIM, SBERT_MODEL_NAME
from pipeline.utils import build_career_text, build_skills_text


# ---------------------------------------------------------------------------
# Core embedding logic
# ---------------------------------------------------------------------------

def compute_embeddings(
    candidates: List[Dict],
    jd_parsed: Dict,
    artifacts_dir: str = "artifacts",
    model_cache_dir: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, str]]:
    """
    Compute dual-track SBERT embeddings for all candidates + JD vector.

    Args:
        candidates: List of candidate dicts (post Stage-1 filtering).
        jd_parsed: Parsed JD dict from Stage 0 (must contain
                   "required_skills" and "jd_embedding_text").
        artifacts_dir: Directory to save .npy outputs.
        model_cache_dir: Optional directory for caching the SBERT model.

    Returns:
        Tuple of (career_embeddings, skills_embeddings, jd_vector,
                  candidate_ids, career_texts_dict) where:
          career_embeddings:  np.ndarray shape (N, 384)
          skills_embeddings:  np.ndarray shape (N, 384)
          jd_vector:          np.ndarray shape (384,)
          candidate_ids:      list of N candidate_id strings (row order)
          career_texts_dict:  {candidate_id: career_text} for Stage 7
    """
    # Deferred import — avoids import-time overhead when module is inspected.
    from sentence_transformers import SentenceTransformer

    n = len(candidates)
    print(
        f"[{time.strftime('%H:%M:%S')}] Stage 3: Loading SBERT model "
        f"'{SBERT_MODEL_NAME}'..."
    )

    load_kwargs: Dict = {}
    if model_cache_dir:
        load_kwargs["cache_folder"] = model_cache_dir

    model = SentenceTransformer(SBERT_MODEL_NAME, **load_kwargs)
    print(
        f"[{time.strftime('%H:%M:%S')}] Stage 3: Model loaded. "
        f"Embedding {n} candidates (batch_size={EMBEDDING_BATCH_SIZE})..."
    )

    # Build required skills set for Track B weighting
    required_skills_lower = {
        s.lower() for s in jd_parsed.get("required_skills", [])
    }

    # Build text inputs and store career texts for Stage 7
    candidate_ids: List[str] = []
    career_texts: List[str] = []
    skills_texts: List[str] = []
    career_texts_dict: Dict[str, str] = {}

    for candidate in candidates:
        cid = candidate["candidate_id"]
        c_text = build_career_text(candidate)
        s_text = build_skills_text(candidate, required_skills_lower)

        candidate_ids.append(cid)
        career_texts.append(c_text)
        skills_texts.append(s_text)
        career_texts_dict[cid] = c_text   # keyed dict for Stage 7 lookup

    # --- Track A: Career Description Embeddings (batched) ---
    # Per instructions.md Section 6.5: normalize all embeddings.
    # Per instructions.md Section 11.3: use batch processing.
    print(
        f"[{time.strftime('%H:%M:%S')}] Stage 3: Track A — "
        f"Career description embeddings ({n} texts)..."
    )
    career_embeddings = model.encode(
        career_texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    career_embeddings = np.asarray(career_embeddings, dtype=np.float32)
    assert career_embeddings.shape == (n, EMBEDDING_DIM), (
        f"career_embeddings shape mismatch: expected ({n}, {EMBEDDING_DIM}), "
        f"got {career_embeddings.shape}"
    )

    # --- Track B: Skills Embeddings (batched) ---
    print(
        f"[{time.strftime('%H:%M:%S')}] Stage 3: Track B — "
        f"Skills embeddings ({n} texts)..."
    )
    skills_embeddings = model.encode(
        skills_texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    skills_embeddings = np.asarray(skills_embeddings, dtype=np.float32)
    assert skills_embeddings.shape == (n, EMBEDDING_DIM), (
        f"skills_embeddings shape mismatch: expected ({n}, {EMBEDDING_DIM}), "
        f"got {skills_embeddings.shape}"
    )

    # --- JD Vector ---
    # Per architecture.md Stage 0: also computed here so Stage 3 is
    # self-contained. jd_vector.npy may already exist from Stage 0;
    # this overwrites it with the identical value (idempotent).
    print(f"[{time.strftime('%H:%M:%S')}] Stage 3: Computing JD vector...")
    jd_embedding_text = jd_parsed.get("jd_embedding_text", "")
    if not jd_embedding_text:
        raise ValueError(
            "jd_parsed['jd_embedding_text'] is empty. "
            "Run Stage 0 (JD Parsing) first."
        )
    jd_vector = model.encode(jd_embedding_text, normalize_embeddings=True)
    jd_vector = np.asarray(jd_vector, dtype=np.float32)
    assert jd_vector.shape == (EMBEDDING_DIM,), (
        f"jd_vector shape mismatch: expected ({EMBEDDING_DIM},), "
        f"got {jd_vector.shape}"
    )

    print(f"[{time.strftime('%H:%M:%S')}] Stage 3: All embeddings computed.")
    return career_embeddings, skills_embeddings, jd_vector, candidate_ids, career_texts_dict


# ---------------------------------------------------------------------------
# Save / Load helpers
# ---------------------------------------------------------------------------

def save_embeddings(
    career_embeddings: np.ndarray,
    skills_embeddings: np.ndarray,
    jd_vector: np.ndarray,
    candidate_ids: List[str],
    career_texts_dict: Dict[str, str],
    artifacts_dir: str = "artifacts",
) -> None:
    """
    Save all embedding artifacts to disk.

    Files created:
      artifacts/career_embeddings.npy   (N, 384)
      artifacts/skills_embeddings.npy   (N, 384)
      artifacts/jd_vector.npy           (384,)
      artifacts/candidate_ids.json      ordered list of candidate_id strings
      artifacts/career_texts.json       {candidate_id: career_text} for Stage 7
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    career_path = os.path.join(artifacts_dir, "career_embeddings.npy")
    skills_path = os.path.join(artifacts_dir, "skills_embeddings.npy")
    jd_path = os.path.join(artifacts_dir, "jd_vector.npy")
    ids_path = os.path.join(artifacts_dir, "candidate_ids.json")
    texts_path = os.path.join(artifacts_dir, "career_texts.json")

    np.save(career_path, career_embeddings)
    print(
        f"[Stage 3] Saved career_embeddings.npy -> {career_path} "
        f"(shape: {career_embeddings.shape})"
    )

    np.save(skills_path, skills_embeddings)
    print(
        f"[Stage 3] Saved skills_embeddings.npy -> {skills_path} "
        f"(shape: {skills_embeddings.shape})"
    )

    np.save(jd_path, jd_vector)
    print(
        f"[Stage 3] Saved jd_vector.npy -> {jd_path} "
        f"(shape: {jd_vector.shape})"
    )

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f)
    print(
        f"[Stage 3] Saved candidate_ids.json -> {ids_path} "
        f"({len(candidate_ids)} entries)"
    )

    # career_texts.json — keyed by candidate_id for O(1) lookup in Stage 7
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(career_texts_dict, f, ensure_ascii=False)
    print(
        f"[Stage 3] Saved career_texts.json -> {texts_path} "
        f"({len(career_texts_dict)} entries)"
    )


def load_embeddings(
    artifacts_dir: str = "artifacts",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load the core pre-computed embedding artifacts from disk.

    Returns:
        Tuple of (career_embeddings, skills_embeddings, jd_vector, candidate_ids)

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """
    career_path = os.path.join(artifacts_dir, "career_embeddings.npy")
    skills_path = os.path.join(artifacts_dir, "skills_embeddings.npy")
    jd_path = os.path.join(artifacts_dir, "jd_vector.npy")
    ids_path = os.path.join(artifacts_dir, "candidate_ids.json")

    for path in [career_path, skills_path, jd_path, ids_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing embedding artifact: {path}. "
                "Run precompute.py (Stage 3) first."
            )

    career_embeddings = np.load(career_path)
    skills_embeddings = np.load(skills_path)
    jd_vector = np.load(jd_path)

    with open(ids_path, "r", encoding="utf-8") as f:
        candidate_ids = json.load(f)

    assert career_embeddings.shape[1] == EMBEDDING_DIM, (
        f"career_embeddings dim mismatch: expected {EMBEDDING_DIM}, "
        f"got {career_embeddings.shape[1]}"
    )
    assert skills_embeddings.shape[1] == EMBEDDING_DIM, (
        f"skills_embeddings dim mismatch: expected {EMBEDDING_DIM}, "
        f"got {skills_embeddings.shape[1]}"
    )
    assert jd_vector.shape == (EMBEDDING_DIM,), (
        f"jd_vector shape mismatch: expected ({EMBEDDING_DIM},), "
        f"got {jd_vector.shape}"
    )
    assert len(candidate_ids) == career_embeddings.shape[0], (
        f"candidate_ids length {len(candidate_ids)} != "
        f"career_embeddings rows {career_embeddings.shape[0]}"
    )

    return career_embeddings, skills_embeddings, jd_vector, candidate_ids


def load_career_texts(artifacts_dir: str = "artifacts") -> Dict[str, str]:
    """
    Load the pre-computed career_texts.json artifact.

    Returns:
        Dict mapping candidate_id -> career_text string.

    Raises:
        FileNotFoundError: If career_texts.json is missing (run precompute.py).
    """
    texts_path = os.path.join(artifacts_dir, "career_texts.json")
    if not os.path.exists(texts_path):
        raise FileNotFoundError(
            f"Missing artifact: {texts_path}. "
            "Run precompute.py (Stage 3) first."
        )
    with open(texts_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_embedding_scores(
    career_embeddings: np.ndarray,
    skills_embeddings: np.ndarray,
    jd_vector: np.ndarray,
) -> np.ndarray:
    """
    Compute combined embedding similarity scores for all candidates.

    Per architecture.md Stage 3:
        career_sim = cosine_similarity(career_vector, jd_vector)
        skills_sim = cosine_similarity(skills_vector, jd_vector)
        embedding_score = 0.65 * career_sim + 0.35 * skills_sim

    Since all vectors are L2-normalized, cosine similarity = dot product.
    Per instructions.md Section 6.3: this formula is fixed.

    Args:
        career_embeddings: (N, 384) normalized career vectors
        skills_embeddings: (N, 384) normalized skill vectors
        jd_vector:         (384,)   normalized JD vector

    Returns:
        np.ndarray of shape (N,) with combined embedding scores.
    """
    career_sim = career_embeddings @ jd_vector   # (N,)
    skills_sim = skills_embeddings @ jd_vector   # (N,)
    return 0.65 * career_sim + 0.35 * skills_sim


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pipeline.stage0_jd_parser import parse_jd
    from pipeline.stage1_filters import apply_hard_filters, apply_honeypot_gate

    print(
        f"[{time.strftime('%H:%M:%S')}] "
        "Stage 3: Dual-Track SBERT Embedding — start"
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_path = os.path.join(project_root, "sample_candidates.json")
    artifacts_dir = os.path.join(project_root, "artifacts")

    jd_parsed = parse_jd()

    if not os.path.exists(sample_path):
        print(f"ERROR: {sample_path} not found.")
        exit(1)

    with open(sample_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"  Loaded {len(candidates)} candidates from sample_candidates.json")

    # Stage 1: filter + honeypot gate
    filtered = apply_hard_filters(candidates, jd_parsed)
    filtered = apply_honeypot_gate(filtered)
    # Per architecture.md Key Design Decisions: no SBERT compute on honeypots
    valid = [c for c in filtered if not c.get("honeypot_flag", False)]
    print(
        f"  After Stage 1: {len(valid)} valid candidates "
        f"(filtered from {len(candidates)})"
    )

    # Stage 3: embeddings
    career_emb, skills_emb, jd_vec, cids, texts_dict = compute_embeddings(
        valid, jd_parsed, artifacts_dir=artifacts_dir
    )
    save_embeddings(
        career_emb, skills_emb, jd_vec, cids, texts_dict,
        artifacts_dir=artifacts_dir,
    )

    # Sanity check
    scores = compute_embedding_scores(career_emb, skills_emb, jd_vec)
    print(f"\n  Embedding score stats:")
    print(f"    min:  {scores.min():.4f}")
    print(f"    max:  {scores.max():.4f}")
    print(f"    mean: {scores.mean():.4f}")
    print(f"    std:  {scores.std():.4f}")

    top5_idx = np.argsort(scores)[::-1][:5]
    print(f"\n  Top 5 candidates by embedding score:")
    for i, idx in enumerate(top5_idx):
        print(f"    {i+1}. {cids[idx]}  score={scores[idx]:.4f}")

    print(
        f"\n[{time.strftime('%H:%M:%S')}] "
        "Stage 3: Dual-Track SBERT Embedding — complete"
    )
