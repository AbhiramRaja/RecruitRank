"""
precompute.py — Offline Pre-computation Runner

Run this ONCE before `rank.py`. It builds all expensive artifacts so that
rank.py can complete in under 5 minutes.

Stages run (per instructions.md §1.2):
  Stage 0  → artifacts/jd_parsed.json + artifacts/jd_vector.npy
  Stage 1  → (in-memory only: filter + honeypot gate on all candidates)
  Stage 2  → artifacts/features.parquet
  Stage 3  → artifacts/career_embeddings.npy + skills_embeddings.npy
             + candidate_ids.json + career_texts.json
  Stage 4/5/6/7 → (in-memory: needed to get the final top_100 for Stage 8)
  Stage 8  → artifacts/reasoning.json

Usage:
  python precompute.py [--data-dir data] [--artifacts-dir artifacts] [--force]

  --data-dir       Directory containing candidates.json / candidates.jsonl
  --artifacts-dir  Directory to write artifacts (default: artifacts/)
  --force          Recompute all stages even if artifacts already exist
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

# Reproducibility (instructions.md §12.4)
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _load_candidates(data_dir: str):
    """
    Load candidate records from data_dir. Supports:
      - candidates.jsonl  (JSONL — one JSON object per line)
      - candidates.json   (JSON array)
      - sample_candidates.json (fallback for local testing)
    """
    search_paths = [
        os.path.join(data_dir, "candidates.jsonl"),
        os.path.join(data_dir, "candidates.json"),
        "sample_candidates.json",
        os.path.join("data", "candidates.json"),
    ]

    for path in search_paths:
        if not os.path.exists(path):
            continue
        print(f"[{_ts()}] Loading candidates from '{path}'...")
        t0 = time.time()
        if path.endswith(".jsonl"):
            candidates = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        candidates.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                candidates = json.load(f)
        elapsed = time.time() - t0
        print(f"[{_ts()}] Loaded {len(candidates):,} candidates in {elapsed:.1f}s.")
        return candidates

    raise FileNotFoundError(
        f"No candidate data found. Searched: {search_paths}\n"
        "Place candidates.json or candidates.jsonl in the --data-dir directory."
    )


def _artifact_exists(artifacts_dir: str, *filenames: str) -> bool:
    """Return True if ALL given artifact filenames exist."""
    return all(
        os.path.exists(os.path.join(artifacts_dir, f)) for f in filenames
    )


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_stage0(artifacts_dir: str, force: bool):
    from pipeline.stage0_jd_parser import (
        compute_and_save_jd_vector,
        parse_jd,
        save_jd_parsed,
    )

    if not force and _artifact_exists(artifacts_dir, "jd_parsed.json", "jd_vector.npy"):
        print(f"[{_ts()}] Stage 0: Artifacts already exist — skipping. (--force to rerun)")
        with open(os.path.join(artifacts_dir, "jd_parsed.json"), "r") as f:
            return json.load(f)

    print(f"[{_ts()}] Stage 0: JD Parsing — start")
    t0 = time.time()
    jd_parsed = parse_jd()
    save_jd_parsed(jd_parsed, artifacts_dir=artifacts_dir)
    compute_and_save_jd_vector(jd_parsed, artifacts_dir=artifacts_dir)
    print(f"[{_ts()}] Stage 0: Complete in {time.time()-t0:.1f}s")
    return jd_parsed


def run_stage1(candidates, jd_parsed):
    from pipeline.stage1_filters import apply_hard_filters, apply_honeypot_gate

    print(f"[{_ts()}] Stage 1: Hard Filters + Honeypot Gate — start ({len(candidates):,} candidates)")
    t0 = time.time()
    filtered = apply_hard_filters(candidates, jd_parsed)
    gated = apply_honeypot_gate(filtered)
    hp_count = sum(1 for c in gated if c.get("honeypot_flag", False))
    valid_count = len(gated) - hp_count
    print(
        f"[{_ts()}] Stage 1: Complete in {time.time()-t0:.1f}s. "
        f"{len(candidates):,} → {len(gated):,} (Pass A). "
        f"Honeypots: {hp_count:,}. Valid: {valid_count:,}."
    )
    return gated


def run_stage2(gated_candidates, jd_parsed, artifacts_dir: str, force: bool):
    from pipeline.stage2_features import extract_features, save_features

    if not force and _artifact_exists(artifacts_dir, "features.parquet"):
        print(f"[{_ts()}] Stage 2: features.parquet already exists — skipping.")
        return pd.read_parquet(os.path.join(artifacts_dir, "features.parquet"))

    print(f"[{_ts()}] Stage 2: Feature Extraction — start ({len(gated_candidates):,} candidates)")
    t0 = time.time()
    features_df = extract_features(gated_candidates, jd_parsed)
    save_features(features_df, artifacts_dir=artifacts_dir)
    print(f"[{_ts()}] Stage 2: Complete in {time.time()-t0:.1f}s. {len(features_df):,} rows.")
    return features_df


def run_stage3(valid_candidates, jd_parsed, artifacts_dir: str, force: bool):
    from pipeline.stage3_embeddings import compute_embeddings, save_embeddings

    required = ["career_embeddings.npy", "skills_embeddings.npy",
                "jd_vector.npy", "candidate_ids.json", "career_texts.json"]
    if not force and _artifact_exists(artifacts_dir, *required):
        print(f"[{_ts()}] Stage 3: Embeddings already exist — skipping.")
        from pipeline.stage3_embeddings import load_embeddings, load_career_texts
        career_emb, skills_emb, jd_vec, cids = load_embeddings(artifacts_dir)
        career_texts = load_career_texts(artifacts_dir)
        return career_emb, skills_emb, jd_vec, cids, career_texts

    print(f"[{_ts()}] Stage 3: SBERT Embedding — start ({len(valid_candidates):,} candidates)")
    t0 = time.time()
    career_emb, skills_emb, jd_vec, cids, career_texts = compute_embeddings(
        valid_candidates, jd_parsed, artifacts_dir=artifacts_dir
    )
    save_embeddings(career_emb, skills_emb, jd_vec, cids, career_texts,
                    artifacts_dir=artifacts_dir)
    print(f"[{_ts()}] Stage 3: Complete in {time.time()-t0:.1f}s. {len(cids):,} candidates embedded.")
    return career_emb, skills_emb, jd_vec, cids, career_texts


def run_stages_456(features_df, career_emb, skills_emb, jd_vec, cids, candidates_dict):
    from pipeline.stage4_scorer import compute_weighted_scores
    from pipeline.stage5_behavioral import apply_behavioral_modifier

    # Stage 4: base scores (only valid/non-honeypot features)
    print(f"[{_ts()}] Stage 4: Weighted Score Combiner — start")
    t0 = time.time()
    valid_features = features_df[features_df["honeypot_flag"] == False].copy()
    scored_df = compute_weighted_scores(valid_features, career_emb, skills_emb, jd_vec, cids)
    print(f"[{_ts()}] Stage 4: Complete in {time.time()-t0:.1f}s. {len(scored_df):,} rows scored.")

    # Stage 5: behavioral modifier
    print(f"[{_ts()}] Stage 5: Behavioral Signal Modifier — start")
    t0 = time.time()
    valid_cids = set(scored_df["candidate_id"])
    valid_candidates = [c for c in candidates_dict.values() if c["candidate_id"] in valid_cids]
    final_df = apply_behavioral_modifier(scored_df, valid_candidates)
    print(f"[{_ts()}] Stage 5: Complete in {time.time()-t0:.1f}s.")

    return final_df


def run_stage6(final_df):
    from pipeline.stage6_retrieval import retrieve_top_500
    return retrieve_top_500(final_df)


def run_stage7(top_500, artifacts_dir: str):
    from pipeline.stage7_crossencoder import run_stage7 as _run7
    return _run7(top_500, artifacts_dir=artifacts_dir)


def run_stage8(top_100, candidates_dict, jd_parsed, artifacts_dir: str,
               project_root: str, force: bool):
    from pipeline.stage8_reasoning import run_stage8 as _run8

    if not force and _artifact_exists(artifacts_dir, "reasoning.json"):
        print(f"[{_ts()}] Stage 8: reasoning.json already exists — skipping.")
        from pipeline.stage8_reasoning import load_reasoning
        return load_reasoning(artifacts_dir)

    print(f"[{_ts()}] Stage 8: Reasoning Generation — start (100 candidates)")
    t0 = time.time()
    reasoning = _run8(
        top_100_df=top_100,
        candidates_dict=candidates_dict,
        jd_parsed=jd_parsed,
        artifacts_dir=artifacts_dir,
        project_root=project_root,
    )
    print(f"[{_ts()}] Stage 8: Complete in {time.time()-t0:.1f}s.")
    return reasoning


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute all expensive artifacts for RecruitRank."
    )
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing candidates.json/jsonl (default: data/)")
    parser.add_argument("--artifacts-dir", default="artifacts",
                        help="Directory to write artifact files (default: artifacts/)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute all stages even if artifacts already exist")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(project_root, args.artifacts_dir)
    data_dir = os.path.join(project_root, args.data_dir)
    os.makedirs(artifacts_dir, exist_ok=True)

    print("=" * 60)
    print("RecruitRank — Offline Pre-computation")
    print(f"  data_dir     : {data_dir}")
    print(f"  artifacts_dir: {artifacts_dir}")
    print(f"  force        : {args.force}")
    print("=" * 60)

    t_total = time.time()

    # --- Stage 0: JD Parsing + JD Vector ---
    jd_parsed = run_stage0(artifacts_dir, args.force)

    # --- Load raw candidates ---
    # Try data_dir first, then project root for sample data
    try:
        candidates = _load_candidates(data_dir)
    except FileNotFoundError:
        candidates = _load_candidates(project_root)

    candidates_dict = {c["candidate_id"]: c for c in candidates}
    print(f"[{_ts()}] Candidate dict built: {len(candidates_dict):,} entries.")

    # --- Stage 1: Hard Filters + Honeypot Gate ---
    gated = run_stage1(candidates, jd_parsed)
    valid_candidates = [c for c in gated if not c.get("honeypot_flag", False)]

    # --- Stage 2: Feature Extraction ---
    features_df = run_stage2(gated, jd_parsed, artifacts_dir, args.force)

    # --- Stage 3: SBERT Embeddings ---
    career_emb, skills_emb, jd_vec, cids, career_texts = run_stage3(
        valid_candidates, jd_parsed, artifacts_dir, args.force
    )

    # --- Stages 4+5: Scoring + Behavioral Modifier ---
    final_df = run_stages_456(features_df, career_emb, skills_emb, jd_vec,
                              cids, candidates_dict)

    # --- Stage 6: Top 500 Retrieval ---
    top_500 = run_stage6(final_df)

    # --- Stage 7: Cross-Encoder Reranking ---
    top_100 = run_stage7(top_500, artifacts_dir)

    # --- Stage 8: Reasoning Generation ---
    run_stage8(top_100, candidates_dict, jd_parsed, artifacts_dir,
               project_root, args.force)

    # --- Summary ---
    elapsed_total = time.time() - t_total
    print()
    print("=" * 60)
    print(f"Pre-computation complete in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min).")
    print(f"Artifacts written to: {artifacts_dir}")
    artifacts_written = [
        "jd_parsed.json", "jd_vector.npy", "features.parquet",
        "career_embeddings.npy", "skills_embeddings.npy",
        "candidate_ids.json", "career_texts.json", "reasoning.json",
    ]
    for fname in artifacts_written:
        full_path = os.path.join(artifacts_dir, fname)
        exists = "✓" if os.path.exists(full_path) else "✗ MISSING"
        size = (
            f"{os.path.getsize(full_path)/1024/1024:.1f} MB"
            if os.path.exists(full_path) else ""
        )
        print(f"  [{exists}] {fname:35s} {size}")
    print()
    print("Next step: python rank.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
