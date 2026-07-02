"""
check_artifacts.py — Artifact status checker (Issue #3 fix)

Quickly shows which precomputed artifacts are present and which are missing,
so you know exactly what needs to be (re)computed before running rank.py.

Usage:
    python check_artifacts.py
    python check_artifacts.py --artifacts-dir path/to/artifacts
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Artifact registry (matches rank.py _load_artifacts + precompute.py stages)
# ---------------------------------------------------------------------------

ARTIFACTS = [
    # (filename, generated_by_stage, description)
    ("jd_parsed.json",        "Stage 0", "Parsed JD skills, requirements, disqualifiers"),
    ("jd_vector.npy",         "Stage 0", "JD sentence embedding vector"),
    ("features.parquet",      "Stage 2", "Extracted candidate feature matrix"),
    ("career_embeddings.npy", "Stage 3", "SBERT career-track embeddings (per candidate)"),
    ("skills_embeddings.npy", "Stage 3", "SBERT skills-track embeddings (per candidate)"),
    ("candidate_ids.json",    "Stage 3", "Ordered candidate ID list (parallel to .npy arrays)"),
    ("career_texts.json",     "Stage 3", "Raw career text strings used for embedding"),
    ("reasoning.json",        "Stage 8", "LLM/template reasoning strings for top 100"),
]

# Colour codes (disabled on Windows unless FORCE_COLOR is set)
_USE_COLOR = sys.stdout.isatty() or os.environ.get("FORCE_COLOR")
GREEN  = "\033[92m" if _USE_COLOR else ""
RED    = "\033[91m" if _USE_COLOR else ""
YELLOW = "\033[93m" if _USE_COLOR else ""
RESET  = "\033[0m"  if _USE_COLOR else ""


def check_artifacts(artifacts_dir: str) -> None:
    print(f"\nArtifact directory: {os.path.abspath(artifacts_dir)}\n")
    print(f"{'File':<28} {'Stage':<12} {'Status':<12} Description")
    print("-" * 90)

    present = 0
    missing = 0

    for fname, stage, desc in ARTIFACTS:
        path = os.path.join(artifacts_dir, fname)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            size_str = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
            status = f"{GREEN}[OK]{RESET} ({size_str})"
            present += 1
        else:
            status = f"{RED}[MISSING]{RESET}"
            missing += 1

        print(f"  {fname:<26} {stage:<12} {status:<24} {desc}")

    print("-" * 90)
    print(f"\n  {GREEN}{present}{RESET} present, {RED}{missing}{RESET} missing out of {len(ARTIFACTS)} total\n")

    if missing > 0:
        print(f"{YELLOW}Fix:{RESET} Run `python precompute.py` to generate all missing artifacts.")
        print("     This is a one-time offline step (may take 10-60 min for 100K candidates).\n")
        sys.exit(1)
    else:
        print(f"{GREEN}All artifacts present.{RESET} You can now run `python rank.py`.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check RecruitRank artifact completeness.")
    parser.add_argument(
        "--artifacts-dir", default="artifacts",
        help="Path to artifacts directory (default: artifacts/)"
    )
    args = parser.parse_args()
    check_artifacts(args.artifacts_dir)
