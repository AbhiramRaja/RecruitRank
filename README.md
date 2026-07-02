# RecruitRank

**Intelligent Candidate Ranking System | India Runs Data & AI Challenge**

> *We rank on evidence, not keywords.*

## Overview

RecruitRank is an 8-stage candidate ranking pipeline that processes candidate profiles against a Job Description and outputs a top-100 ranked CSV with grounded reasoning. The pipeline is designed around three core principles:

1. **NDCG@10 maximisation** — getting the top-10 right matters more than perfect ordering of ranks 50–100.
2. **Honeypot immunity** — fake/impossible profiles are zeroed before any scoring computation.
3. **Explainability without hallucination** — every reasoning string is fact-checked against the source profile.

## Pipeline Architecture

The system uses an 8-stage offline/online architecture to ensure the final ranking process is incredibly fast:

- **Stage 0: JD Parser** - Extracts requirements from the job description.
- **Stage 1: Hard Filters & Honeypots** - Zeroes out fake or unqualified profiles immediately.
- **Stage 2: Feature Engineering** - Normalizes and extracts features.
- **Stage 3: Embeddings** - SBERT embeddings for semantic matching.
- **Stage 4: Scoring** - Initial lightweight scoring of candidates.
- **Stage 5: Behavioral Analysis** - Incorporates redrob signals and behavioral metrics.
- **Stage 6: Retrieval** - Selects the top pool of candidates.
- **Stage 7: Cross-Encoder** - Reranks the top pool for higher precision.
- **Stage 8: LLM Reasoning** - Generates grounded, fact-checked explainability for the final top 100.

## Repository Structure

```
├── README.md                 # This file
├── architecture.md           # Detailed architecture specification
├── instructions.md           # Implementation and rules specification
├── requirements.txt          # Python dependencies
├── precompute.py             # Offline precomputation entry point
├── rank.py                   # Online ranking entry point
├── check_artifacts.py        # Validates precomputed artifacts
├── download_models.sh        # Script to download necessary local models
├── data/                     # Dataset schemas, candidates, and job descriptions
├── pipeline/                 # The 8-stage pipeline modules
├── artifacts/                # Generated embeddings and precomputed outputs
├── models/                   # Cached local models (SBERT, LLM)
└── submissions/              # Output CSVs and metadata for submission
```

## Running the Pipeline

The system is split into two phases: **Precomputation** (offline, slow) and **Ranking** (online, fast).

### 1. Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the required local models (you do not need API keys, everything runs locally):
```bash
bash download_models.sh
```

Ensure your `data/` directory is populated with `candidates.jsonl` and the job description files.

### 2. Precomputation (Offline)

Run the offline precomputation to generate embeddings, features, and reasoning. This process takes time but generates the necessary artifacts so that `rank.py` can run instantly.
```bash
python precompute.py
```
This will populate the `artifacts/` directory.

### 3. Ranking (Online)

Execute the ranking pipeline. This step loads the precomputed artifacts and produces the final `submission.csv` in under 5 minutes without using any GPUs or external APIs.
```bash
python rank.py
```

## Output Format

The output is a CSV file (`submission.csv`) containing the top 100 candidates ranked, with detailed, non-hallucinated reasoning for each candidate's rank.
