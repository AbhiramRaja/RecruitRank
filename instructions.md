# RecruitRank — Builder Instructions
**Strict boundaries for implementation. Read every rule before writing a single line of code.**

---

## 0. The Prime Directive

This document governs how `architecture.md` is implemented. Where these instructions conflict with intuition or convenience, these instructions win. No exceptions.

The architecture exists as designed. Your job is to implement it faithfully, not to redesign it.

---

## 1. Entry Point Rules

### 1.1 `rank.py` is the only ranked execution entry point

```
rank.py must:
  - Accept no command-line arguments (or exactly: --data-dir, --artifacts-dir, --output)
  - Complete in ≤ 5 minutes wall clock on a 16GB RAM CPU machine with no GPU
  - Produce exactly one file: submission.csv
  - Make zero network calls
  - Make zero LLM API calls
  - Load all pre-computed artifacts from disk (embeddings, features, reasoning)
  - Print progress to stdout (stage name + timestamp) for debugging

rank.py must NOT:
  - Re-compute SBERT embeddings at runtime (these are pre-loaded from .npy files)
  - Re-generate reasoning at runtime (loaded from reasoning.json)
  - Download any model at runtime (models must be cached locally)
  - Use any GPU-specific libraries
```

### 1.2 `precompute.py` runs offline

```
precompute.py must:
  - Run all pre-computation: Stage 0, Stage 2, Stage 3, Stage 8
  - Be idempotent (running twice produces identical artifacts)
  - Save all outputs to artifacts/ directory with exact filenames in architecture.md
  - Print estimated time remaining for each stage

precompute.py has no time constraint.
```

---

## 2. Stage Ordering Rules

**Do not reorder stages. Do not merge stages. Do not skip stages.**

The pipeline order is:
```
Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6 → Stage 7 → Stage 8
```

Stage 1 (hard filters + honeypot gate) MUST execute before Stage 4 (scoring). This is non-negotiable. Running scoring on candidates that will be zeroed wastes compute and is an architectural error.

---

## 3. Field Name Rules

**Use exact field names from `architecture.md` Section "Dataset Schema Reference".** Do not invent field aliases.

Forbidden patterns:
```python
# WRONG — invented alias
candidate["yoe"]          # use profile["years_of_experience"]
candidate["bio"]          # use profile["summary"]
entry["months"]           # use career_history[i]["duration_months"]
signals["active"]         # use redrob_signals["last_active_date"]

# CORRECT
candidate["profile"]["years_of_experience"]
candidate["career_history"][i]["duration_months"]
candidate["redrob_signals"]["last_active_date"]
```

If a field is missing in a candidate record (rare but possible), use a safe default:
```python
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
```

---

## 4. Honeypot Rules

**These rules are absolute. They override all other scoring logic.**

### 4.1 A honeypotted candidate MUST have `final_score = 0.0`

No partial scoring. No "softened" penalty. If `honeypot_flag == True`, the score is exactly 0.0.

### 4.2 Honeypot detection runs in Stage 1, before any embedding computation

Do not compute embeddings for honeypotted candidates. Filter them first.

### 4.3 Honeypot threshold for submission safety

Before writing submission.csv, run this check:
```python
top_100 = final_df.head(100)
honeypot_count = top_100["honeypot_flag"].sum()
if honeypot_count > 0:
    raise RuntimeError(
        f"ABORT: {honeypot_count} honeypot(s) in top 100. "
        "This would trigger competition disqualification. "
        "Fix honeypot detection before submitting."
    )
```

### 4.4 All five honeypot checks must be implemented

From `architecture.md` Stage 1 Pass B:
- [ ] Claimed YOE > actual career span check
- [ ] Working before graduation year check
- [ ] Expert skill with 0-month duration check
- [ ] Expert/advanced claim with assessment score < 25 check
- [ ] Claimed YOE > 40 check

Implementing fewer than all five is incomplete.

---

## 5. Scoring Formula Rules

### 5.1 Do not change the weights in Stage 4

The exact formula from `architecture.md` Stage 4:
```python
base_score = (
    0.35 * title_career_score
  + 0.30 * skills_score
  + 0.20 * experience_score
  + 0.10 * location_edu_score
  + 0.05 * embedding_score
)
```

Do not adjust these weights without a documented, evidence-based reason. If you want to experiment with weights, create a `rank_experimental.py` — never modify the main `rank.py` weights without justification.

### 5.2 Required skills must be penalised 3× harder than nice-to-have skills

```python
# CORRECT
raw_skills_score = (3.0 * required_score + 1.0 * nth_score) / 4.0

# WRONG — equal weighting
raw_skills_score = (required_score + nth_score) / 2.0
```

### 5.3 The behavioral modifier is multiplicative, not additive

```python
# CORRECT
final_base_score = base_score * behavioral_modifier

# WRONG — behavioral signals added directly to base score
final_base_score = base_score + 0.15 * behavioral_raw_score
```

### 5.4 The behavioral modifier MUST be clipped to [0.4, 1.2]

```python
behavioral_modifier = max(0.4, min(1.2, raw_behavioral))
```

No candidate can receive a behavioral modifier below 0.4 or above 1.2.

### 5.5 Experience score counts product company experience only

Do not include months at consulting/IT services companies in `product_yoe`. Use the `CONSULTING_INDUSTRIES` set from `architecture.md` Stage 1 to filter.

---

## 6. Embedding Rules

### 6.1 Use `all-MiniLM-L6-v2` only

Do not substitute a different SBERT model without explicit approval. This model is:
- ~40MB on disk
- CPU-fast (25ms/batch)
- Produces 384-dimensional vectors

### 6.2 Dual-track embedding is mandatory

Both tracks must be computed and stored:
- `artifacts/career_embeddings.npy` — shape `(N, 384)`
- `artifacts/skills_embeddings.npy` — shape `(N, 384)`

Do not combine them into a single embedding at pre-computation time. They are blended at scoring time.

### 6.3 The embedding combination formula is fixed

```python
embedding_score = 0.65 * career_sim + 0.35 * skills_sim
```

### 6.4 Embed career descriptions, not just skill tags

The primary embedding target is `career_history[*].description` concatenated with title and company context. Not `profile.summary`. Not `skills[*].name` alone.

```python
# CORRECT primary embedding input
career_text = " ".join(
    f"{entry['title']} at {entry['company']}: {entry['description']}"
    for entry in career_history
)

# WRONG — summary-only embedding misses evidence
career_text = profile["summary"]
```

### 6.5 Normalize all embeddings

```python
vector = sbert_model.encode(text, normalize_embeddings=True)
```

Unnormalized vectors will produce incorrect cosine similarities.

---

## 7. Cross-Encoder Rules

### 7.1 Cross-encoder runs only on top 500 candidates

Never run the cross-encoder on the full 100K or full 28K. It is computationally expensive and the architecture specifically limits it to top-500.

### 7.2 Use the specified model

Primary: `cross-encoder/ms-marco-MiniLM-L-6-v2`  
Fallback: `BAAI/bge-reranker-base`

### 7.3 Normalize cross-encoder scores before blending

```python
cross_norm = (cross_scores - cross_scores.min()) / (cross_scores.max() - cross_scores.min() + 1e-9)
```

Raw cross-encoder logits are not on a [0,1] scale. Do not blend unnormalized scores with `final_base_score`.

### 7.4 Final score blend formula is fixed

```python
final_score = 0.55 * final_base_score + 0.45 * cross_norm
```

---

## 8. Reasoning Generation Rules

### 8.1 Reasoning is pre-computed offline in `precompute.py`

`rank.py` loads `artifacts/reasoning.json` and maps candidate_id to reasoning string. It does not call any LLM at ranking time.

### 8.2 The anti-hallucination check is mandatory

Every generated reasoning string must pass entity verification before being saved. The check must verify:
- Every skill mentioned exists in `skills[*].name`
- Every company mentioned exists in `career_history[*].company`
- Every year/duration claim is consistent with the profile

Strings that fail verification must be either regenerated (once) or replaced with the fallback template.

### 8.3 The fallback template must be used when LLM is unavailable

```python
FALLBACK_TEMPLATE = (
    "{title} with {yoe:.1f} yrs experience; "
    "{n_required} of {total_required} required skills matched; "
    "response rate {response_rate:.0%}."
)
```

This is acceptable output. Template reasoning is not ideal but is better than hallucinated reasoning.

### 8.4 Reasoning must NEVER contain invented facts

Forbidden patterns in reasoning strings:
```
"Built a RAG system at [company not in profile]"   # hallucinated company
"10 years of experience in..."                     # if profile says 6 years
"Expert in PyTorch"                                # if PyTorch not in skills
```

### 8.5 Reasoning length limit

Max 300 characters per reasoning string. Enforce with:
```python
reasoning = reasoning[:297] + "..." if len(reasoning) > 300 else reasoning
```

---

## 9. Output Format Rules

### 9.1 Submission CSV must have exactly these columns in this order

```
candidate_id,rank,score,reasoning
```

### 9.2 Exactly 100 rows, ranks 1–100

No more, no fewer. Rank 1 is the best candidate. Rank 100 is the 100th best.

### 9.3 Score format

4 decimal places, descending, in range [0.0, 1.0]:
```python
df["score"] = df["final_score"].round(4)
```

Scores must be strictly decreasing (no ties):
```python
# Break ties by candidate_id to ensure reproducibility
df = df.sort_values(["final_score", "candidate_id"], ascending=[False, True])
df["rank"] = range(1, 101)
```

### 9.4 Validate before writing

Run `validate_submission.py` (provided in dataset) before finalising. The script checks column names, row count, score range, and rank sequence. If it fails, fix the issue — do not override the validator.

---

## 10. Prohibited Patterns

The following are explicitly forbidden:

```python
# 1. No LLM API calls during ranking
import openai                          # FORBIDDEN in rank.py
import anthropic                       # FORBIDDEN in rank.py
requests.get("https://api.openai.com") # FORBIDDEN

# 2. No model downloads at ranking time
snapshot_download("bert-base-uncased") # FORBIDDEN in rank.py (pre-download in precompute.py)
AutoModel.from_pretrained("...")        # FORBIDDEN in rank.py unless model is cached locally

# 3. No hardcoded candidate IDs
if candidate_id == "CAND_0001234":     # FORBIDDEN — pipeline must generalise
    score = 1.0

# 4. No skipping the honeypot gate
df["final_score"] = compute_scores(df) # FORBIDDEN if honeypot_flag not zeroed first

# 5. No LTR without labels
xgb.train(params, dtrain, ...)         # FORBIDDEN — no training labels exist
lgb.train(params, train_data, ...)     # FORBIDDEN

# 6. No template reasoning that ignores actual profile content
reasoning = f"Strong candidate with relevant skills."  # FORBIDDEN — not grounded
```

---

## 11. Performance Constraints

### 11.1 Stage timing targets (within the 5-minute budget)

| Stage | Target time |
|---|---|
| Load artifacts from disk | < 30 seconds |
| Stage 1: Hard filters | < 5 seconds |
| Stage 4: Weighted scoring (28K rows) | < 10 seconds |
| Stage 5: Behavioral modifier (28K rows) | < 10 seconds |
| Stage 6: Sort + top-500 selection | < 5 seconds |
| Stage 7: Cross-encoder (500 pairs) | < 2 minutes |
| Final sort, reasoning join, CSV write | < 10 seconds |
| **Total** | **< 5 minutes** |

### 11.2 Memory targets

| Artifact | Estimated size |
|---|---|
| `career_embeddings.npy` (100K × 384 float32) | ~150 MB |
| `skills_embeddings.npy` (100K × 384 float32) | ~150 MB |
| `features.parquet` | ~50 MB |
| `reasoning.json` | < 1 MB |
| Cross-encoder model | ~80 MB |
| SBERT model (if loaded) | ~90 MB |
| **Total peak RAM estimate** | ~700 MB (well within 16GB) |

### 11.3 Use batch processing for embeddings

```python
# CORRECT — batched
embeddings = sbert_model.encode(texts, batch_size=256, show_progress_bar=True)

# WRONG — one by one (100K iterations, 10x slower)
embeddings = [sbert_model.encode(text) for text in texts]
```

---

## 12. Code Quality Rules

### 12.1 Every stage must be a separate importable module

```python
# pipeline/stage1_filters.py
def apply_hard_filters(df: pd.DataFrame, jd_parsed: dict) -> pd.DataFrame: ...
def apply_honeypot_gate(df: pd.DataFrame) -> pd.DataFrame: ...
```

No monolithic 1000-line `rank.py`. Each stage is testable in isolation.

### 12.2 Log stage boundaries

```python
import time
print(f"[{time.strftime('%H:%M:%S')}] Stage 7: Cross-encoder reranking on {len(top_500)} candidates...")
# ... do work ...
print(f"[{time.strftime('%H:%M:%S')}] Stage 7: Complete. Top 100 selected.")
```

### 12.3 Fail loudly, never silently

```python
# CORRECT — explicit failure
assert len(submission_df) == 100, f"Expected 100 rows, got {len(submission_df)}"
assert submission_df["rank"].tolist() == list(range(1, 101)), "Ranks must be 1-100 sequential"

# WRONG — silent continuation
if len(submission_df) != 100:
    submission_df = submission_df.head(100)  # silently truncates without explanation
```

### 12.4 Reproducibility

Set random seeds where any stochastic process is used:
```python
import random, numpy as np
random.seed(42)
np.random.seed(42)
```

The pipeline must produce identical output given identical input.

---

## 13. Pre-Submission Checklist

Run through every item before submitting:

- [ ] `validate_submission.py` passes with zero errors
- [ ] `submission.csv` has exactly 100 rows
- [ ] Column order: `candidate_id, rank, score, reasoning`
- [ ] Ranks are sequential integers 1–100
- [ ] Scores are in descending order, range [0.0, 1.0], 4 decimal places
- [ ] No honeypotted candidate in top 100 (honeypot safety check passes)
- [ ] No reasoning string exceeds 300 characters
- [ ] No reasoning string contains the words "seems", "appears", "likely", "probably"
- [ ] `rank.py` completes in under 5 minutes on CPU without network
- [ ] `precompute.py` has been run and all `artifacts/` files exist
- [ ] `submission_metadata_template.yaml` is filled out completely
- [ ] Git history has meaningful commits (not a single "initial commit")
