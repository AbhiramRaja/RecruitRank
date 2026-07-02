# RecruitRank — Architecture Specification
**Intelligent Candidate Ranking System | India Runs Data & AI Challenge**

> *Yogyata (योग्यता) — merit. We rank on evidence, not keywords.*

---

## Overview

RecruitRank is an 8-stage candidate ranking pipeline that processes 100,000 candidate profiles against a Job Description and outputs a top-100 ranked CSV with grounded reasoning. The pipeline is designed around three core principles:

1. **NDCG@10 maximisation** — getting the top-10 right matters more than perfect ordering of ranks 50–100.
2. **Honeypot immunity** — fake/impossible profiles are zeroed before any scoring computation.
3. **Explainability without hallucination** — every reasoning string is fact-checked against the source profile.

---

## Dataset Schema Reference

All field names used in this pipeline are derived from the official `data/candidate_schema.json`.

### Top-level keys (required)
```
candidate_id         string   Format: CAND_XXXXXXX (7 digits)
profile              object   See below
career_history       array    1–10 items
education            array
skills               array
redrob_signals       object   See below
```

### `profile` fields used
```
profile.years_of_experience     number    Claimed total YOE
profile.current_title           string    Current job title
profile.current_company         string
profile.current_company_size    enum      "1-10" … "10001+"
profile.current_industry        string
profile.location                string    City, region
profile.country                 string
profile.headline                string
profile.summary                 string
```

### `career_history[*]` fields used
```
career_history[].company          string
career_history[].title            string
career_history[].start_date       date     YYYY-MM-DD
career_history[].end_date         date     YYYY-MM-DD or null
career_history[].duration_months  integer
career_history[].is_current       boolean
career_history[].industry         string
career_history[].company_size     string
career_history[].description      string   PRIMARY EMBEDDING TARGET
```

### `skills[*]` fields used
```
skills[].name            string
skills[].proficiency     enum     "beginner" | "intermediate" | "advanced" | "expert"
skills[].endorsements    integer
skills[].duration_months integer
```

### `education[*]` fields used
```
education[].institution     string
education[].degree          string
education[].field_of_study  string
education[].end_year        integer   graduation year
education[].tier            string    "tier_1" | "tier_2" | "tier_3"
```

### `redrob_signals` fields used
```
redrob_signals.last_active_date             date
redrob_signals.open_to_work_flag            boolean
redrob_signals.notice_period_days           integer   0–180
redrob_signals.recruiter_response_rate      float     0.0–1.0
redrob_signals.avg_response_time_hours      float
redrob_signals.profile_views_received_30d  integer
redrob_signals.search_appearance_30d       integer
redrob_signals.saved_by_recruiters_30d     integer
redrob_signals.interview_completion_rate   float     0.0–1.0
redrob_signals.offer_acceptance_rate       float     -1 = no history
redrob_signals.verified_email              boolean
redrob_signals.verified_phone              boolean
redrob_signals.skill_assessment_scores     dict      {skill_name: 0–100}
redrob_signals.github_activity_score       float     -1 = not linked, 0–100
redrob_signals.profile_completeness_score  float     0–100
redrob_signals.willing_to_relocate         boolean
redrob_signals.preferred_work_mode         enum      "remote"|"hybrid"|"onsite"|"flexible"
```

---

## Output Format

The final submission is a CSV with exactly these columns, exactly 100 rows, ranked 1–100:

```
candidate_id, rank, score, reasoning
CAND_0000001, 1, 0.9720, "Built production ranking system at Flipkart (4 yrs product co); strong SBERT + vector DB background from career history; open to work, notice period 15 days."
```

- `score` is a float in [0.0, 1.0], 4 decimal places, descending
- `reasoning` is a single sentence, max 300 characters, grounded in profile facts only

---

## Runtime Constraints

| Constraint | Value |
|---|---|
| Pre-computation (embeddings, feature extraction) | Unbounded — runs offline once |
| Ranking step (rank.py execution) | ≤ 5 minutes wall clock |
| RAM | ≤ 16 GB |
| GPU | None — CPU only |
| Network | None — fully offline |
| LLM API calls during ranking | Forbidden — reasoning pre-computed |

---

## Pipeline Architecture

```
candidates.jsonl (100K) + job_description.md
          │
          ▼
┌─────────────────────────────────────────────────┐
│  STAGE 0  JD Parsing (offline, once)            │
│  Split JD → required skills / nice-to-have /    │
│  hard disqualifier keywords                     │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STAGE 1  Hard Filters + Honeypot Gate          │
│  100K → ~28K (milliseconds)                     │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STAGE 2  Feature Extraction (offline)          │
│  Title/career · Skills · XP · Location/Edu     │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STAGE 3  Dual-Track SBERT Embedding (offline)  │
│  Career descriptions + Skills, stored to disk   │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STAGE 4  Weighted Score Combiner               │
│  base_score = weighted sum of feature scores    │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STAGE 5  Behavioral Signal Modifier            │
│  final_base = base_score × clamp(beh, 0.4, 1.2)│
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STAGE 6  ANN Retrieval → Top 500               │
│  Sort by final_base score, take top 500         │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STAGE 7  Cross-Encoder Reranking               │
│  ms-marco-MiniLM-L-6-v2 on top 500 → top 100   │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STAGE 8  Reasoning Generation (offline)        │
│  Phi-3-mini · grounded prompt · entity check    │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
             submission CSV (100 rows)
```

---

## Stage-by-Stage Specification

---

### Stage 0 — JD Parsing

**Runs:** Offline, once, before any candidate processing.  
**Input:** `job_description.md`  
**Output:** Three Python structures saved to `artifacts/jd_parsed.json`

Parse the JD into three buckets:

```python
jd_parsed = {
    "required_skills": [...],       # Hard must-haves. Missing = penalty × 3.0
    "nice_to_have_skills": [...],   # Preferences. Missing = penalty × 1.0
    "hard_disqualifiers": {
        "wrong_domains": ["computer vision", "speech recognition", "robotics", "NLP-only", "TTS"],
        "wrong_company_types": ["consulting_only"],  # see Stage 1
        "location_required": "India",
        "exclude_pure_research": True
    },
    "jd_embedding_text": "..."      # Full JD text, pre-embedded as JD vector
}
```

Embed `jd_embedding_text` using `all-MiniLM-L6-v2` and save the JD vector to `artifacts/jd_vector.npy`.

---

### Stage 1 — Hard Filters + Honeypot Gate

**Runs:** Inline at ranking time (first step in `rank.py`).  
**Input:** All 100K candidates  
**Output:** Surviving candidates (~28K) with honeypots zeroed

This stage runs in two sequential passes on the raw data. No embedding, no ML — pure Python logic.

#### Pass A: Hard Filters (discard entirely)

Remove candidate from consideration if ANY of the following are true:

```python
# 1. Wrong country
candidate.profile.country != "India"

# 2. Consulting-only career
# ALL career_history entries have industry in:
ALL(entry.industry in CONSULTING_INDUSTRIES for entry in career_history)
# CONSULTING_INDUSTRIES = {"IT Services", "Consulting", "Outsourcing", "BPO", "KPO", "Staffing"}

# 3. Wrong domain (title clearly outside scope)
current_title_lower contains any of:
["marketing", "sales", "accountant", "civil engineer", "mechanical engineer",
 "graphic designer", "hr manager", "customer support", "content writer",
 "operations manager", "project manager (non-tech)"]
# Use substring match. If title is ambiguous (e.g. "Project Manager - ML Platform"), keep.

# 4. Zero technical skills
len([s for s in skills if s.name in TECH_SKILLS_LIST]) == 0
```

Discarded candidates are excluded from all further processing. They do not appear in output CSV.

#### Pass B: Honeypot Detection (zero the score, keep in dataset)

Flag a candidate as a honeypot (set `honeypot_flag = True`, `final_score = 0.0`) if ANY of the following are true:

```python
# 1. Impossible experience: claimed YOE > actual career span
claimed_yoe = profile.years_of_experience
earliest_start = min(entry.start_date for entry in career_history)
actual_span_years = (TODAY - earliest_start).days / 365.25
if claimed_yoe > actual_span_years + 2:   # +2 yr tolerance
    flag_honeypot()

# 2. Impossible graduation: working before degree completed
grad_year = min(edu.end_year for edu in education)
earliest_job_year = min(entry.start_date.year for entry in career_history)
if earliest_job_year < grad_year - 1:     # -1 yr tolerance for final-year internships
    flag_honeypot()

# 3. Expert skill with 0-month duration
for skill in skills:
    if skill.proficiency == "expert" and skill.duration_months == 0:
        flag_honeypot()

# 4. Expert/advanced claim with very low assessment score
for skill in skills:
    assessment = redrob_signals.skill_assessment_scores.get(skill.name)
    if assessment is not None:
        if skill.proficiency in ("expert", "advanced") and assessment < 25:
            flag_honeypot()

# 5. Claimed YOE exceeds biological maximum (working age check)
if claimed_yoe > 40:
    flag_honeypot()
```

Honeypotted candidates receive `final_score = 0.0` and rank > 900 in output if included, but are NOT included in the top-100 output.

---

### Stage 2 — Feature Extraction

**Runs:** Offline pre-computation. Output saved to `artifacts/features.parquet`.  
**Input:** ~28K surviving candidates  
**Output:** One row per candidate with all computed feature columns

Compute four feature groups per candidate:

#### 2A. Title + Career Score (`title_career_score`, range 0.0–1.0)

```python
score = 0.0

# Base: is current title in target scope?
TARGET_TITLES = [
    "machine learning engineer", "ml engineer", "ai engineer",
    "data scientist", "research engineer", "applied scientist",
    "software engineer", "backend engineer", "platform engineer",
    "mlops engineer", "llm engineer", "nlp engineer"
]
title_lower = profile.current_title.lower()
if any(t in title_lower for t in TARGET_TITLES):
    score += 0.4

# Product company bonus: how many months at product companies?
product_months = sum(
    entry.duration_months
    for entry in career_history
    if entry.company_size in ("201-500", "501-1000", "1001-5000", "5001-10000", "10001+")
    and entry.industry not in CONSULTING_INDUSTRIES
)
score += min(product_months / 60, 1.0) * 0.4   # caps at 5 yrs product exp

# Seniority bonus
SENIOR_KEYWORDS = ["senior", "lead", "principal", "staff", "head of", "director"]
if any(k in title_lower for k in SENIOR_KEYWORDS):
    score += 0.2

return min(score, 1.0)
```

#### 2B. Skills Score (`skills_score`, range 0.0–1.0)

```python
required_skills = jd_parsed["required_skills"]      # from Stage 0
nice_to_have    = jd_parsed["nice_to_have_skills"]  # from Stage 0

candidate_skill_names = {s.name.lower() for s in skills}

# Required skill coverage (weighted 3× heavier)
required_matched = sum(
    1 for rs in required_skills
    if rs.lower() in candidate_skill_names
)
required_score = required_matched / len(required_skills) if required_skills else 0.0

# Nice-to-have coverage
nth_matched = sum(
    1 for nth in nice_to_have
    if nth.lower() in candidate_skill_names
)
nth_score = nth_matched / len(nice_to_have) if nice_to_have else 0.0

# Endorsement trust multiplier
avg_endorsements = mean(s.endorsements for s in skills) if skills else 0
endorsement_trust = min(avg_endorsements / 20, 1.0)   # caps at 20 endorsements

raw_skills_score = (3.0 * required_score + 1.0 * nth_score) / 4.0
return raw_skills_score * (0.7 + 0.3 * endorsement_trust)
```

#### 2C. Experience Score (`experience_score`, range 0.0–1.0)

Piecewise function — target band is 4–10 years, product companies only.

```python
# Only count experience at product companies (not consulting)
product_yoe = sum(
    entry.duration_months / 12
    for entry in career_history
    if entry.industry not in CONSULTING_INDUSTRIES
    and entry.company_size not in ("1-10", "11-50")  # exclude tiny startups
)

# Piecewise scoring
if product_yoe < 2:
    score = 0.1
elif product_yoe < 4:
    score = 0.1 + (product_yoe - 2) / 2 * 0.4   # 0.1 → 0.5 ramp
elif product_yoe <= 10:
    score = 0.5 + (product_yoe - 4) / 6 * 0.5   # 0.5 → 1.0 ramp (sweet spot)
else:
    score = 1.0 - min((product_yoe - 10) / 10, 0.2)  # soft cap, max -0.2

return score
```

#### 2D. Location + Education Score (`location_edu_score`, range 0.0–1.0)

```python
score = 0.0

# Location bonus (JD targets Pune, Noida, NCR, Bangalore, Hyderabad, Mumbai)
TARGET_CITIES = ["pune", "noida", "gurugram", "gurgaon", "delhi", "ncr",
                 "bangalore", "bengaluru", "hyderabad", "mumbai", "chennai"]
loc_lower = profile.location.lower()
if any(city in loc_lower for city in TARGET_CITIES):
    score += 0.5
elif redrob_signals.willing_to_relocate:
    score += 0.3

# Education tier bonus
best_tier = min(edu.tier for edu in education)   # tier_1 < tier_2 < tier_3 alphabetically
if best_tier == "tier_1":
    score += 0.5
elif best_tier == "tier_2":
    score += 0.3
else:
    score += 0.1

return min(score, 1.0)
```

---

### Stage 3 — Dual-Track SBERT Embedding

**Runs:** Offline pre-computation. Output saved to `artifacts/`.  
**Model:** `sentence-transformers/all-MiniLM-L6-v2` (CPU, ~40MB)

Two embedding tracks run in parallel:

#### Track A: Career Description Embedding
```python
# Concatenate all career history descriptions
career_text = " ".join(
    f"{entry.title} at {entry.company}: {entry.description}"
    for entry in career_history
)
career_vector = sbert_model.encode(career_text, normalize_embeddings=True)
# Save: artifacts/career_embeddings.npy  shape: (N, 384)
```

#### Track B: Skills Embedding
```python
# Weighted skill text — required skills get 3× repetition
skill_text = " ".join(
    ([skill.name] * 3 if skill.name.lower() in required_skills_lower else [skill.name])
    for skill in skills
)
skills_vector = sbert_model.encode(skill_text, normalize_embeddings=True)
# Save: artifacts/skills_embeddings.npy  shape: (N, 384)
```

#### JD Vector (computed once in Stage 0)
```python
jd_vector = sbert_model.encode(jd_parsed["jd_embedding_text"], normalize_embeddings=True)
# Save: artifacts/jd_vector.npy  shape: (384,)
```

#### Combined Similarity Score (computed at ranking time)
```python
career_sim = cosine_similarity(career_vector, jd_vector)   # float
skills_sim = cosine_similarity(skills_vector, jd_vector)   # float
embedding_score = 0.65 * career_sim + 0.35 * skills_sim
```

---

### Stage 4 — Weighted Score Combiner

**Runs:** Inline in `rank.py` — pure numpy/pandas, no ML.  
**Formula:**

```python
base_score = (
    0.35 * title_career_score     # career depth signal
  + 0.30 * skills_score           # required vs nice-to-have, endorsement-weighted
  + 0.20 * experience_score       # piecewise, product co only
  + 0.10 * location_edu_score     # location match + institution tier
  + 0.05 * embedding_score        # dual-track SBERT similarity
)
```

Weight rationale:
- `title_career_score` (0.35): Most predictive of actual fit. Product company background is the primary JD requirement.
- `skills_score` (0.30): Required skill coverage is critical. Endorsed skills with assessment scores are trusted.
- `experience_score` (0.20): 4–10 years product co experience is the sweet spot per JD.
- `location_edu_score` (0.10): Location matters for onsite roles; institution tier is a signal.
- `embedding_score` (0.05): Catches semantic matches the structured features miss, but structured features dominate.

---

### Stage 5 — Behavioral Signal Modifier

**Runs:** Inline in `rank.py`.  
**Design:** Multiplicative modifier, not additive. Clips to [0.4, 1.2].

```python
def compute_behavioral_modifier(signals):
    score = 1.0  # neutral baseline

    # Availability (can they start soon?)
    days_since_active = (TODAY - signals.last_active_date).days
    if signals.open_to_work_flag:
        score += 0.08
    if signals.notice_period_days <= 30:
        score += 0.05
    if days_since_active <= 7:
        score += 0.05
    elif days_since_active <= 30:
        score += 0.02
    elif days_since_active > 90:
        score -= 0.10

    # Engagement quality
    if signals.recruiter_response_rate >= 0.7:
        score += 0.06
    elif signals.recruiter_response_rate <= 0.2:
        score -= 0.08
    if signals.avg_response_time_hours <= 4:
        score += 0.03

    # Recruiter interest (social proof)
    if signals.saved_by_recruiters_30d >= 5:
        score += 0.05
    if signals.search_appearance_30d >= 10:
        score += 0.03

    # Trust + reliability
    if signals.interview_completion_rate >= 0.8:
        score += 0.06
    elif signals.interview_completion_rate < 0.4:
        score -= 0.10
    if signals.verified_email and signals.verified_phone:
        score += 0.04
    if signals.offer_acceptance_rate > 0 and signals.offer_acceptance_rate >= 0.8:
        score += 0.04

    # GitHub activity (engineering signal)
    if signals.github_activity_score >= 70:
        score += 0.04
    elif signals.github_activity_score == -1:
        pass  # no penalty for no GitHub

    return max(0.4, min(1.2, score))   # hard clip


final_base_score = base_score * compute_behavioral_modifier(redrob_signals)
```

---

### Stage 6 — ANN Retrieval → Top 500

**Runs:** Inline in `rank.py`.

```python
# Sort all non-honeypot candidates by final_base_score descending
ranked = df[df.honeypot_flag == False].sort_values("final_base_score", ascending=False)
top_500 = ranked.head(500)
```

This is not ANN in the traditional sense (no FAISS needed at this scale — numpy sort is fast enough on 28K rows). The name reflects the conceptual role: recall phase narrowing to 500 for the precision phase.

---

### Stage 7 — Cross-Encoder Reranking

**Runs:** Inline in `rank.py`. Operates only on top 500 candidates.  
**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (or `BAAI/bge-reranker-base`)

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Build input pairs: (JD text, candidate career text)
pairs = [
    (jd_text, candidate_career_text[cid])
    for cid in top_500.candidate_id
]

# Score all 500 pairs jointly
cross_scores = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)

top_500["cross_score"] = cross_scores

# Final combined score: weighted blend of base score and cross-encoder score
# Normalize cross_scores to [0,1] first
cross_norm = (cross_scores - cross_scores.min()) / (cross_scores.max() - cross_scores.min() + 1e-9)
top_500["final_score"] = 0.55 * top_500["final_base_score"] + 0.45 * cross_norm

# Sort and take top 100
top_100 = top_500.sort_values("final_score", ascending=False).head(100)
top_100["rank"] = range(1, 101)
```

---

### Stage 8 — Reasoning Generation

**Runs:** Offline pre-computation. Output saved to `artifacts/reasoning.json`.  
**Model:** `microsoft/Phi-3-mini-4k-instruct` (GGUF quantised via `llama-cpp-python`) or `TinyLlama-1.1B-Chat`  
**Scope:** Top 100 candidates only (after Stage 7 finalises the list)

#### Grounded Prompt Template

```python
SYSTEM_PROMPT = """You are a technical recruiter. Write exactly one sentence (max 250 characters)
explaining why this candidate is ranked at their position. You MUST only mention facts that
explicitly appear in the candidate profile provided. Do not invent skills, companies,
or experience. Do not use the words "seems", "appears", or "likely"."""

USER_PROMPT = """
Rank: {rank}
Candidate title: {current_title}
Years of experience: {years_of_experience}
Career history (relevant): {career_summary}
Matched required skills: {matched_required}
Missing required skills: {missing_required}
Open to work: {open_to_work}
Notice period: {notice_period_days} days
Interview completion rate: {interview_completion_rate}

Write one grounded reasoning sentence for this rank.
"""
```

#### Anti-Hallucination Check

After generation, run entity verification:

```python
def verify_reasoning(reasoning_text, candidate_profile):
    """
    Extract all nouns/entities from reasoning_text.
    Verify each appears in the candidate's raw profile text.
    Return (is_valid, flagged_entities).
    """
    profile_text = build_profile_text(candidate_profile).lower()
    
    # Extract candidate-specific claims (skills, companies, numbers)
    import re
    years_claimed = re.findall(r'\d+\s*(?:years?|yrs?)', reasoning_text.lower())
    skills_claimed = extract_skill_mentions(reasoning_text)
    companies_claimed = extract_company_mentions(reasoning_text)
    
    hallucinations = []
    for skill in skills_claimed:
        if skill.lower() not in profile_text:
            hallucinations.append(skill)
    for company in companies_claimed:
        if company.lower() not in profile_text:
            hallucinations.append(company)
    
    return len(hallucinations) == 0, hallucinations


# If hallucination detected: regenerate once. If still hallucinating: use fallback template.
FALLBACK_TEMPLATE = (
    "{title} with {yoe:.1f} yrs experience; "
    "{n_required} of {total_required} required skills matched; "
    "response rate {response_rate:.0%}."
)
```

---

## File Structure

```
yogyata_ai/
├── rank.py                    # ENTRY POINT — produces submission.csv in ≤5min
├── precompute.py              # Run offline: embeddings, features, reasoning
├── pipeline/
│   ├── stage0_jd_parser.py
│   ├── stage1_filters.py
│   ├── stage2_features.py
│   ├── stage3_embeddings.py
│   ├── stage4_scorer.py
│   ├── stage5_behavioral.py
│   ├── stage6_retrieval.py
│   ├── stage7_crossencoder.py
│   └── stage8_reasoning.py
├── artifacts/                 # Pre-computed outputs (gitignored if large)
│   ├── features.parquet
│   ├── career_embeddings.npy
│   ├── skills_embeddings.npy
│   ├── jd_vector.npy
│   ├── jd_parsed.json
│   └── reasoning.json
├── data/
│   ├── candidates.jsonl
│   └── job_description.md
├── requirements.txt
└── README.md
```

---

## Dependencies

```txt
sentence-transformers>=2.6.0
transformers>=4.40.0
torch>=2.2.0          # CPU build is fine
pandas>=2.0.0
numpy>=1.26.0
pyarrow>=14.0.0       # for parquet
scikit-learn>=1.4.0
llama-cpp-python>=0.2.0   # for local LLM reasoning (optional, fallback template available)
tqdm>=4.66.0
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Honeypot gate before embedding | No wasted SBERT compute on zero-score candidates |
| Career descriptions as primary embedding target | Contains actual evidence of what was built, not just buzzwords |
| Dual-track embedding (career + skills) | Career catches semantic fit; skills catches explicit keyword matches |
| Multiplicative behavioral modifier clipped [0.4, 1.2] | Behavioral signals scale merit, cannot manufacture it |
| Required vs nice-to-have split with 3× weight | Asymmetric JD reading matches how real recruiters think |
| Cross-encoder only on top 500 | Joint (candidate, JD) scoring too expensive on 100K; perfectly fast on 500 |
| Local LLM with entity verification | Judges will spot template reasoning; hallucination check prevents fabrication |
| Weighted fusion not LTR | No training labels exist in this dataset; transparent weights are auditable |
