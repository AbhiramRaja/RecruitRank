"""
Shared constants for the RecruitRank pipeline.

Centralised here to avoid copy-paste across stage files.
All stage modules should import from here rather than re-defining.
"""

import datetime

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Fixed "TODAY" for all YOE / date calculations — ensures deterministic output.
# Used in Stage 1 (honeypot gate) and Stage 5 (behavioral modifier).
COMPETITION_DATE = datetime.date(2026, 6, 30)

# ---------------------------------------------------------------------------
# Industry classification (architecture.md Stage 1)
# ---------------------------------------------------------------------------
CONSULTING_INDUSTRIES = {
    "IT Services", "Consulting", "Outsourcing", "BPO", "KPO", "Staffing"
}

# ---------------------------------------------------------------------------
# Field defaults (instructions.md Section 3)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Title / keyword lists (Stage 1 filters + Stage 2 feature extraction)
# ---------------------------------------------------------------------------
WRONG_DOMAIN_TITLES = [
    "marketing", "sales", "accountant", "civil engineer", "mechanical engineer",
    "graphic designer", "hr manager", "customer support", "content writer",
    "operations manager", "project manager (non-tech)",
]

TARGET_TITLES = [
    "machine learning engineer", "ml engineer", "ai engineer",
    "data scientist", "research engineer", "applied scientist",
    "software engineer", "backend engineer", "platform engineer",
    "mlops engineer", "llm engineer", "nlp engineer",
]

SENIOR_KEYWORDS = ["senior", "lead", "principal", "staff", "head of", "director"]

# Cities targeted by JD (Pune/Noida primary; Bangalore/NCR/etc accepted)
TARGET_CITIES = [
    "pune", "noida", "gurugram", "gurgaon", "delhi", "ncr",
    "bangalore", "bengaluru", "hyderabad", "mumbai", "chennai",
]

# ---------------------------------------------------------------------------
# SBERT model config (instructions.md Section 6.1: use all-MiniLM-L6-v2 only)
# ---------------------------------------------------------------------------
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 256   # instructions.md Section 11.3
