"""
Shared utility functions for the RecruitRank pipeline.

Functions here are imported by multiple stage modules to eliminate
code duplication (DRY principle, instructions.md Section 12.1).
"""

import datetime
from typing import Any, Dict, List

from pipeline.constants import COMPETITION_DATE, FIELD_DEFAULTS


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_date(date_str: str) -> datetime.date:
    """
    Parse a YYYY-MM-DD (or YYYY-MM or YYYY) date string to datetime.date.
    Handles ISO-8601 timestamps by stripping the time component.
    Returns COMPETITION_DATE on any parse failure.

    Used in Stage 1 (honeypot gate) and Stage 5 (behavioral modifier).
    """
    if not date_str:
        return COMPETITION_DATE
    try:
        parts = date_str.split("T")[0].split("-")
        if len(parts) == 3:
            return datetime.date(int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            return datetime.date(int(parts[0]), int(parts[1]), 1)
        elif len(parts) == 1:
            return datetime.date(int(parts[0]), 1, 1)
        return COMPETITION_DATE
    except Exception:
        return COMPETITION_DATE


# ---------------------------------------------------------------------------
# Safe field access
# ---------------------------------------------------------------------------

def get_field(obj: Dict, key: str, default: Any = None) -> Any:
    """
    Safely get a field from a dict, falling back to FIELD_DEFAULTS then
    to an explicit default.  Treats None values as missing.

    Per instructions.md Section 3: use safe defaults; never crash on
    missing fields.
    """
    val = obj.get(key)
    if val is None:
        return FIELD_DEFAULTS.get(key, default)
    return val


# ---------------------------------------------------------------------------
# Candidate text builders (used by Stage 2 storage + Stage 3 embedding)
# ---------------------------------------------------------------------------

def build_career_text(candidate: Dict) -> str:
    """
    Build the career embedding / cross-encoder input text for one candidate.

    Per architecture.md Stage 3, Track A:
        career_text = " ".join(
            f"{entry['title']} at {entry['company']}: {entry['description']}"
            for entry in career_history
        )

    Per instructions.md Section 6.4: embed career descriptions, not just
    skill tags. The primary embedding target is career_history[*].description
    concatenated with title and company context.
    """
    career_history = candidate.get("career_history", [])
    parts: List[str] = []
    for entry in career_history:
        title = entry.get("title", "")
        company = entry.get("company", "")
        description = entry.get("description", "")
        parts.append(f"{title} at {company}: {description}")
    return " ".join(parts) if parts else ""


def build_skills_text(candidate: Dict, required_skills_lower: set) -> str:
    """
    Build the skills embedding input text for one candidate.

    Per architecture.md Stage 3, Track B:
        skill_text = " ".join(
            ([skill.name] * 3 if skill.name.lower() in required_skills_lower
             else [skill.name])
            for skill in skills
        )

    Required skills are repeated 3× to boost their semantic weight
    in the embedding space, giving them stronger signal vs the JD vector.
    """
    skills = candidate.get("skills", [])
    tokens: List[str] = []
    for skill in skills:
        name = skill.get("name", "")
        if not name:
            continue
        if name.lower() in required_skills_lower:
            tokens.extend([name] * 3)
        else:
            tokens.append(name)
    return " ".join(tokens) if tokens else ""
