# data/

This directory is the spec-compliant location for candidate data files.

Per `architecture.md`, the pipeline expects:

```
data/
  candidates.jsonl   ← Full competition dataset (JSONL, 100K+ records)
  candidates.json    ← Alternative JSON array format
  sample_candidates.json ← Small sample for testing
  candidate_schema.json ← Official schema definition
  job_description.docx ← Job description source file
  README.docx ← Original readme document
  redrob_signals_doc.docx ← Documentation for redrob signals
```

## Current Status

The data files have been successfully moved to this directory, matching the architecture spec. Both `rank.py` and `precompute.py` will find them automatically when run with `--data-dir data` (or by default, as they search `data/` automatically).

Search order:
1. `data/candidates.jsonl`   ← current location
2. `data/candidates.json`
3. `<project_root>/candidates.jsonl`  ← fallback
4. `<project_root>/data/sample_candidates.json`  ← testing fallback
