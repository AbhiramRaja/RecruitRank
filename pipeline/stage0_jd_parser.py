"""
Stage 0 — JD Parsing

Runs: Offline, once, before any candidate processing.
Input: Job Description text (from job_description.docx)
Output: jd_parsed dict saved to artifacts/jd_parsed.json

Parses the JD into three buckets (required_skills, nice_to_have_skills,
hard_disqualifiers) plus a jd_embedding_text string for SBERT embedding.
"""

import json
import os


def parse_jd() -> dict:
    """
    Parse the Job Description into structured scoring signals.

    The JD is for: Senior AI Engineer — Founding Team at Redrob AI.
    All skill lists are curated to match plausible candidate skill names
    in the dataset (skills[].name field, matched case-insensitively).

    Returns:
        dict matching the architecture.md Stage 0 output spec.
    """

    jd_parsed = {
        # ----------------------------------------------------------------
        # REQUIRED SKILLS — Hard must-haves.  Missing = penalty × 3.0
        #
        # Source: JD section "Things you absolutely need"
        #   1. Production embeddings-based retrieval systems
        #   2. Production vector databases / hybrid search infrastructure
        #   3. Strong Python
        #   4. Evaluation frameworks for ranking systems
        #
        # These are the specific skill names (matched case-insensitively
        # against skills[].name) that represent the JD's non-negotiables.
        # ----------------------------------------------------------------
        "required_skills": [
            # --- Core programming ---
            "Python",

            # --- ML / AI foundations ---
            "Machine Learning",
            "Deep Learning",
            "NLP",

            # --- Embeddings & retrieval (JD must-have #1) ---
            "Sentence Transformers",
            "SBERT",
            "Embeddings",
            "Information Retrieval",

            # --- Vector DBs & search infra (JD must-have #2) ---
            "FAISS",
            "Pinecone",
            "Weaviate",
            "Qdrant",
            "Milvus",
            "Elasticsearch",
            "OpenSearch",
            "Vector Database",

            # --- ML frameworks (implied by production ML requirement) ---
            "PyTorch",
            "TensorFlow",
            "Transformers",
            "Hugging Face",

            # --- Ranking & evaluation (JD must-have #4) ---
            "Ranking Systems",
            "Search Systems",
            "Recommendation Systems",

            # --- LLMs (prominent in JD role description) ---
            "LLMs",
            "Large Language Models",
            "RAG",
        ],

        # ----------------------------------------------------------------
        # NICE-TO-HAVE SKILLS — Preferences.  Missing = penalty × 1.0
        #
        # Source: JD section "Things we'd like you to have but won't
        # reject you for"
        #   - LLM fine-tuning (LoRA, QLoRA, PEFT)
        #   - Learning-to-rank models (XGBoost-based or neural)
        #   - HR-tech / recruiting tech / marketplace products
        #   - Distributed systems / large-scale inference optimization
        #   - Open-source contributions in AI/ML
        # ----------------------------------------------------------------
        "nice_to_have_skills": [
            # --- LLM fine-tuning ---
            "Fine-tuning LLMs",
            "LoRA",
            "QLoRA",
            "PEFT",

            # --- Learning-to-rank ---
            "XGBoost",
            "LightGBM",
            "Learning to Rank",

            # --- Infra & optimization ---
            "Distributed Systems",
            "Kubernetes",
            "Docker",
            "MLOps",
            "Model Optimization",
            "TensorRT",
            "ONNX",

            # --- Data & pipeline ---
            "Spark",
            "Airflow",
            "Data Engineering",

            # --- Cloud platforms ---
            "AWS",
            "GCP",
            "Azure",

            # --- Adjacent ML skills ---
            "Scikit-learn",
            "Pandas",
            "NumPy",
            "Statistical Modeling",
            "Feature Engineering",

            # --- Serving & deployment ---
            "FastAPI",
            "Flask",
            "Model Serving",
            "BentoML",
            "Triton",

            # --- Experiment tracking ---
            "Weights & Biases",
            "MLflow",
        ],

        # ----------------------------------------------------------------
        # HARD DISQUALIFIERS
        #
        # Source: JD sections "Things we explicitly do NOT want" and
        # "What we mean by 5-9 years" (the disqualifier paragraphs)
        # ----------------------------------------------------------------
        "hard_disqualifiers": {
            # Domains that are out of scope per JD:
            # "People whose primary expertise is computer vision, speech,
            #  or robotics without significant NLP/IR exposure"
            "wrong_domains": [
                "computer vision",
                "speech recognition",
                "robotics",
                "NLP-only",    # pure NLP without retrieval/ranking
                "TTS",         # text-to-speech
            ],

            # Consulting-only careers are explicitly called out:
            # "People who have only worked at consulting firms (TCS, Infosys,
            #  Wipro, Accenture, Cognizant, Capgemini, etc.)"
            # Note: prior product-company experience overrides this.
            "wrong_company_types": [
                "consulting_only",
            ],

            # JD says: Pune/Noida India (Hybrid), open to relocation from
            # Tier-1 Indian cities. "Outside India: case-by-case, but we
            # don't sponsor work visas."
            "location_required": "India",

            # JD says: "If you've spent your career in pure research
            # environments (academic labs, research-only roles) without
            # any production deployment — we will not move forward."
            "exclude_pure_research": True,
        },

        # ----------------------------------------------------------------
        # JD EMBEDDING TEXT
        #
        # The full semantic content of the JD, formatted for SBERT
        # embedding (all-MiniLM-L6-v2).  This captures:
        #   - Role title and mandate
        #   - Core technical requirements
        #   - What the person will actually do
        #   - The "ideal candidate" profile
        #
        # This is embedded once in Stage 0 and compared against each
        # candidate's career_text and skills_text embeddings in Stage 3.
        # ----------------------------------------------------------------
        "jd_embedding_text": (
            "Senior AI Engineer at an AI-native talent intelligence platform. "
            "Own the intelligence layer: ranking, retrieval, and matching systems "
            "that decide what recruiters see when they search for candidates. "

            "Ship a v2 ranking system using embeddings, hybrid retrieval, and "
            "LLM-based re-ranking. Set up evaluation infrastructure with offline "
            "benchmarks, online A/B testing, and recruiter-feedback loops. "
            "Drive long-term architecture for candidate-JD matching at scale. "

            "Must have production experience with embeddings-based retrieval "
            "systems such as sentence-transformers, BGE, or E5 deployed to real "
            "users. Must have production experience with vector databases or "
            "hybrid search infrastructure including Pinecone, Weaviate, Qdrant, "
            "Milvus, FAISS, or Elasticsearch. Strong Python and code quality. "
            "Hands-on experience designing evaluation frameworks for ranking "
            "systems using NDCG, MRR, MAP, and A/B test interpretation. "

            "Ideal candidate: 6-8 years total experience, 4-5 years in applied "
            "ML/AI roles at product companies. Has shipped at least one "
            "end-to-end ranking, search, or recommendation system to real users "
            "at meaningful scale. Strong opinions about retrieval architecture, "
            "evaluation methodology, and LLM integration. "

            "Nice to have: LLM fine-tuning with LoRA, QLoRA, or PEFT. "
            "Learning-to-rank models. HR-tech or marketplace product experience. "
            "Distributed systems and large-scale inference optimization. "
            "Open-source contributions in AI/ML."
        ),
    }

    return jd_parsed


def save_jd_parsed(jd_parsed: dict, artifacts_dir: str = "artifacts") -> str:
    """
    Save the parsed JD to artifacts/jd_parsed.json.

    Args:
        jd_parsed: The parsed JD dict from parse_jd().
        artifacts_dir: Directory to save artifacts into.

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    output_path = os.path.join(artifacts_dir, "jd_parsed.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(jd_parsed, f, indent=2, ensure_ascii=False)

    print(f"[Stage 0] Saved jd_parsed.json -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Standalone execution: parse JD and save to artifacts/
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print(f"[{time.strftime('%H:%M:%S')}] Stage 0: JD Parsing — start")

    parsed = parse_jd()

    # Resolve artifacts dir relative to project root (RecruitRank/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_jd_parsed(parsed, artifacts_dir=os.path.join(project_root, "artifacts"))

    # Summary
    print(f"  Required skills   : {len(parsed['required_skills'])} items")
    print(f"  Nice-to-have      : {len(parsed['nice_to_have_skills'])} items")
    print(f"  Hard disqualifiers: {list(parsed['hard_disqualifiers'].keys())}")
    print(f"  Embedding text    : {len(parsed['jd_embedding_text'])} chars")
    print(f"[{time.strftime('%H:%M:%S')}] Stage 0: JD Parsing — complete")
