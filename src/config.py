"""
config.py
---------
Single source of truth for every setting used across the project.
Change values here rather than hunting through every file.
"""

import os
from dotenv import load_dotenv

# --- Paths -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOTENV_PATH = os.path.join(BASE_DIR, ".env")
PAPERS_DIR = os.path.join(BASE_DIR, "data", "papers")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# Load environment variables from the project root regardless of the
# current working directory. This makes the notebook, app, and tests
# behave consistently when launched from different folders.
load_dotenv(DOTENV_PATH, override=False)

# Disable Chroma anonymous telemetry to avoid noisy startup warnings in
# local/dev environments.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE = 700          # characters per chunk
CHUNK_OVERLAP = 120       # characters shared between consecutive chunks

# --- Embedding models ------------------------------------------------------
# 1) Open-source, runs locally, free, no API key needed
HF_EMBEDDING_MODEL = "BAAI/bge-m3"
HF_COLLECTION_NAME = "cricket_papers_bge_m3"

# 2) Commercial, called over the Google Gemini API, needs GEMINI_API_KEY
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
GEMINI_COLLECTION_NAME = "cricket_papers_gemini_embed"

# --- Generation LLM (Google Gemini) ----------------------------------------
DEFAULT_GEMINI_MODEL = "3.1-flash"
GEMINI_MODEL_OPTIONS = ["3.1-flash-lite", "3.1-flash", "3.1-pro"]

GEMINI_MODEL_MAP = {
    "3.1-flash-lite": "gemini-3.6-flash-lite",
    "3.1-flash": "gemini-3.6-flash",
    "3.1-pro": "gemini-3.6-pro",
}

def resolve_gemini_model(model_name: str) -> str:
    """Resolve friendly model selection labels to official Gemini model identifiers."""
    if not model_name:
        return GEMINI_MODEL_MAP.get(DEFAULT_GEMINI_MODEL, "gemini-2.5-flash")
    return GEMINI_MODEL_MAP.get(model_name, model_name)

# --- Retrieval ---------------------------------------------------------
TOP_K = 3                # how many chunks to retrieve / show as sources
HYBRID_DENSE_WEIGHT = 0.5  # weight given to dense score in hybrid search (rest goes to BM25)

# --- API keys -------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
