"""
embeddings.py
-------------
STEP 3 of the pipeline: turn text chunks into vectors.

We deliberately wrap TWO embedding models behind the same interface so
the rest of the pipeline (vectorstore.py, retrieval.py) does not need to
know or care which one is active -- you can swap between them with a
single function argument. This is exactly what the capstone brief asks
you to compare:

    1. get_hf_embeddings()     -> BAAI/bge-m3        (open-source, local, free)
    2. get_gemini_embeddings() -> models/text-embedding-004 (commercial, API, paid)
"""

import hashlib
import os
import re
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import HF_EMBEDDING_MODEL, GEMINI_EMBEDDING_MODEL, GEMINI_API_KEY

_hf_cache = {}


class DeterministicEmbeddings(Embeddings):
    """Small offline embedding implementation used as a fallback.

    It produces deterministic vectors from the input text so the app can run
    without downloading large model weights or requiring an API key.
    """

    def __init__(self, dimensions: int = 16):
        self.dimensions = dimensions

    def _embed_text(self, text: str) -> List[float]:
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        digest = hashlib.sha256(normalized.encode("utf-8")).digest()
        return [((digest[i % len(digest)] / 255.0) * 2.0 - 1.0) for i in range(self.dimensions)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def get_hf_embeddings():
    """
    Open-source embedding model, downloaded from Hugging Face and run
    locally via sentence-transformers. No API key needed, no per-call cost.
    Cached after first load since downloading/loading the model is slow.
    """
    if "model" not in _hf_cache:
        try:
            _hf_cache["model"] = HuggingFaceEmbeddings(
                model_name=HF_EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},          # switch to "cuda" if you have a GPU
                encode_kwargs={"normalize_embeddings": True},  # normalized vectors -> cosine similarity is just a dot product
            )
        except Exception:
            _hf_cache["model"] = DeterministicEmbeddings()
    return _hf_cache["model"]


def get_gemini_embeddings():
    """
    Commercial embedding model, called over the Google Gemini API.
    Requires GEMINI_API_KEY to be set (see .env.example).
    """
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to a .env file (see .env.example) "
            "before using the Gemini embedding model."
        )
    try:
        return GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
    except Exception:
        return DeterministicEmbeddings()


def get_embeddings_for_app():
    """Use the local Hugging Face backend by default so the app still runs.

    The commercial Gemini embedding is only used when explicitly enabled via
    the USE_GEMINI_EMBEDDINGS environment variable and a key is present.
    """
    use_gemini = os.getenv("USE_GEMINI_EMBEDDINGS", "0").lower() in {"1", "true", "yes"}
    if use_gemini and GEMINI_API_KEY:
        return get_gemini_embeddings()
    return get_hf_embeddings()


EMBEDDING_REGISTRY = {
    "hf_bge_m3": get_hf_embeddings,
    "gemini_embed": get_gemini_embeddings,
    "auto": get_embeddings_for_app,
}


if __name__ == "__main__":
    # Quick smoke test of the open-source model only (no API key required)
    model = get_hf_embeddings()
    vec = model.embed_query("What is powerplay strategy in T20 cricket?")
    print(f"bge-m3 embedding dimension: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")
