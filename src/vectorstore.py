"""
vectorstore.py
--------------
STEP 4 of the pipeline: store embedded chunks in ChromaDB and load them
back later.

Why ChromaDB over FAISS?
    - Chroma stores each chunk's metadata (paper_title, page_number,
      chunk_id) alongside its vector natively, so a similarity search
      returns ready-to-cite results directly.
    - FAISS only stores raw vectors -- you would need to maintain a
      separate parallel list/dict mapping vector index -> metadata
      yourself, which is extra bookkeeping for no real benefit at the
      scale of a handful of PDFs.
    - Chroma persists to a local folder automatically, so you do not
      need to re-embed every time you restart the app.

We build ONE Chroma collection PER embedding model (see config.py),
so the two collections can be compared side-by-side without either one
overwriting the other.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import CHROMA_DIR


def build_vectorstore(chunks: List[Document], embedding_function, collection_name: str) -> Chroma:
    """
    Embed `chunks` with `embedding_function` and persist them into a
    Chroma collection named `collection_name` under CHROMA_DIR.
    Safe to call again later -- Chroma will just open the existing
    collection if it already has data (see load_vectorstore below for
    the "load without re-embedding" path).
    """
    persist_path = os.path.join(CHROMA_DIR, collection_name)
    os.makedirs(persist_path, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        collection_name=collection_name,
        persist_directory=persist_path,
    )
    return vectorstore


def load_vectorstore(embedding_function, collection_name: str) -> Chroma:
    """
    Reopen an already-indexed Chroma collection without re-embedding
    anything -- use this in the Streamlit app so it starts instantly
    instead of re-running embedding on every launch.
    """
    persist_path = os.path.join(CHROMA_DIR, collection_name)
    if not os.path.exists(persist_path):
        raise FileNotFoundError(
            f"No existing collection at {persist_path}. "
            f"Run build_vectorstore() first (see notebook Step 4)."
        )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_path,
    )


def validate_retrieval(vectorstore: Chroma, query: str, k: int = 3) -> None:
    """Quick sanity-check print: does a similarity search return sensible chunks?"""
    results = vectorstore.similarity_search_with_score(query, k=k)
    print(f"Query: {query!r}")
    for doc, score in results:
        print(f"  [score={score:.4f}] {doc.metadata['paper_title']} "
              f"(p.{doc.metadata['page_number']}): {doc.page_content[:120]}...")
