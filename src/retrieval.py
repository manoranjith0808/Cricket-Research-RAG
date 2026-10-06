"""
retrieval.py
------------
STEP 5 of the pipeline: given a Chroma vectorstore (+ the raw chunks),
build retrievers that implement two different retrieval strategies so
we can compare them, as required by the capstone brief.

Strategy 1 - Dense retrieval (cosine similarity)
    The baseline: embed the query, return the chunks whose vectors are
    closest by cosine distance. Fast, simple, and works well when the
    query and the relevant passage use similar wording/phrasing.

Strategy 2 - Hybrid retrieval (BM25 keyword search + dense retrieval)
    BM25 is a classic sparse/keyword scoring algorithm: it rewards exact
    term overlap between the query and a chunk, weighted by how rare
    that term is across the whole corpus. This is strong for queries
    that use a specific technical term (e.g. "powerplay", "DRS") that
    MUST appear literally in the right chunk, but weak for queries that
    are phrased differently from the source text. Combining both (via a
    rank-fusion EnsembleRetriever-style wrapper) gives you both:
        - the semantic recall of dense retrieval
        - the exact-term precision of BM25
"""

from typing import Any, Dict, List, Tuple
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

from src.config import TOP_K


class HybridRetriever:
    """Compatibility wrapper for BM25 + dense retrieval.

    The older ``EnsembleRetriever`` import path is brittle across LangChain
    versions and can fail even when the underlying retrievers are installed.
    This local implementation combines the two retrievers by rank fusion so the
    app can keep the same public behavior without depending on that class.
    """

    def __init__(self, bm25_retriever, dense_retriever, k: int = TOP_K, dense_weight: float = 0.5):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.k = k
        self.dense_weight = dense_weight

    @staticmethod
    def _doc_key(doc: Document) -> Tuple[Tuple[str, Any], ...]:
        metadata = tuple(sorted((str(k), v) for k, v in (doc.metadata or {}).items()))
        return (doc.page_content, metadata)

    def invoke(self, query: str) -> List[Document]:
        bm25_docs = self.bm25_retriever.invoke(query)[: self.k]
        dense_docs = self.dense_retriever.invoke(query)[: self.k]

        combined: Dict[Tuple[Tuple[str, Any], ...], Dict[str, Any]] = {}

        for rank, doc in enumerate(bm25_docs):
            key = self._doc_key(doc)
            score = (1.0 - self.dense_weight) / (rank + 1)
            if key not in combined:
                combined[key] = {"doc": doc, "score": 0.0}
            combined[key]["score"] += score

        for rank, doc in enumerate(dense_docs):
            key = self._doc_key(doc)
            score = self.dense_weight / (rank + 1)
            if key not in combined:
                combined[key] = {"doc": doc, "score": 0.0}
            combined[key]["score"] += score

        ranked = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
        return [entry["doc"] for entry in ranked[: self.k]]

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)


def get_dense_retriever(vectorstore: Chroma, k: int = TOP_K):
    """Strategy 1: plain cosine-similarity retrieval straight from Chroma."""
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


def get_hybrid_retriever(vectorstore: Chroma, chunks: List[Document], k: int = TOP_K,
                          dense_weight: float = 0.5):
    """
    Strategy 2: hybrid retrieval combining BM25 (keyword) with dense
    (cosine) retrieval via rank fusion.

    `chunks` must be the SAME list of Documents that was embedded into
    `vectorstore` -- BM25Retriever builds its own in-memory keyword
    index directly from these Documents (it does not touch the vector
    store at all).
    """
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    dense_retriever = get_dense_retriever(vectorstore, k=k)

    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        k=k,
        dense_weight=dense_weight,
    )
    return hybrid_retriever


def compare_retrievers(query: str, dense_retriever, hybrid_retriever) -> None:
    """
    Side-by-side print used in the notebook's retrieval-comparison cell.
    Look for cases where the two strategies disagree -- that is exactly
    the kind of observation the rubric wants documented.
    """
    print(f"Query: {query!r}\n")

    print("Dense (cosine) results:")
    for doc in dense_retriever.invoke(query):
        print(f"  - {doc.metadata['paper_title']} (p.{doc.metadata['page_number']}): "
              f"{doc.page_content[:100]}...")

    print("\nHybrid (BM25 + dense) results:")
    for doc in hybrid_retriever.invoke(query):
        print(f"  - {doc.metadata['paper_title']} (p.{doc.metadata['page_number']}): "
              f"{doc.page_content[:100]}...")
