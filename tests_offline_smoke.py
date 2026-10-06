"""
tests_offline_smoke.py
-----------------------
NOT part of the deliverable. This sandbox environment cannot reach
huggingface.co or generativelanguage.googleapis.com, so this script substitutes a tiny
deterministic hash-based "fake embedding" purely to prove the plumbing
(chunking -> Chroma indexing -> dense retrieval -> BM25 hybrid retrieval)
works end-to-end with zero network calls.

When you run the real notebook on your own machine / Colab (which has
full internet access), you will use the real get_hf_embeddings() /
get_gemini_embeddings() from src/embeddings.py instead of this stand-in.
"""

import hashlib
import numpy as np
from src.data_loading import load_all_papers
from src.chunking import chunk_documents
from src.vectorstore import build_vectorstore
from src.retrieval import get_dense_retriever, get_hybrid_retriever, compare_retrievers

DIM = 384


class FakeDeterministicEmbeddings:
    """Turns text into a reproducible pseudo-random vector via hashing.
    Good enough to prove Chroma indexing/search plumbing works; NOT a
    real semantic embedding model."""

    def _embed(self, text: str):
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=DIM)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


print("=" * 70)
print("STEP 1: Load PDFs")
print("=" * 70)
docs = load_all_papers()
print(f"Loaded {len(docs)} pages\n")

print("=" * 70)
print("STEP 2: Chunk documents")
print("=" * 70)
chunks = chunk_documents(docs)
print(f"Created {len(chunks)} chunks\n")

print("=" * 70)
print("STEP 3 & 4: Embed (fake, offline) + index into Chroma")
print("=" * 70)
fake_embeddings = FakeDeterministicEmbeddings()
vectorstore = build_vectorstore(chunks, fake_embeddings, collection_name="smoke_test_collection")
print("Indexed into Chroma successfully\n")

print("=" * 70)
print("STEP 5: Retrieval strategies")
print("=" * 70)
dense = get_dense_retriever(vectorstore, k=3)
hybrid = get_hybrid_retriever(vectorstore, chunks, k=3)

compare_retrievers("What is the acute:chronic workload ratio?", dense, hybrid)
print()
compare_retrievers("How does hybrid retrieval combine BM25 and dense search?", dense, hybrid)

print("\nAll plumbing checks passed. (Retrieval quality itself will look")
print("random here since this uses a FAKE embedding -- real semantic")
print("quality only shows up once you run this with bge-m3 / Gemini")
print("embeddings on your own machine.)")
