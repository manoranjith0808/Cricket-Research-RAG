"""
data_loading.py
----------------
STEP 1 of the pipeline: turn PDF files on disk into LangChain `Document`
objects, one per page, each carrying metadata (source file name + page
number) that we need later for citations.

Why PyPDFLoader?
    It is the simplest LangChain loader and, importantly, it returns one
    Document PER PAGE automatically with `page` already set in metadata --
    exactly the granularity we need for "cite the page number" later.
    (PDFPlumberLoader is a good alternative if your PDFs have tables you
    care about, since it preserves layout better.)
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from src.config import PAPERS_DIR


def load_all_papers(papers_dir: str = PAPERS_DIR) -> List[Document]:
    """
    Load every PDF in `papers_dir` and return a flat list of per-page
    LangChain Documents. Each Document.metadata will contain:
        - source: the original filename (used as "paper title" downstream)
        - page:   the 0-indexed page number from PyPDFLoader
    """
    all_docs: List[Document] = []
    pdf_files = sorted(f for f in os.listdir(papers_dir) if f.lower().endswith(".pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {papers_dir}. "
            f"Run data/generate_sample_papers.py first, or drop your own cricket PDFs in that folder."
        )

    for filename in pdf_files:
        filepath = os.path.join(papers_dir, filename)
        loader = PyPDFLoader(filepath)
        pages = loader.load()  # one Document per page

        for doc in pages:
            # Store a clean paper title separately from the raw file path,
            # so citations look like "paper_1_cricket_analytics.pdf" -> readable title
            doc.metadata["paper_title"] = _filename_to_title(filename)
            doc.metadata["source_file"] = filename
            # PyPDFLoader's "page" is 0-indexed; store a 1-indexed version for display
            doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1

        all_docs.extend(pages)

    return all_docs


def _filename_to_title(filename: str) -> str:
    """Turn 'paper_1_cricket_analytics.pdf' into 'Paper 1 Cricket Analytics'."""
    name = os.path.splitext(filename)[0]
    return name.replace("_", " ").title()


def inspect_documents(docs: List[Document]) -> None:
    """Quick sanity-check print used in the notebook after loading."""
    print(f"Loaded {len(docs)} pages total.")
    titles = sorted(set(d.metadata["paper_title"] for d in docs))
    print(f"From {len(titles)} paper(s):")
    for t in titles:
        n_pages = sum(1 for d in docs if d.metadata["paper_title"] == t)
        print(f"  - {t} ({n_pages} pages)")
    print("\nSample of first document's metadata:")
    print(docs[0].metadata)
    print("\nSample text (first 300 chars):")
    print(docs[0].page_content[:300])


if __name__ == "__main__":
    docs = load_all_papers()
    inspect_documents(docs)
