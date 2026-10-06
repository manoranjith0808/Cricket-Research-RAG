"""
chunking.py
-----------
STEP 2 of the pipeline: split page-level Documents into smaller chunks.

Why RecursiveCharacterTextSplitter?
    It tries to split on paragraph breaks first, then sentences, then
    words -- only falling back to a hard character cut if nothing else
    fits. This keeps chunks semantically coherent instead of cutting
    mid-sentence, which is what a naive fixed-size splitter would do.

CHUNK_SIZE / CHUNK_OVERLAP trade-off (documented for the viva):
    - Too small (e.g. 100 chars): chunks lose surrounding context, so the
      LLM may misinterpret an isolated sentence.
    - Too large (e.g. 3000 chars): a chunk mixes multiple ideas, which
      dilutes its embedding and makes retrieval less precise -- the
      chunk's vector becomes an "average" of several topics instead of
      a sharp representation of one.
    - CHUNK_OVERLAP (a small % of chunk size) prevents losing a sentence
      that happens to fall exactly on a chunk boundary.
    We use 700 characters / 120 overlap as a reasonable middle ground for
    prose-heavy research papers; tune this per your own dataset.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split page-level Documents into smaller chunk Documents.
    Metadata (paper_title, source_file, page_number) is automatically
    propagated to every chunk derived from a page, which is what lets us
    cite "paper title + page number" later -- this is not automatic magic,
    it is a property of how LangChain's splitter copies metadata, so do not
    remove metadata before this step.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # try paragraph, then line, then sentence, then word
    )
    chunks = splitter.split_documents(docs)

    # Give every chunk a stable, human-readable ID -- useful for debugging
    # and for showing "which exact chunk" fed into an answer.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source_file']}::p{chunk.metadata['page_number']}::c{i}"

    return chunks


def inspect_chunks(chunks: List[Document]) -> None:
    """Sanity-check print used in the notebook after chunking."""
    print(f"Total chunks created: {len(chunks)}")
    lengths = [len(c.page_content) for c in chunks]
    print(f"Avg chunk length: {sum(lengths) / len(lengths):.0f} chars "
          f"(min {min(lengths)}, max {max(lengths)})")
    print("\nExample chunk:")
    print(f"  chunk_id: {chunks[0].metadata['chunk_id']}")
    print(f"  paper:    {chunks[0].metadata['paper_title']} (page {chunks[0].metadata['page_number']})")
    print(f"  text:     {chunks[0].page_content[:200]}...")


if __name__ == "__main__":
    from src.data_loading import load_all_papers

    docs = load_all_papers()
    chunks = chunk_documents(docs)
    inspect_chunks(chunks)
