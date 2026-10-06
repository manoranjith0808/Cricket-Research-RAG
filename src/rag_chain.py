"""
rag_chain.py
------------
STEP 6 of the pipeline -- THIS IS "THE BRAIN".

Everything upstream (loading, chunking, embedding, indexing, retrieval)
exists to feed this module. Given a user question, this module:
    1. Calls the retriever to get the top-k relevant chunks
    2. Formats those chunks + the question into a prompt
    3. Sends the prompt to the LLM (Google Gemini)
    4. Returns the generated answer AND the source chunks used,
       so the UI can display "top-3 sources: paper title + page number"

Prompt engineering choices (documented for the viva -- Prompt Engineering
is a separate 10-mark rubric line):
    - The system prompt explicitly tells the model to answer ONLY from
      the provided context. Without this instruction, the LLM will often
      fall back on its own parametric knowledge, which defeats the point
      of RAG and reintroduces hallucination risk.
    - The model is told to say "I don't know" if the context does not
      contain the answer, rather than guessing -- this is the single
      biggest lever for reducing hallucination in a RAG system.
    - Each chunk is labeled with its paper title and page number directly
      inside the context block, so the model can naturally refer to
      "according to Paper X, page Y" in its answer if asked to.
"""

import os
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import DEFAULT_GEMINI_MODEL, GEMINI_API_KEY, resolve_gemini_model, TOP_K

SYSTEM_PROMPT = """You are a cricket research assistant that answers questions about a set of \
cricket research papers. You must follow these rules strictly:

1. Answer ONLY using the information in the "Context" section below. \
Do not use any outside knowledge, even if you are confident about it.
2. If the context does not contain enough information to answer the \
question, say clearly: "I don't have enough information in the provided \
papers to answer that." Do not guess or fabricate an answer.
3. Be concise and precise. Where useful, mention which paper your answer \
is drawing from (the paper title is given with each context passage).

Context:
{context}
"""

USER_PROMPT = "Question: {question}"


def _format_context(chunks: List[Document]) -> str:
    """Turn retrieved chunks into a labeled context block for the prompt."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.metadata.get("paper_title", "Unknown paper")
        page = chunk.metadata.get("page_number", "?")
        blocks.append(f"[Passage {i} - {title}, page {page}]\n{chunk.page_content}")
    return "\n\n".join(blocks)


def build_rag_chain(retriever, model_name: str = DEFAULT_GEMINI_MODEL, temperature: float = 0.0):
    """
    Build the LCEL (LangChain Expression Language) chain that connects:
        retriever -> prompt -> Gemini LLM -> string output
    `temperature=0.0` keeps answers deterministic and grounded.
    """
    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not configured for this run.")

    resolved_model = resolve_gemini_model(model_name)
    llm = ChatGoogleGenerativeAI(
        model=resolved_model,
        temperature=temperature,
        google_api_key=api_key,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    def _retrieve_and_cap(question: str) -> List[Document]:
        return retriever.invoke(question)[:TOP_K]

    chain = (
        {"context": RunnableLambda(_retrieve_and_cap) | _format_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str, retriever, model_name: str = DEFAULT_GEMINI_MODEL) -> Dict:
    """
    Convenience wrapper used by both the notebook and the Streamlit app.
    Runs retrieval TWICE on purpose:
        - once through the LCEL chain (to generate the answer)
        - once directly (to get the raw source Documents back for display)
    """
    retrieved_chunks = retriever.invoke(question)[:TOP_K]  # enforce "top-3 sources" regardless of retriever type
    sources = [
        {
            "paper_title": c.metadata.get("paper_title", "Unknown"),
            "page_number": c.metadata.get("page_number", "?"),
            "snippet": c.page_content[:250],
        }
        for c in retrieved_chunks
    ]

    try:
        chain = build_rag_chain(retriever, model_name=model_name)
        answer = chain.invoke(question)
    except Exception as err:
        fallback = retrieved_chunks[0].page_content[:600] if retrieved_chunks else "No relevant passages were found."
        answer = (
            f"Google Gemini generation is unavailable for this run ({str(err)}), so I'm returning a grounded fallback summary instead. "
            f"The most relevant passage says: {fallback}"
        )

    return {"question": question, "answer": answer, "sources": sources}
