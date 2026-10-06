"""
app.py
------
Streamlit UI for the Cricket Research Answer Bot (Stretch Goal / Advanced
Option 2 from the capstone brief).

Run with:
    streamlit run app.py

This file contains NO pipeline logic of its own -- it only calls into
src/ modules, exactly the same ones used in the notebook. That keeps the
notebook and the app perfectly consistent and avoids duplicated code.

First run will be slower (building both vector collections from scratch);
after that, indexes are cached to disk (ChromaDB) and in Streamlit's
session cache, so subsequent questions are fast.
"""

import os

import streamlit as st

from src.data_loading import load_all_papers
from src.chunking import chunk_documents
from src.embeddings import get_hf_embeddings, get_gemini_embeddings, get_embeddings_for_app
from src.vectorstore import build_vectorstore
from src.retrieval import get_dense_retriever, get_hybrid_retriever
from src.rag_chain import ask
from src.config import HF_COLLECTION_NAME, GEMINI_COLLECTION_NAME, TOP_K, GEMINI_API_KEY, GEMINI_MODEL_OPTIONS

st.set_page_config(page_title="Cricket Research Answer Bot", page_icon="🏏", layout="wide")


# ---------------------------------------------------------------------------
# Build (or load) the pipeline once, cache it across reruns/questions.
# Streamlit reruns this whole script top-to-bottom on every interaction,
# so @st.cache_resource is essential -- without it, we would re-embed and
# re-index the entire corpus on every single question asked.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing cricket papers (first run only)...")
def load_pipeline():
    docs = load_all_papers()
    chunks = chunk_documents(docs)

    use_gemini = os.getenv("USE_GEMINI_EMBEDDINGS", "0").lower() in {"1", "true", "yes"}
    if use_gemini:
        embedding_function = get_gemini_embeddings()
        collection_name = GEMINI_COLLECTION_NAME
    else:
        embedding_function = get_hf_embeddings()
        collection_name = HF_COLLECTION_NAME

    vectorstore = build_vectorstore(chunks, embedding_function, collection_name=collection_name)

    dense_retriever = get_dense_retriever(vectorstore, k=TOP_K)
    hybrid_retriever = get_hybrid_retriever(vectorstore, chunks, k=TOP_K)

    paper_titles = sorted(set(c.metadata["paper_title"] for c in chunks))
    return dense_retriever, hybrid_retriever, paper_titles, len(chunks), use_gemini


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🏏 Cricket Research Answer Bot")
st.sidebar.markdown(
    "A RAG system over a curated set of cricket research papers, built with "
    "LangChain + ChromaDB + **Google Gemini**."
)

strategy = st.sidebar.radio(
    "Retrieval strategy",
    options=["Hybrid (BM25 + dense)", "Dense (cosine only)"],
    help="Switch strategies live to compare how each affects retrieved sources and the final answer.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 LLM Model Selection")
selected_model = st.sidebar.selectbox(
    "Gemini Model Tier",
    options=GEMINI_MODEL_OPTIONS,
    index=1,
    help="Toggle between 3.1-flash-lite (fastest speed), 3.1-flash (balanced), and 3.1-pro (deep reasoning)."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Embedding model: local Hugging Face (default)\n\n"
    f"LLM: Google Gemini ({selected_model})\n\n"
    "Vector DB: ChromaDB (local)"
)

if not GEMINI_API_KEY:
    st.sidebar.warning("`GEMINI_API_KEY` is not configured in `.env`. Please add your key to enable generation.")
else:
    st.sidebar.success(f"Gemini key configured. Tier: `{selected_model}`")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Ask a question about the cricket research papers")

dense_retriever, hybrid_retriever, paper_titles, n_chunks, use_gemini = load_pipeline()

with st.expander(f"📄 Indexed papers ({n_chunks} chunks total)"):
    for t in paper_titles:
        st.write(f"- {t}")

question = st.text_input(
    "Your question",
    placeholder="e.g. Why is the acute:chronic workload ratio important for fast bowlers?",
)

ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    active_retriever = hybrid_retriever if strategy.startswith("Hybrid") else dense_retriever

    with st.spinner(f"Retrieving relevant passages and generating answer using Gemini ({selected_model})..."):
        result = ask(question, active_retriever, model_name=selected_model)

    st.markdown("### Answer")
    st.write(result["answer"])

    st.markdown(f"### Top-{TOP_K} Supporting Sources")
    for i, source in enumerate(result["sources"], start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. {source['paper_title']} -- page {source['page_number']}**")
            st.caption(source["snippet"] + "...")

elif ask_clicked:
    st.warning("Please type a question first.")

if not use_gemini:
    st.info("This run is using the local Hugging Face embedding backend, so retrieval works offline.")
