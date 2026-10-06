"""
evaluation/run_ragas_eval.py
-----------------------------
OPTIONAL. The capstone brief mentions you may "consider using automated
evaluation with DeepEval or another LLM-as-a-judge setup." This is a
starter script for RAGAS, which scores each (question, answer, retrieved
context) triple on:
    - faithfulness       : is the answer actually supported by the retrieved context?
    - answer_relevancy    : does the answer address the question asked?
    - context_precision   : how much of the retrieved context was actually useful?
    - context_recall      : did retrieval find everything needed to answer?

Not required for a passing submission, but a nice addition if you have
spare time -- it turns your Step 7 evaluation from "read it and judge by
eye" into a reproducible, numeric result you can show on a slide.

Install: pip install ragas datasets
"""

import sys, os
sys.path.append(os.path.abspath(".."))

from dotenv import load_dotenv
load_dotenv("../.env")

from src.data_loading import load_all_papers
from src.chunking import chunk_documents
from src.embeddings import get_gemini_embeddings
from src.vectorstore import build_vectorstore
from src.retrieval import get_hybrid_retriever
from src.rag_chain import ask
from src.config import GEMINI_COLLECTION_NAME, TOP_K

# The same 10+ test questions used in the notebook's Step 7.
# Add a ground-truth reference answer for each if you want context_recall too.
TEST_SET = [
    {"question": "What context-aware metrics are used instead of raw batting average?"},
    {"question": "What inputs does a win probability model typically use?"},
    {"question": "Why do win probabilities swing faster in T20 than in ODIs or Tests?"},
    {"question": "Which model types are commonly used for player performance prediction and why?"},
    {"question": "What is the delivery stride phase in fast bowling and why does it matter?"},
    {"question": "Why is a mixed bowling action associated with higher injury risk?"},
    {"question": "What is the acute:chronic workload ratio and why is it tracked?"},
    {"question": "Why is bowling variation important in the death overs?"},
    {"question": "How do teams use matchup analytics during a T20 innings?"},
    {"question": "What is the capital of France?"},  # deliberately out-of-scope
]


def main():
    docs = load_all_papers()
    chunks = chunk_documents(docs)
    embeddings = get_gemini_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings, collection_name=GEMINI_COLLECTION_NAME)
    retriever = get_hybrid_retriever(vectorstore, chunks, k=TOP_K)

    records = []
    for item in TEST_SET:
        result = ask(item["question"], retriever)
        records.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": [s["snippet"] for s in result["sources"]],
        })

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        dataset = Dataset.from_list(records)
        scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        print(scores)
        scores.to_pandas().to_csv("ragas_results.csv", index=False)
        print("Saved detailed results to evaluation/ragas_results.csv")

    except ImportError:
        print("ragas / datasets not installed. Run: pip install ragas datasets")
        print("\nRaw generated Q&A pairs (for manual scoring instead):\n")
        for r in records:
            print(f"Q: {r['question']}")
            print(f"A: {r['answer']}\n")


if __name__ == "__main__":
    main()
