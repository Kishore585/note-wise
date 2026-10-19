"""
Reproducible Empirical Information Retrieval Evaluation Script for NoteWise
Evaluates TF-IDF + Cosine Similarity retrieval against representative educational queries.
"""

import time
import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.pdf_service import extract_text_from_pdf
from utils.text_processing import create_chunks
from services.nlp_service import NLPService

# Representative queries with domain keywords and target chunk expectations
BENCHMARK_QUERIES = [
    {
        "query": "What is an artificial neuron and how is the weighted sum calculated?",
        "expected_keywords": ["neuron", "weighted", "bias", "activation"],
        "expected_chunk_ids": [1]
    },
    {
        "query": "Why are non-linear activation functions necessary in neural networks?",
        "expected_keywords": ["non-linear", "linear", "transformation", "collapse"],
        "expected_chunk_ids": [1, 2]
    },
    {
        "query": "What is the formula and definition of ReLU activation?",
        "expected_keywords": ["relu", "rectified", "gradient", "vanishing"],
        "expected_chunk_ids": [2]
    },
    {
        "query": "How is categorical cross-entropy loss formulated for classification?",
        "expected_keywords": ["cross-entropy", "loss", "softmax", "classification"],
        "expected_chunk_ids": [3]
    },
    {
        "query": "Explain backpropagation using the mathematical chain rule of calculus",
        "expected_keywords": ["backpropagation", "chain rule", "gradient", "loss"],
        "expected_chunk_ids": [4]
    },
    {
        "query": "What is the Adam optimizer and how does it combine momentum with RMSProp?",
        "expected_keywords": ["adam", "momentum", "optimizer", "learning rate"],
        "expected_chunk_ids": [5]
    },
    {
        "query": "What techniques prevent neural network overfitting on training noise?",
        "expected_keywords": ["overfitting", "regularization", "dropout", "weight decay"],
        "expected_chunk_ids": [6]
    },
    {
        "query": "How does L2 regularization penalize large weights?",
        "expected_keywords": ["regularization", "weight decay", "penalty", "weights"],
        "expected_chunk_ids": [6]
    },
    {
        "query": "What is dropout and what fraction of neurons is randomly deactivated?",
        "expected_keywords": ["dropout", "deactivates", "neurons", "co-adaptations"],
        "expected_chunk_ids": [6]
    },
    {
        "query": "Why does a learning rate that is too large cause divergence?",
        "expected_keywords": ["learning rate", "divergence", "optimizer", "gradient"],
        "expected_chunk_ids": [5, 7]
    }
]


def run_evaluation():
    print("Loading test notes: sample_data/machine_learning_notes.pdf...")
    pdf_res = extract_text_from_pdf("sample_data/machine_learning_notes.pdf")
    chunks = create_chunks(pdf_res.pages_data, chunk_size_words=120, chunk_overlap_words=25)
    print(f"Extracted {len(chunks)} semantic chunks across {pdf_res.stats['num_pages']} pages.")

    nlp = NLPService(chunks)

    # Benchmark retrieval speed
    start_time = time.perf_counter()
    for item in BENCHMARK_QUERIES:
        _ = nlp.retrieve_relevant_passages(item["query"], top_k=2)
    elapsed_ms = (time.perf_counter() - start_time) / len(BENCHMARK_QUERIES) * 1000

    results_k2 = nlp.evaluate_retrieval(BENCHMARK_QUERIES, k=2)
    results_k3 = nlp.evaluate_retrieval(BENCHMARK_QUERIES, k=3)

    print("\n=======================================================")
    print("      NOTEWISE INFORMATION RETRIEVAL BENCHMARK RESULTS")
    print("=======================================================")
    print(f"Total Benchmark Queries:   {len(BENCHMARK_QUERIES)}")
    print(f"Indexed Chunk Corpus:      {len(chunks)} chunks")
    print(f"Average Retrieval Latency: {elapsed_ms:.2f} ms / query")
    print("\n[Metrics @ K=2]:")
    print(f"  • Hit Rate @ 2:   {results_k2['hit_rate_at_k'] * 100:.1f}%")
    print(f"  • Precision @ 2:  {results_k2['precision_at_k'] * 100:.1f}%")
    print(f"  • Recall @ 2:     {results_k2['recall_at_k'] * 100:.1f}%")
    print(f"  • MRR:            {results_k2['mrr']:.3f}")

    print("\n[Metrics @ K=3]:")
    print(f"  • Hit Rate @ 3:   {results_k3['hit_rate_at_k'] * 100:.1f}%")
    print(f"  • Precision @ 3:  {results_k3['precision_at_k'] * 100:.1f}%")
    print(f"  • Recall @ 3:     {results_k3['recall_at_k'] * 100:.1f}%")
    print(f"  • MRR:            {results_k3['mrr']:.3f}")
    print("=======================================================\n")

    return {
        "latency_ms": round(elapsed_ms, 2),
        "k2": results_k2,
        "k3": results_k3
    }


if __name__ == "__main__":
    run_evaluation()
