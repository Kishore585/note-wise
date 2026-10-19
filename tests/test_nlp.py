"""
Unit Tests for NoteWise NLP and Machine Learning Pipeline
Tests:
1. Text cleaning and sentence splitting
2. Document chunking with page retention
3. TF-IDF concept extraction and keyword ranking
4. Cosine similarity passage retrieval and ranking accuracy
"""

import os
import pytest
from utils.text_processing import clean_text, split_into_sentences, create_chunks, compute_document_stats
from services.pdf_service import extract_text_from_pdf
from services.nlp_service import NLPService


SAMPLE_CORPUS = [
    {
        "chunk_id": 1,
        "page_number": 1,
        "text": "Artificial Neural Networks are computational models inspired by biological neural networks. Each neuron computes a weighted sum of inputs and applies a non-linear activation function such as ReLU or Sigmoid."
    },
    {
        "chunk_id": 2,
        "page_number": 2,
        "text": "Backpropagation calculates the gradient of the loss function with respect to weights using the mathematical chain rule of calculus. This allows efficient weight updates."
    },
    {
        "chunk_id": 3,
        "page_number": 2,
        "text": "Gradient descent optimization updates network parameters in the direction of steepest descent. Mini-batch gradient descent and Adam optimizers balance speed and convergence stability."
    },
    {
        "chunk_id": 4,
        "page_number": 3,
        "text": "Overfitting occurs when a neural network memorizes training noise rather than learning generalizable patterns. Regularization techniques like L2 weight decay and Dropout prevent overfitting."
    }
]


def test_clean_text():
    raw_text = "This is a text  with   extra spaces \nand a hyphen- \nated word."
    cleaned = clean_text(raw_text)
    assert "hyphenated" in cleaned
    assert "  " not in cleaned


def test_split_into_sentences():
    text = "Dr. Vance explained the backpropagation algorithm. For e.g. consider a 3-layer net! Does it converge?"
    sentences = split_into_sentences(text)
    assert len(sentences) >= 2
    assert any("backpropagation" in s for s in sentences)


def test_create_chunks():
    pages = [
        {"page_number": 1, "text": "Word " * 200},
        {"page_number": 2, "text": "Page two content is short."}
    ]
    chunks = create_chunks(pages, chunk_size_words=100, chunk_overlap_words=20)
    assert len(chunks) >= 3
    assert chunks[0]["page_number"] == 1
    assert chunks[-1]["page_number"] == 2


def test_document_stats():
    text = "Machine learning models learn patterns from training data. Optimization minimizes empirical loss."
    stats = compute_document_stats(text, num_pages=1)
    assert stats["word_count"] > 10
    assert stats["sentence_count"] == 2
    assert stats["est_reading_time_min"] >= 1


def test_tfidf_concept_extraction():
    full_text = " ".join(c["text"] for c in SAMPLE_CORPUS)
    nlp = NLPService(SAMPLE_CORPUS)
    concepts = nlp.extract_key_concepts(full_text, top_n=10)
    
    assert len(concepts) > 0
    concept_names = [c["concept"].lower() for c in concepts]
    
    # Check that core technical terms were identified
    assert any("neural" in name or "gradient" in name or "backpropagation" in name for name in concept_names)
    assert "tfidf_score" in concepts[0]
    assert concepts[0]["tfidf_score"] > 0


def test_summary_uses_plain_language_without_emoji():
    full_text = " ".join(chunk["text"] for chunk in SAMPLE_CORPUS)
    nlp = NLPService(SAMPLE_CORPUS)

    summary = nlp.textrank_extractive_summary(full_text, document_title="Test Notes")

    assert "## Main idea" in summary
    assert "## Key points" in summary
    assert "## Topics" in summary
    assert not any(term in summary for term in ("TF-IDF", "K-Means", "TextRank"))
    assert not any(symbol in summary for symbol in ("📚", "📌", "🎯", "🔍", "🚀"))


def test_cosine_similarity_retrieval():
    nlp = NLPService(SAMPLE_CORPUS)
    
    # Query 1: Query specifically about backpropagation
    query = "How does backpropagation compute gradients?"
    results = nlp.retrieve_relevant_passages(query, top_k=2)
    
    assert len(results) > 0
    top_result = results[0]
    assert top_result["chunk_id"] == 2  # Chunk 2 contains backpropagation & chain rule
    assert top_result["page_number"] == 2
    assert top_result["similarity_score"] > 0.1
    assert "gradient" in [t.lower() for t in top_result["matched_terms"]] or "backpropagation" in [t.lower() for t in top_result["matched_terms"]]

    # Query 2: Query about overfitting and dropout
    query_2 = "What regularization methods prevent overfitting?"
    results_2 = nlp.retrieve_relevant_passages(query_2, top_k=2)
    assert len(results_2) > 0
    assert results_2[0]["chunk_id"] == 4  # Chunk 4 contains overfitting and dropout


def test_retrieval_evaluation_metrics():
    nlp = NLPService(SAMPLE_CORPUS)
    test_suite = [
        {
            "query": "explain backpropagation and chain rule",
            "expected_keywords": ["backpropagation", "gradient", "chain rule"]
        },
        {
            "query": "how to stop neural network overfitting with dropout",
            "expected_keywords": ["overfitting", "dropout", "regularization"]
        }
    ]
    eval_results = nlp.evaluate_retrieval(test_suite, k=2)
    assert eval_results["hit_rate_at_k"] == 1.0
    assert eval_results["precision_at_k"] > 0.0


def test_pdf_extraction_real_file():
    pdf_path = "sample_data/machine_learning_notes.pdf"
    if os.path.exists(pdf_path):
        result = extract_text_from_pdf(pdf_path)
        assert len(result.pages_data) == 3
        assert "Artificial Neural Networks" in result.full_text
        assert result.stats["word_count"] > 100
        assert not result.is_scanned_or_empty


def test_pdf_extraction_empty_and_corrupted():
    # Empty bytes input
    empty_res = extract_text_from_pdf(b"")
    assert empty_res.is_scanned_or_empty is True
    assert "empty" in empty_res.warning_message.lower()

    # Corrupted stream input
    corrupt_res = extract_text_from_pdf(b"%PDF-corrupted-stream-garbage")
    assert corrupt_res.is_scanned_or_empty is True
    assert len(corrupt_res.warning_message) > 0


def test_nlp_empty_vocab_and_stopwords():
    # Chunks with only common stop words
    stopword_chunks = [{"chunk_id": 1, "page_number": 1, "text": "the of and in for to with at"}]
    nlp = NLPService(stopword_chunks)
    assert nlp.chunk_matrix is not None  # Fallback succeeds safely
    
    # Chunks with pure symbols
    symbol_chunks = [{"chunk_id": 1, "page_number": 1, "text": "... !!! ??? ---"}]
    nlp_sym = NLPService(symbol_chunks)
    # Should not raise exception; safe empty matrix
    retrieved = nlp_sym.retrieve_relevant_passages("query", top_k=2)
    assert retrieved == []


def test_ai_json_extraction_resilience():
    from services.ai_service import AIService
    ai = AIService()

    # Valid JSON in markdown code blocks
    raw_md = """```json
    [
        {"id": 1, "question": "What is SGD?", "options": ["A) Opt 1", "B) Opt 2"], "correct_answer": "A"}
    ]
    ```"""
    parsed = ai._extract_json_array(raw_md)
    assert len(parsed) == 1
    assert parsed[0]["id"] == 1

    # Malformed text without JSON
    assert ai._extract_json_array("Just plain text with no json") == []


def test_quiz_option_regex_matching():
    import re
    # Test various model outputs for correct_answer
    test_cases = [
        ("B", "B"),
        ("Option B", "B"),
        ("B) Second Option", "B"),
        ("  c  ", "C"),
        ("D: Final Choice", "D")
    ]
    for raw, expected in test_cases:
        match = re.search(r'\b([A-D])\b', raw.upper())
        letter = match.group(1) if match else raw.strip()[:1].upper()
        assert letter == expected, f"Failed for {raw}: got {letter}, expected {expected}"
