"""
Text Processing Utilities for NoteWise
Handles text cleaning, sentence segmentation, semantic chunking, and statistics.
"""

import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """
    Cleans raw extracted text from PDFs:
    - Normalizes unicode whitespaces
    - Fixes hyphenated words broken across line breaks (e.g. 'infor- \nmation' -> 'information')
    - Removes non-printable characters while preserving punctuation and structure
    - Consolidates redundant empty lines
    """
    if not text:
        return ""

    # Replace windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphenated words broken at line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # Normalize multiple whitespaces (except newlines)
    text = re.sub(r'[ \t]+', ' ', text)

    # Replace 3 or more consecutive newlines with 2 newlines (paragraph boundary)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespaces per line
    lines = [line.strip() for line in text.split('\n')]
    cleaned = '\n'.join(lines).strip()

    return cleaned


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using regex boundary rules,
    handling common abbreviations (e.g., e.g., i.e., Dr., Prof., Fig., vs.).
    """
    if not text:
        return []

    # Replace common abbreviation dots with a placeholder
    abbreviations = [
        (r'\b(e\.g\.)', 'eg_placeholder'),
        (r'\b(i\.e\.)', 'ie_placeholder'),
        (r'\b(Dr\.)', 'Dr_placeholder'),
        (r'\b(Prof\.)', 'Prof_placeholder'),
        (r'\b(Fig\.)', 'Fig_placeholder'),
        (r'\b(vs\.)', 'vs_placeholder'),
        (r'\b(etc\.)', 'etc_placeholder'),
        (r'\b(al\.)', 'al_placeholder'),
    ]

    protected = text
    for pattern, placeholder in abbreviations:
        protected = re.sub(pattern, placeholder, protected, flags=re.IGNORECASE)

    # Sentence boundary: punctuation followed by whitespace and capital letter or end of string
    sentence_end = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
    raw_sentences = sentence_end.split(protected)

    sentences = []
    for s in raw_sentences:
        # Restore placeholders
        s = s.replace('eg_placeholder', 'e.g.')
        s = s.replace('ie_placeholder', 'i.e.')
        s = s.replace('Dr_placeholder', 'Dr.')
        s = s.replace('Prof_placeholder', 'Prof.')
        s = s.replace('Fig_placeholder', 'Fig.')
        s = s.replace('vs_placeholder', 'vs.')
        s = s.replace('etc_placeholder', 'etc.')
        s = s.replace('al_placeholder', 'al.')
        s = s.strip()
        if len(s) > 5:
            sentences.append(s)

    return sentences


def create_chunks(
    pages_data: List[Dict[str, Any]], 
    chunk_size_words: int = 150, 
    chunk_overlap_words: int = 30
) -> List[Dict[str, Any]]:
    """
    Creates overlapping text chunks from page-level extracted data.
    Each chunk retains page numbers and paragraph context for explainable retrieval.
    
    Args:
        pages_data: List of dicts with 'page_number' and 'text'
        chunk_size_words: Target word count per chunk
        chunk_overlap_words: Number of overlapping words between consecutive chunks
        
    Returns:
        List of chunk dicts: {
            'chunk_id': int,
            'text': str,
            'page_number': int,
            'word_count': int
        }
    """
    chunks = []
    chunk_id = 1

    for page in pages_data:
        page_num = page.get("page_number", 1)
        page_text = clean_text(page.get("text", ""))
        if not page_text:
            continue

        words = page_text.split()
        if not words:
            continue

        # If page is short, keep as single chunk
        if len(words) <= chunk_size_words:
            chunks.append({
                "chunk_id": chunk_id,
                "text": page_text,
                "page_number": page_num,
                "word_count": len(words)
            })
            chunk_id += 1
            continue

        # Sliding window chunking
        step = max(1, chunk_size_words - chunk_overlap_words)
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size_words]
            chunk_str = " ".join(chunk_words)

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_str,
                "page_number": page_num,
                "word_count": len(chunk_words)
            })
            chunk_id += 1

            if i + chunk_size_words >= len(words):
                break

    return chunks


def compute_document_stats(text: str, num_pages: int = 1) -> Dict[str, Any]:
    """
    Computes summary statistics for the processed document.
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentences = split_into_sentences(text)
    sentence_count = len(sentences)

    # Average reading speed ~ 200 words per minute
    est_reading_time_min = max(1, round(word_count / 200, 1))

    return {
        "num_pages": num_pages,
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": round(word_count / max(1, sentence_count), 1),
        "est_reading_time_min": est_reading_time_min
    }
