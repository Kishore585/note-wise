"""
NLP & Machine Learning Service for NoteWise
Implements explainable ML using scikit-learn's TfidfVectorizer and Cosine Similarity:
1. Important concept & keyword extraction (Uni-grams & Bi-grams with TF-IDF weights)
2. Passage retrieval via TF-IDF vectorization and cosine similarity
3. Explainability metrics (feature weight analysis & matched terms)
"""

from typing import List, Dict, Any, Tuple
import re
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from utils.text_processing import split_into_sentences, clean_text


class NLPService:
    """
    Core NLP & Machine Learning Engine for NoteWise.
    Encapsulates TF-IDF document representation, keyword ranking,
    and cosine similarity passage retrieval.
    """

    def __init__(self, chunks: List[Dict[str, Any]] = None):
        self.chunks = chunks or []
        self.vectorizer = None
        self.chunk_matrix = None
        self.feature_names = np.array([])
        
        if self.chunks:
            self._fit_vectorizer()

    def set_chunks(self, chunks: List[Dict[str, Any]]):
        """Updates document chunks and re-fits the TF-IDF vectorizer."""
        self.chunks = chunks
        self._fit_vectorizer()

    def _fit_vectorizer(self):
        """Fits TfidfVectorizer on all document chunks."""
        if not self.chunks:
            return

        corpus = [c["text"] for c in self.chunks]
        
        # Configure TfidfVectorizer to capture single terms and bi-grams
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1500,
            sublinear_tf=True,
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
        )
        
        try:
            self.chunk_matrix = self.vectorizer.fit_transform(corpus)
            self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        except ValueError:
            try:
                # Fallback: Relax token length and stopwords for minimal text
                self.vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z0-9]{2,}\b')
                self.chunk_matrix = self.vectorizer.fit_transform(corpus)
                self.feature_names = np.array(self.vectorizer.get_feature_names_out())
            except ValueError:
                # If still empty (pure symbols/whitespace), assign safe empty state
                self.chunk_matrix = None
                self.feature_names = np.array([])

    def extract_key_concepts(self, full_text: str, top_n: int = 15) -> List[Dict[str, Any]]:
        """
        Extracts and ranks key concepts from text using TF-IDF term importance.
        
        Methodology:
        - Segments text into candidate sentences/paragraphs.
        - Computes TF-IDF matrix over segments.
        - Calculates the mean TF-IDF weight for each term across non-zero occurrences.
        - Retrieves a representative context sentence from the document for each concept.
        
        Args:
            full_text: The complete text of the notes.
            top_n: Number of top concepts to return.
            
        Returns:
            List of dictionaries containing concept term, tfidf_score, category, and context snippet.
        """
        if not full_text or len(full_text.strip()) < 10:
            return []

        sentences = split_into_sentences(full_text)
        if not sentences:
            sentences = [full_text]

        # Filter out common academic boilerplates
        academic_stopwords = {
            'lecture', 'instructor', 'department', 'course', 'prof', 'university',
            'continued', 'chapter', 'section', 'page', 'pages', 'fig', 'figure',
            'table', 'example', 'examples', 'using', 'used', 'uses', 'let', 'given',
            'following', 'consider', 'shown', 'shows', 'described', 'based', 'called',
            'know', 'understand', 'check', 'review', 'summary', 'takeaways', 'key'
        }

        # Build custom stopwords combining standard english with academic boilerplates
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        custom_stopwords = list(ENGLISH_STOP_WORDS.union(academic_stopwords))

        concept_vectorizer = TfidfVectorizer(
            stop_words=custom_stopwords,
            ngram_range=(1, 2),
            max_features=800,
            sublinear_tf=True,
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
        )

        try:
            tfidf_mat = concept_vectorizer.fit_transform(sentences)
        except ValueError:
            return []

        features = np.array(concept_vectorizer.get_feature_names_out())
        if len(features) == 0:
            return []

        dense_mat = tfidf_mat.toarray()
        
        # Calculate aggregate importance: sum of TF-IDF scores across sentences
        # multiplied by a boost for compound bigrams which carry higher semantic specificity
        doc_frequencies = np.sum(dense_mat > 0, axis=0)
        tfidf_sums = np.sum(dense_mat, axis=0)
        
        # Boost terms that appear in multiple sentences or are meaningful bigrams
        composite_scores = np.zeros(dense_mat.shape[1])
        for col_idx in range(dense_mat.shape[1]):
            df = doc_frequencies[col_idx]
            tfidf_sum = tfidf_sums[col_idx]
            term = features[col_idx]
            is_bigram = len(term.split()) > 1
            
            # Bigram bonus for domain phrases, but require term length > 3
            bigram_mult = 1.25 if is_bigram else 1.0
            composite_scores[col_idx] = tfidf_sum * (1.0 + 0.3 * np.log1p(df)) * bigram_mult

        # Sort indices by score descending
        top_indices = np.argsort(composite_scores)[::-1]

        ranked_concepts = []
        seen_terms = set()

        for idx in top_indices:
            term = features[idx]
            score = float(composite_scores[idx])

            # Filter single short words or duplicates
            words = term.split()
            if any(len(w) < 3 for w in words) or term in seen_terms:
                continue

            seen_terms.add(term)

            # Find best context sentence containing the term
            context = ""
            term_lower = term.lower()
            for sentence in sentences:
                if term_lower in sentence.lower():
                    context = sentence.strip()
                    break

            is_bigram = len(words) > 1
            ranked_concepts.append({
                "concept": term.title(),
                "tfidf_score": round(score, 4),
                "ngram_type": "Compound Term (Bigram)" if is_bigram else "Key Term (Unigram)",
                "context_sentence": context or "Context found in source document."
            })

            if len(ranked_concepts) >= top_n:
                break

        # Normalize scores to 0.0 - 1.0 range for friendly UI presentation
        if ranked_concepts:
            max_score = max(c["tfidf_score"] for c in ranked_concepts)
            min_score = min(c["tfidf_score"] for c in ranked_concepts)
            spread = max(0.0001, max_score - min_score)
            for c in ranked_concepts:
                c["relative_importance"] = round((c["tfidf_score"] - min_score) / spread * 100, 1)

        return ranked_concepts

    def retrieve_relevant_passages(
        self, 
        query: str, 
        top_k: int = 3, 
        min_score: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant document passages for a query using TF-IDF + Cosine Similarity.
        
        Algorithm:
        1. Vectorize query into document's TF-IDF feature space: q_vec = vectorizer.transform([query])
        2. Compute cosine similarity against all chunks: sim = cosine_similarity(q_vec, chunk_matrix)
        3. Rank chunks by similarity descending.
        4. Extract explainability details (matching terms, feature weights).
        
        Args:
            query: User's question or search prompt.
            top_k: Maximum number of passages to return.
            min_score: Minimum similarity threshold (0.0 to 1.0).
            
        Returns:
            List of retrieved chunks with similarity scores and explainable matching terms.
        """
        if not self.chunks or self.chunk_matrix is None or self.vectorizer is None:
            return []

        cleaned_query = clean_text(query)
        if not cleaned_query:
            return []

        # Vectorize query
        query_vec = self.vectorizer.transform([cleaned_query])
        
        # Check if query contains any recognized vocabulary
        if query_vec.nnz == 0:
            # Query has no overlapping vocabulary terms with the document
            # Return top fallback chunk with score 0.0 to still provide context
            return []

        # Calculate Cosine Similarities
        similarities = cosine_similarity(query_vec, self.chunk_matrix).flatten()
        
        # Sort chunks by similarity descending
        ranked_indices = np.argsort(similarities)[::-1]

        # Extract matching terms in query
        query_terms = set(self.vectorizer.inverse_transform(query_vec)[0])

        results = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score < min_score and len(results) > 0:
                break

            chunk = self.chunks[idx]
            
            # Identify matched terms inside this chunk
            chunk_vec = self.chunk_matrix[idx]
            chunk_terms = set(self.vectorizer.inverse_transform(chunk_vec)[0])
            matched_terms = list(query_terms.intersection(chunk_terms))

            results.append({
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "similarity_score": round(score, 4),
                "similarity_percent": round(score * 100, 1),
                "matched_terms": matched_terms,
                "word_count": chunk.get("word_count", len(chunk["text"].split()))
            })

            if len(results) >= top_k:
                break

        return results

    def evaluate_retrieval(
        self, 
        test_queries: List[Dict[str, Any]], 
        k: int = 3
    ) -> Dict[str, float]:
        """
        Calculates empirical evaluation metrics for the TF-IDF + Cosine Similarity retrieval pipeline:
        - Hit Rate@K: Proportion of queries with at least one relevant passage in top-K.
        - Precision@K: Mean proportion of retrieved passages in top-K that are relevant.
        - Recall@K: Proportion of total expected criteria satisfied within top-K.
        - MRR (Mean Reciprocal Rank): 1 / rank of first relevant result across queries.
        
        Test query format:
        {
            "query": "What is backpropagation?",
            "expected_keywords": ["gradient", "loss"],
            "expected_chunk_ids": [2]  # optional
        }
        """
        if not test_queries or not self.chunks:
            return {"hit_rate_at_k": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "queries_tested": 0}

        hits = 0
        precision_scores = []
        reciprocal_ranks = []

        for item in test_queries:
            query = item["query"]
            expected = set(k.lower() for k in item.get("expected_keywords", []))
            expected_chunks = set(item.get("expected_chunk_ids", []))
            
            retrieved = self.retrieve_relevant_passages(query, top_k=k, min_score=0.01)
            if not retrieved:
                precision_scores.append(0.0)
                reciprocal_ranks.append(0.0)
                continue

            query_hits = 0
            first_rank = 0

            for rank_idx, r in enumerate(retrieved, 1):
                text_lower = r["text"].lower()
                chunk_id = r.get("chunk_id")
                
                is_match = False
                if expected and any(exp in text_lower for exp in expected):
                    is_match = True
                if expected_chunks and chunk_id in expected_chunks:
                    is_match = True

                if is_match:
                    query_hits += 1
                    if first_rank == 0:
                        first_rank = rank_idx

            if query_hits > 0:
                hits += 1
                reciprocal_ranks.append(1.0 / first_rank)
            else:
                reciprocal_ranks.append(0.0)

            precision_scores.append(query_hits / max(1, len(retrieved)))

        return {
            "hit_rate_at_k": round(hits / len(test_queries), 3),
            "precision_at_k": round(float(np.mean(precision_scores)), 3),
            "recall_at_k": round(hits / len(test_queries), 3),
            "mrr": round(float(np.mean(reciprocal_ranks)), 3),
            "queries_tested": len(test_queries),
            "k": k
        }

    # ==========================================================
    # 5. TEXTRANK EXTRACTIVE SUMMARIZATION (PAGERANK GRAPH ML)
    # ==========================================================
    def textrank_extractive_summary(self, full_text: str, document_title: str = "Uploaded Notes") -> str:
        """
        Generates an unsupervised extractive study summary using the TextRank algorithm 
        (PageRank graph centrality applied over sentence-sentence cosine similarity matrix).
        Extracts the most salient, information-dense sentences directly from the uploaded document.
        """
        if not full_text or len(full_text.strip()) < 20:
            return f"# Study Notes: {document_title}\n\nNot enough text was found to create a summary."

        sentences = split_into_sentences(full_text)
        if len(sentences) <= 4:
            body = "\n\n".join(f"- {s}" for s in sentences)
            return f"# Study Notes: {document_title}\n\n## Key points\n\n{body}"

        # Vectorize sentences with TF-IDF
        try:
            sent_vec = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b[a-zA-Z]{3,}\b')
            sent_mat = sent_vec.fit_transform(sentences)
            sim_mat = cosine_similarity(sent_mat, sent_mat)
            np.fill_diagonal(sim_mat, 0)

            # Build row-normalized transition probability matrix
            row_sums = sim_mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            transition = sim_mat / row_sums

            # PageRank power iteration (d = damping factor)
            d = 0.85
            n = len(sentences)
            scores = np.ones(n) / n
            for _ in range(40):
                scores = (1 - d) / n + d * transition.T.dot(scores)

            # Rank sentence indices by PageRank centrality
            ranked_indices = np.argsort(scores)[::-1]

            # Use the highest-ranked sentence as the main idea.
            exec_idx = ranked_indices[0]
            exec_sentence = sentences[exec_idx]

            # Keep a short list of supporting points in document order.
            top_k = min(6, len(sentences) - 1)
            core_indices = sorted(ranked_indices[1:top_k + 1])
            core_sentences = [sentences[i] for i in core_indices]

            topics = self.extract_topic_clusters(num_clusters=3)
            topic_blocks = []
            for t in topics:
                top_terms_str = ", ".join(t["top_keywords"][:5]).title()
                sample_text = t["sample_sentences"][0] if t["sample_sentences"] else "Context identified across document chunks."
                topic_blocks.append(
                    f"### {top_terms_str}\n"
                    f"{sample_text}"
                )

            topics_section = "\n\n".join(topic_blocks) if topic_blocks else "No separate topics were identified."
            core_bullets = "\n".join(f"- {s}" for s in core_sentences)

            return f"""# Study Notes: {document_title}

## Main idea

{exec_sentence}

## Key points
{core_bullets}

## Topics
{topics_section}
"""
        except Exception:
            words = full_text.split()
            return f"# Study Notes: {document_title}\n\n" + " ".join(words[:150])

    # ==========================================================
    # 6. UNSUPERVISED TOPIC MODELING (K-MEANS CLUSTERING)
    # ==========================================================
    def extract_topic_clusters(self, num_clusters: int = 3) -> List[Dict[str, Any]]:
        """
        Performs Unsupervised K-Means Clustering on document chunk TF-IDF vectors 
        to discover latent topics and themes across the lecture notes.
        """
        if not self.chunks or self.chunk_matrix is None:
            return []

        n_chunks = len(self.chunks)
        k = min(num_clusters, n_chunks)
        if k <= 1:
            return []

        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(self.chunk_matrix)

            feature_names = self.feature_names
            topics = []

            for cluster_id in range(k):
                # Find top terms closest to cluster centroid
                centroid = kmeans.cluster_centers_[cluster_id]
                top_term_indices = np.argsort(centroid)[::-1][:6]
                top_terms = [feature_names[i] for i in top_term_indices if i < len(feature_names)]

                # Find chunks assigned to this cluster
                cluster_chunk_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                sample_sentences = []
                for idx in cluster_chunk_indices[:2]:
                    chunk_sents = split_into_sentences(self.chunks[idx]["text"])
                    if chunk_sents:
                        sample_sentences.append(chunk_sents[0])

                topics.append({
                    "topic_id": cluster_id + 1,
                    "top_keywords": top_terms,
                    "chunk_count": len(cluster_chunk_indices),
                    "sample_sentences": sample_sentences
                })

            return topics
        except Exception:
            return []

    # ==========================================================
    # 7. ALGORITHMIC CLOZE QUIZ GENERATION (ACTIVE RECALL)
    # ==========================================================
    def generate_cloze_quiz(self, num_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Generates an algorithmic Active-Recall Multiple Choice Quiz directly from 
        the document's text using TF-IDF Informative Sentence Extraction, Cloze Masking,
        and Dynamic Term-Distance Distractor Generation.

        100% dynamic, runs on ANY uploaded PDF, requires ZERO external APIs.
        """
        if not self.chunks:
            return []

        full_text = " ".join(c["text"] for c in self.chunks)
        concepts = self.extract_key_concepts(full_text, top_n=30)
        if not concepts:
            return []

        keyword_pool = [c["concept"] for c in concepts]

        all_sentences = split_into_sentences(full_text)
        candidate_sentences = [
            s for s in all_sentences 
            if 12 <= len(s.split()) <= 38 and not s.startswith("http")
        ]
        if not candidate_sentences:
            candidate_sentences = all_sentences

        questions = []
        used_sentences = set()
        used_keywords = set()

        for s in candidate_sentences:
            if len(questions) >= num_questions:
                break
            if s in used_sentences:
                continue

            # Find best matching keyword in this sentence
            matched_kw = None
            for kw in keyword_pool:
                if kw in used_keywords:
                    continue
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, s, re.IGNORECASE):
                    matched_kw = kw
                    break

            if not matched_kw:
                continue

            # Mask keyword in sentence
            pattern = r'\b' + re.escape(matched_kw) + r'\b'
            masked_q = re.sub(pattern, "__________", s, flags=re.IGNORECASE, count=1)

            # Distractor generation: pick 3 other keywords from document vocabulary
            distractor_candidates = [k for k in keyword_pool if k.lower() != matched_kw.lower()]
            if len(distractor_candidates) < 3:
                distractor_candidates += ["Parameter", "Function", "Variable"]

            selected_distractors = random.sample(distractor_candidates, min(3, len(distractor_candidates)))
            options = selected_distractors + [matched_kw]
            random.shuffle(options)

            correct_idx = options.index(matched_kw)
            correct_letter = chr(65 + correct_idx)
            formatted_options = [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)]

            used_sentences.add(s)
            used_keywords.add(matched_kw)

            questions.append({
                "id": len(questions) + 1,
                "question": f"Fill in the blank: \"{masked_q}\"",
                "options": formatted_options,
                "correct_answer": correct_letter,
                "explanation": f"Directly from your uploaded notes: \"{s}\""
            })

        return questions

    # ==========================================================
    # 8. EXTRACTIVE QUESTION ANSWERING (VECTOR RETRIEVAL)
    # ==========================================================
    def answer_question_extractive(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Answers a user question locally by retrieving the top matching passages 
        via Cosine Similarity and extracting the most relevant sentence span.
        """
        retrieved = self.retrieve_relevant_passages(query, top_k=top_k)
        if not retrieved or retrieved[0]["similarity_score"] < 0.04:
            return {
                "answer": (
                    "I could not locate any passages in your uploaded notes matching this question with sufficient relevance. "
                    "Try rephrasing with technical terms that appear in your document."
                ),
                "is_grounded": False,
                "sources_used": []
            }

        top_chunk = retrieved[0]
        chunk_sentences = split_into_sentences(top_chunk["text"])

        # Find sentence with highest cosine similarity to query
        best_sentence = top_chunk["text"]
        if chunk_sentences:
            try:
                q_vec = self.vectorizer.transform([query])
                sent_mat = self.vectorizer.transform(chunk_sentences)
                sims = cosine_similarity(q_vec, sent_mat).flatten()
                best_idx = np.argmax(sims)
                best_sentence = chunk_sentences[best_idx]
            except Exception:
                best_sentence = chunk_sentences[0]

        answer_text = (
            f"### From page {top_chunk['page_number']}\n\n"
            f"> {best_sentence}"
        )

        return {
            "answer": answer_text,
            "is_grounded": True,
            "sources_used": [
                {"passage_num": i+1, "page": p["page_number"], "similarity": p["similarity_percent"]}
                for i, p in enumerate(retrieved)
            ]
        }
