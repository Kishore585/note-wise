"""
NoteWise — AI-Powered Study Notes Analyzer
"""

import os
import html
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from utils.text_processing import create_chunks, compute_document_stats
from services.pdf_service import extract_text_from_pdf
from services.nlp_service import NLPService

# ==========================================================
# PAGE CONFIG & STYLING
# ==========================================================
st.set_page_config(
    page_title="NoteWise — Study Notes Analyzer",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
        padding: 24px 32px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; font-weight: 800; color: #fff !important; }
    .main-header p { margin: 4px 0 0 0; font-size: 0.95rem; opacity: 0.85; color: #e0f2fe !important; }

    .metric-card {
        background: rgba(37, 99, 235, 0.04);
        border: 1px solid rgba(37, 99, 235, 0.12);
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-value { font-size: 1.7rem; font-weight: 800; color: #2563eb; }
    .metric-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.8px; color: #64748b; margin-top: 2px; font-weight: 600; }

    .concept-card {
        border-left: 3px solid #2563eb;
        padding: 12px 16px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 10px;
        background: rgba(37, 99, 235, 0.02);
    }
    .concept-badge {
        display: inline-block; padding: 2px 8px; border-radius: 10px;
        font-size: 0.7rem; font-weight: 700; background: #dbeafe; color: #1d4ed8; margin-right: 6px;
    }
    .badge-bigram { background: #fef3c7; color: #b45309; }

    .retrieval-box {
        border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 12px; margin-top: 8px; background: #f8fafc;
    }
    .topic-card {
        border-left: 3px solid #7c3aed;
        padding: 12px 16px; border-radius: 0 10px 10px 0;
        margin-bottom: 10px; background: rgba(124, 58, 237, 0.02);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# SESSION STATE
# ==========================================================
for key, default in {
    "extracted_data": None, "nlp_service": None, "summary_content": None,
    "key_concepts": None, "quiz_questions": None, "quiz_user_answers": {},
    "quiz_submitted": False, "qa_history": [], "doc_title": "Document",
    "loaded_doc_id": None, "topic_clusters": None, "full_text": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.title("NoteWise")
    st.caption("Study Notes Analyzer")
    st.divider()

    st.subheader("Upload notes")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    sample_pdf_path = "sample_data/machine_learning_notes.pdf"
    load_sample = False
    if os.path.exists(sample_pdf_path):
        if st.button("Load Sample Notes", use_container_width=True):
            load_sample = True

    doc_name_custom = st.text_input("Document Title:", value=st.session_state.doc_title)
    if doc_name_custom:
        st.session_state.doc_title = doc_name_custom

    st.divider()
    st.subheader("Options")
    num_concepts_slider = st.slider("Number of concepts", min_value=5, max_value=25, value=12)
    quiz_count_slider = st.slider("Quiz questions", min_value=3, max_value=10, value=5)
    chunk_size = st.slider("Passage length (words)", min_value=80, max_value=250, value=140)
    num_topics = st.slider("Number of topics", min_value=2, max_value=6, value=3)

    st.divider()
    process_btn = st.button("Analyze notes", type="primary", use_container_width=True)


# ==========================================================
# HEADER
# ==========================================================
st.markdown("""
<div class="main-header">
    <h1>NoteWise</h1>
    <p>Upload lecture notes and get summaries, key concepts, quizzes, and answers — all processed locally.</p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# PROCESSING PIPELINE
# ==========================================================
should_process = process_btn or (load_sample and st.session_state.extracted_data is None)

if should_process:
    with st.status("Processing notes...", expanded=True) as status:
        # Read PDF
        status.update(label="Reading PDF...")
        pdf_bytes = None
        if load_sample and os.path.exists(sample_pdf_path):
            with open(sample_pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.session_state.doc_title = "Machine Learning & Deep Learning Notes"
        elif uploaded_file is not None:
            pdf_bytes = uploaded_file.read()
            if not doc_name_custom or doc_name_custom == "Document":
                st.session_state.doc_title = uploaded_file.name.replace(".pdf", "").replace("_", " ").title()
        else:
            st.warning("Upload a PDF or load the sample notes first.")
            st.stop()

        extraction_result = extract_text_from_pdf(pdf_bytes)
        if extraction_result.is_scanned_or_empty:
            st.error(extraction_result.warning_message)
            st.stop()

        # Chunk text
        status.update(label="Organizing notes...")
        chunks = create_chunks(extraction_result.pages_data, chunk_size_words=chunk_size, chunk_overlap_words=25)

        # Build ML models
        status.update(label="Finding key concepts...")
        nlp = NLPService(chunks)
        raw_concepts = nlp.extract_key_concepts(extraction_result.full_text, top_n=num_concepts_slider)

        enriched_concepts = []
        for c in raw_concepts:
            c_copy = dict(c)
            context = c.get("context_sentence", "").strip()
            if context and len(context) > 10:
                c_copy["definition"] = context[:200]
                c_copy["memory_hook"] = f'Think of "{c["concept"]}" in context: "{context[:80]}..."'
            else:
                c_copy["definition"] = f"Important term (score: {c['tfidf_score']})"
                c_copy["memory_hook"] = f'Focus on how "{c["concept"]}" is used in the document.'
            enriched_concepts.append(c_copy)

        status.update(label="Preparing summary...")
        summary = nlp.textrank_extractive_summary(extraction_result.full_text, document_title=st.session_state.doc_title)

        status.update(label="Grouping topics...")
        topics = nlp.extract_topic_clusters(num_clusters=num_topics)

        status.update(label="Creating quiz...")
        quiz = nlp.generate_cloze_quiz(num_questions=quiz_count_slider)

        # Save to session
        st.session_state.extracted_data = {"result": extraction_result, "chunks": chunks}
        st.session_state.nlp_service = nlp
        st.session_state.summary_content = summary
        st.session_state.key_concepts = enriched_concepts
        st.session_state.quiz_questions = quiz
        st.session_state.quiz_user_answers = {}
        st.session_state.quiz_submitted = False
        st.session_state.qa_history = []
        st.session_state.topic_clusters = topics
        st.session_state.full_text = extraction_result.full_text
        current_id = "sample" if (load_sample and not uploaded_file) else (uploaded_file.name if uploaded_file else "sample")
        st.session_state.loaded_doc_id = current_id

        status.update(label="Done.", state="complete", expanded=False)


# ==========================================================
# DASHBOARD
# ==========================================================
if st.session_state.extracted_data is not None:
    data = st.session_state.extracted_data
    stats = data["result"].stats
    chunks = data["chunks"]

    if uploaded_file is not None and st.session_state.loaded_doc_id is not None and uploaded_file.name != st.session_state.loaded_doc_id:
        st.warning("A new document is selected. Choose Analyze notes to update the results.")

    # Metrics
    cols = st.columns(5)
    for col, val, label in zip(cols,
        [stats['num_pages'], f"{stats['word_count']:,}", len(chunks), f"{stats['est_reading_time_min']}m", len(st.session_state.key_concepts or [])],
        ["Pages", "Words", "Passages", "Read time", "Concepts"]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.write("")

    # Tabs
    tab_summary, tab_concepts, tab_topics, tab_quiz, tab_qa = st.tabs([
        "Summary", "Key Concepts", "Topics", "Quiz", "Ask Notes"
    ])

    # ---- SUMMARY ----
    with tab_summary:
        st.subheader("Summary")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.session_state.summary_content:
                st.download_button("Download summary", data=st.session_state.summary_content,
                    file_name=f"{st.session_state.doc_title.lower().replace(' ', '_')}_summary.md",
                    mime="text/markdown", use_container_width=True)
        with col1:
            if st.session_state.summary_content:
                st.markdown(st.session_state.summary_content)
            else:
                st.info("No summary yet. Process a document first.")

    # ---- KEY CONCEPTS ----
    with tab_concepts:
        st.subheader("Key Concepts")
        st.caption("Key terms and phrases from your notes.")

        concepts = st.session_state.key_concepts or []
        if concepts:
            concept_scores = pd.DataFrame([
                {"Concept": c["concept"], "Importance": c.get("relative_importance", 100.0)}
                for c in concepts
            ])
            st.bar_chart(concept_scores.set_index("Concept")["Importance"])
            for c in concepts:
                badge_class = "badge-bigram" if "Bigram" in c["ngram_type"] else ""
                concept_type = "Phrase" if badge_class else "Term"
                st.markdown(f"""
                <div class="concept-card">
                    <span class="concept-badge {badge_class}">{concept_type}</span>
                    <strong>{html.escape(c['concept'])}</strong>
                    <span style="float:right; font-size:0.8rem; color:#64748b;">Importance: {c.get('relative_importance', 100)}%</span>
                    <p style="margin:6px 0 0 0; font-size:0.88rem; color:#334155;">{html.escape(c.get('definition', ''))}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No concepts extracted yet.")

    # ---- TOPICS ----
    with tab_topics:
        st.subheader("Topics")
        st.caption("Related ideas grouped from your notes.")

        topics = st.session_state.topic_clusters or []
        if topics:
            topic_counts = pd.DataFrame([
                {"Topic": ", ".join(t["top_keywords"][:3]).title(), "Passages": t["chunk_count"]}
                for t in topics
            ])
            st.bar_chart(topic_counts.set_index("Topic")["Passages"])
            for t in topics:
                keywords = ", ".join(t["top_keywords"][:6]).title()
                st.markdown(f"""
                <div class="topic-card">
                    <strong>Topic {t['topic_id']}</strong>
                    <span style="float:right; font-size:0.8rem; color:#7c3aed;">{t['chunk_count']} passages</span>
                    <p style="margin:6px 0 0 0; font-size:0.88rem; color:#334155;">Keywords: {html.escape(keywords)}</p>
                </div>
                """, unsafe_allow_html=True)
                if t["sample_sentences"]:
                    with st.expander(f"Example from Topic {t['topic_id']}"):
                        for s in t["sample_sentences"]:
                            st.markdown(f"> {s}")
        else:
            st.info("No topics generated yet.")

    # ---- QUIZ ----
    with tab_quiz:
        st.subheader("Quiz")
        st.caption("Answer these questions using what you learned from your notes.")

        quiz_items = st.session_state.quiz_questions or []
        if quiz_items:
            with st.form("quiz_form"):
                for idx, q in enumerate(quiz_items, 1):
                    st.markdown(f"**Q{idx}. {q['question']}**")
                    current = st.session_state.quiz_user_answers.get(f"q_{idx}")
                    sel = st.radio(f"Answer Q{idx}:", q["options"],
                        index=None if current is None else q["options"].index(current) if current in q["options"] else 0,
                        key=f"radio_q_{idx}", label_visibility="collapsed")
                    st.session_state.quiz_user_answers[f"q_{idx}"] = sel
                    st.write("")
                submitted = st.form_submit_button("Check answers", type="primary")

            if submitted:
                st.session_state.quiz_submitted = True

            if st.session_state.quiz_submitted:
                st.divider()
                correct_count = 0
                for idx, q in enumerate(quiz_items, 1):
                    user_ans = st.session_state.quiz_user_answers.get(f"q_{idx}")
                    correct_raw = str(q.get("correct_answer", "A")).strip()
                    c_m = re.search(r'\b([A-D])\b', correct_raw.upper())
                    correct_letter = c_m.group(1) if c_m else correct_raw[:1].upper()
                    user_letter = ""
                    if user_ans:
                        u_m = re.search(r'^\s*([A-D])\b', str(user_ans).upper())
                        user_letter = u_m.group(1) if u_m else str(user_ans).strip()[:1].upper()
                    is_correct = user_letter == correct_letter
                    if is_correct:
                        correct_count += 1
                        st.success(f"Q{idx}: Correct ({user_ans})")
                    else:
                        st.error(f"Q{idx}: Incorrect (You: {user_ans or 'N/A'}, Answer: {correct_letter})")
                    with st.expander(f"Explanation for Q{idx}"):
                        st.markdown(q.get("explanation", "Refer to the source document."))

                score_pct = round((correct_count / len(quiz_items)) * 100, 1)
                st.markdown(f"### Score: {correct_count}/{len(quiz_items)} ({score_pct}%)")
                if st.button("Retake"):
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_user_answers = {}
                    st.rerun()
        else:
            st.info("No quiz generated. Process a document first.")

    # ---- ASK NOTES ----
    with tab_qa:
        st.subheader("Ask Your Notes")
        st.caption("Ask a question to find relevant passages in your notes.")

        query = st.text_input("Your question:", placeholder="e.g. How does backpropagation work?", key="qa_input")
        if st.button("Search", type="primary") and query.strip():
            nlp = st.session_state.nlp_service
            if nlp is None:
                st.error("Process a document first.")
            else:
                with st.spinner("Searching..."):
                    qa_result = nlp.answer_question_extractive(query, top_k=3)
                    retrieved = nlp.retrieve_relevant_passages(query, top_k=3, min_score=0.03)
                    st.session_state.qa_history.insert(0, {
                        "question": query, "answer": qa_result["answer"], "sources": retrieved
                    })

        if st.session_state.qa_history:
            st.divider()
            for item in st.session_state.qa_history:
                st.markdown(f"**Q: {item['question']}**")
                st.markdown(item["answer"])
                if item.get("sources"):
                    with st.expander(f"Retrieved passages ({len(item['sources'])})"):
                        for i, p in enumerate(item["sources"], 1):
                            st.markdown(f"""
                            <div class="retrieval-box">
                                <strong>Passage {i}</strong>
                                <span style="font-size:0.8rem; color:#64748b; margin-left:8px;">Page {p['page_number']}</span>
                                <p style="margin:6px 0 0 0; font-size:0.88rem; color:#1e293b;"><em>"{html.escape(p['text'])}"</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                st.divider()

else:
    st.info("Upload a PDF or try the sample notes to get started.")
    st.markdown("""
    ### How it works

    1. Upload your lecture notes as a PDF.
    2. Review the summary, key concepts, and related topics.
    3. Test yourself with a quiz or ask a question about your notes.

    Your notes stay on your computer.
    """)
