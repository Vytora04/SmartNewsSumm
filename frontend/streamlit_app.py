# streamlit_app.py
import streamlit as st
import requests
from typing import Dict, Any, Optional
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="SmartNewsSumm - News Summarizer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# SESSION STATE INIT
# =========================
for key in ["article_text", "summary", "qa_report", "metrics", "run_summarize", "results", "start_time", "end_time"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key == "article_text" else None

# =========================
# EXAMPLE ARTICLES
# =========================
EXAMPLE_ARTICLES = {
    "Technology Example": """Artificial Intelligence Transforms Healthcare Industry
    
Medical professionals are increasingly turning to artificial intelligence to improve patient care and diagnostic accuracy. Recent studies show that AI-powered diagnostic tools can identify certain diseases with accuracy rates exceeding 95%, often surpassing human doctors in speed and consistency.

Major hospitals across the United States have begun implementing AI systems that analyze medical imaging, predict patient deterioration, and recommend treatment plans. Dr. Sarah Chen, chief medical officer at Stanford Medical Center, notes that these systems don't replace doctors but augment their capabilities, allowing them to make more informed decisions faster.

The technology has proven particularly effective in radiology, where AI algorithms can detect subtle abnormalities in X-rays, CT scans, and MRIs that human eyes might miss. One system developed by Google Health demonstrated a 11.5% reduction in false negatives and a 5.7% reduction in false positives compared to human radiologists.

However, challenges remain. Privacy concerns, the need for massive datasets, and questions about liability when AI makes mistakes continue to spark debate among medical professionals and ethicists. The FDA has approved over 100 AI-based medical devices, but regulatory frameworks are still evolving to keep pace with rapid technological advancement.""",
    
    "Business Example": """Global Tech Giant Announces Record Quarterly Earnings

Tech industry leader announced record-breaking quarterly earnings today, surpassing analyst expectations and sending stock prices soaring. The company reported revenue of $150 billion for the quarter, representing a 25% increase year-over-year, driven primarily by strong cloud computing and artificial intelligence service adoption.

CEO Jennifer Martinez attributed the success to strategic investments in emerging technologies and a shift toward subscription-based services. "Our focus on innovation and customer-centric solutions has positioned us perfectly for the digital transformation era," Martinez said during the earnings call.

The cloud computing division alone generated $60 billion in revenue, up 35% from the previous year. Enterprise clients increasingly migrating to cloud infrastructure drove this growth, with the company securing several major contracts with Fortune 500 companies.

Analysts praised the results but cautioned about potential headwinds, including increased competition, regulatory scrutiny, and global economic uncertainties. The company announced plans to invest an additional $20 billion in research and development over the next fiscal year, focusing on quantum computing and advanced AI systems.""",
    
    "Sports Example": """Underdog Team Clinches Championship in Thrilling Finale

In a stunning upset that will be remembered for years, the underdog Phoenix Flames secured their first championship title in franchise history, defeating the heavily favored Metropolitan Titans 98-95 in a nail-biting finale. The victory caps an incredible playoff run for a team that barely secured a playoff spot in the final week of the regular season.

Star player Marcus Johnson delivered a masterclass performance, scoring 42 points including the game-winning three-pointer with just 3.2 seconds remaining. Johnson, who joined the team mid-season, has proven to be the catalyst the Flames needed, averaging 31 points throughout the playoffs.

"This is what dreams are made of," an emotional Johnson said during the post-game press conference. "Nobody gave us a chance, but we believed in ourselves and in each other. This championship belongs to our fans who never stopped supporting us."

The Titans, who won three championships in the past five years, struggled to contain the Flames' dynamic offense in the fourth quarter. Despite leading by 12 points at halftime, they couldn't maintain their momentum as the Flames mounted an impressive comeback.

Head coach Sarah Williams, in her second year with the franchise, becomes the youngest coach to win a championship at age 38. The victory parade is scheduled for next week, expected to draw hundreds of thousands of celebrating fans."""
}

# =========================
# SIDEBAR — CONTROLS
# =========================
st.sidebar.title("Controls")

api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000/summarize")
st.sidebar.markdown("---")

st.sidebar.subheader("Model")
model_name = st.sidebar.selectbox(
    "Summarization Model",
    [
        "sshleifer/distilbart-cnn-12-6",
        "facebook/bart-large-cnn",
        "google/flan-t5-base",
        "google/pegasus-cnn_dailymail",
    ],
)

st.sidebar.subheader("Model Variant")
use_lora = st.sidebar.checkbox(
    "🔥 Enable LoRA Adapter (Fine-tuned)",
    value=True,
    help="Use the custom fine-tuned weights (results/lora_kfold/fold_0) instead of the base model."
)

st.sidebar.subheader("Method")
method = st.sidebar.radio("Method", ["Abstractive", "Extractive (Lead)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Quality Enhancement")

# Simplified RAG control
use_rag = st.sidebar.checkbox(
    "📚 Use RAG (Retrieval Augmented Generation)",
    value=False,
    help="Retrieves relevant sentences from the article to ground the summary."
)

hybrid_mode = st.sidebar.checkbox(
    "🚀 Hybrid Mode (Extract-then-Abstract)",
    value=True,
    help="Filters article to most important sentences before summarization."
)

if hybrid_mode:
    hybrid_ratio = st.sidebar.slider(
        "Extractive filter strength",
        min_value=0.2,
        max_value=0.5,
        value=0.3,
        step=0.05,
        help="Keep top X% of sentences"
    )
else:
    hybrid_ratio = 0.3

st.sidebar.markdown("---")
st.sidebar.subheader("Baselines")
compare_variants = st.sidebar.checkbox(
    "Compare Variants (Base vs LoRA vs RAG)",
    value=False,
    help="Runs multiple summaries side-by-side.",
)

# -------------------------
# Generation Settings
# -------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Generation")
num_beams = st.sidebar.slider("Beam Search Size", 1, 8, 4, help="Higher = better quality but slower.")
no_repeat_ngram_size = st.sidebar.slider("No Repeat N-Gram Size", 0, 5, 3, help="Prevents repetition of phrases. 2 is strict, 3 is standard. 0 = off.")

# Defaults for hidden advanced settings (to satisfy API payload logic)
behavior_mode = "Manual"
rag_top_k = 3
rag_query_on = "article"
condensation_strategy = "iterative"
chunking_mode = "token"

# ... (rest of sidebar) ...

# =========================
# MAIN HEADER
# =========================
st.title("🎯 SmartNewsSumm: Automatic News Summarization")
st.caption("Transformer-based abstractive summarization with RAG and hybrid extraction")
st.divider()

mode_col, length_col, _ = st.columns([1, 2, 7])

with mode_col:
    summary_mode = st.selectbox("Render as", ["Paragraph", "Bullet Points"])

with length_col:
    length_level = st.slider(
        "Summary Length",
        min_value=0,
        max_value=2,
        value=1,
        help="0 = Short, 1 = Medium, 2 = Long",
    )

def length_to_bounds(level: int):
    if level == 0:
        return 20, 60
    if level == 1:
        return 50, 120
    return 130, 300

def count_words(text: str) -> int:
    import re
    if not text: return 0
    return len(re.findall(r"[\w']+", text))

def clean_summary_text(summary_text: str) -> str:
    import re
    # Fix common spacing issues (e.g., "end.The" -> "end. The")
    summary_text = re.sub(r'\.([A-Z])', r'. \1', summary_text)
    
    # Fix "TheCloud" -> "The Cloud" (tokenizer artifact where space is dropped)
    summary_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', summary_text)

    # Fix "year."Our" -> "year." Our" (closing quote spacing)
    summary_text = re.sub(r'([.!?;])\"([A-Z])', r'\1" \2', summary_text)

    # **Cleanup**: Strict "Complete Thought" rule.
    if summary_text and summary_text[-1] not in '.!?"':
        last_punc = max(summary_text.rfind('.'), summary_text.rfind('!'), summary_text.rfind('?'))
        if last_punc != -1:
             summary_text = summary_text[:last_punc+1]
        else:
             summary_text += "."
    return summary_text

def render_summary_card(summary_text: str, mode: str):
    if not summary_text:
        st.info("No summary returned.")
        return

    # Text is already cleaned by clean_summary_text() before reaching here
    
    if mode == "Bullet Points":
        sentences = [s.strip() for s in summary_text.split(".") if s.strip()]
        rendered = "<ul>" + "".join(f"<li>{s}.</li>" for s in sentences) + "</ul>"
    else:
        rendered = summary_text.replace("\n", "<br><br>")

    st.markdown(
        f"""
        <div style="
            background-color: #0e1117;
            padding: 1.1rem;
            border-radius: 10px;
            line-height: 1.6;
            font-size: 16px;
            color: #e6e6e6;
            border: 1px solid rgba(255,255,255,0.08);
        ">
            {rendered}
        </div>
        """,
        unsafe_allow_html=True,
    )

    wc = count_words(summary_text)
    st.caption(f"{wc} words")

def qa_overview(qa_report: Optional[dict]):
    if not qa_report:
        return None
    sent_checks = qa_report.get("sentence_checks", []) or []
    flagged_sentences = sum(1 for s in sent_checks if s.get("flagged"))
    total_sentences = len(sent_checks)
    flagged_entities = qa_report.get("overall_flagged_entities", 0) or 0
    return flagged_sentences, total_sentences, flagged_entities

# =========================
# API CALL
# =========================
if st.session_state.run_summarize:
    min_length, max_length = length_to_bounds(length_level)

    # Base payload
    base_payload: Dict[str, Any] = {
        "text": st.session_state.article_text,
        "model_name": model_name,
        "extractive_method": "lead" if method.startswith("Extractive") else None,
        "hybrid_mode": hybrid_mode,
        "hybrid_extractive_ratio": hybrid_ratio,
        "min_length": min_length,
        "max_length": max_length,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "run_qa": True,
        # Pass RAG params directly here for single run
        "use_rag": use_rag,
        "rag_top_k": 3,
        "rag_query_on": "article",
    }
    
    # We need to handle LoRA. The backend app currently loads LoRA via env var.
    # To toggle it dynamically, we might need a backend change OR distinct model names.
    # checking backend/app.py... it doesn't support 'use_lora' param yet.
    # Plan: We will need to update backend/app.py to handle this.
    # For now, we pass it in payload, expecting backend update.
    base_payload["use_lora"] = use_lora

    try:
        st.session_state.start_time = time.time()  # Start timing
        with st.spinner("Summarizing..."):
            if compare_variants:
                # Use BART for comparisons
                compare_base = {
                    **base_payload,
                    "model_name": baseline_model,
                    "extractive_method": None,
                }

                variants = {
                    "BART": {
                        "use_rag": False,
                        "use_reranker": False,
                        "condensation_strategy": "onepass",
                        "chunking_mode": "token",
                    },
                    "BART + RAG": {
                        "use_rag": True,
                        "use_reranker": False,
                        "rag_top_k": 3,
                        "rag_query_on": "article",
                        "condensation_strategy": "iterative",
                        "chunking_mode": "token",
                    },
                    "BART + RAG + reranker": {
                        "use_rag": True,
                        "use_reranker": True,
                        "top_k": 5,
                        "rag_top_k": 3,
                        "rag_query_on": "article",
                        "condensation_strategy": "iterative",
                        "chunking_mode": "token",
                    },
                }

                results: Dict[str, Dict[str, Any]] = {}
                for label, vparams in variants.items():
                    payload = {**compare_base, **vparams}
                    r = requests.post(api_url, json=payload, timeout=300)
                    r.raise_for_status()
                    results[label] = r.json()

                st.session_state.results = results
                st.session_state.summary = None
                st.session_state.qa_report = None
                st.session_state.metrics = None

            else:
                payload = dict(base_payload)
                if behavior_mode.startswith("Auto"):
                    payload.update(auto_params)
                else:
                    payload.update(
                        {
                            "use_rag": use_rag,
                            "rag_top_k": rag_top_k,
                            "rag_query_on": rag_query_on,
                            "condensation_strategy": condensation_strategy,
                            "chunking_mode": chunking_mode,
                        }
                    )

                r = requests.post(api_url, json=payload, timeout=300)
                r.raise_for_status()
                resp = r.json()

                raw_summary = resp.get("summary", "")
                st.session_state.summary = clean_summary_text(raw_summary)
                st.session_state.qa_report = resp.get("qa_report")
                st.session_state.metrics = resp.get("metrics")
                st.session_state.results = None

        st.session_state.end_time = time.time()  # End timing
        st.success("✅ Summarization finished!")
    except Exception as e:
        st.session_state.end_time = time.time()
        st.error(f"❌ API error: {e}")

    st.session_state.run_summarize = False

# =========================
# INPUT / OUTPUT LAYOUT
# =========================
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("📝 Input Article")

    # Example articles selector
    example_choice = st.selectbox(
        "Quick Demo - Load Example:",
        ["None"] + list(EXAMPLE_ARTICLES.keys()),
        help="Select an example article for quick testing"
    )
    
    if example_choice != "None" and example_choice in EXAMPLE_ARTICLES:
        if st.button("Load Example Article", use_container_width=True):
            st.session_state.article_text = EXAMPLE_ARTICLES[example_choice]
            st.rerun()

    st.markdown("---")
    
    # ✅ NEW: upload .txt (optional)
    uploaded = st.file_uploader(
        "📎 Upload a .txt file (optional)",
        type=["txt"],
        accept_multiple_files=False,
        help="Loads the file content into the editor below.",
    )
    if uploaded is not None:
        try:
            file_text = uploaded.read().decode("utf-8", errors="ignore").strip()
            if file_text:
                st.session_state.article_text = file_text
                st.success(f"Loaded: {uploaded.name}")
            else:
                st.warning("Uploaded file is empty.")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    # Editor actions
    a1, a2 = st.columns([1, 1])
    with a1:
        if st.button("🗑️ Clear input", use_container_width=True):
            st.session_state.article_text = ""
            st.session_state.summary = None
            st.session_state.qa_report = None
            st.session_state.metrics = None
            st.session_state.results = None
            st.session_state.start_time = None
            st.session_state.end_time = None
            st.rerun()

    st.session_state.article_text = st.text_area(
        "Paste or edit article text:",
        value=st.session_state.article_text,
        height=400,
    )
    wc = len(st.session_state.article_text.split())
    sent_count = st.session_state.article_text.count('.') + st.session_state.article_text.count('!') + st.session_state.article_text.count('?')
    st.caption(f"📊 {wc} words | {sent_count} sentences (approx)")

    if st.button("✨ Summarize", use_container_width=True, type="primary"):
        if not st.session_state.article_text.strip():
            st.warning("⚠️ Please paste an article first (or upload a .txt).")
        else:
            st.session_state.run_summarize = True
            st.rerun()

with right_col:
    st.subheader("📋 Generated Summary")
    
    # Display statistics if summary exists
    if st.session_state.summary or st.session_state.results:
        original_wc = count_words(st.session_state.article_text)
        
        if st.session_state.summary:
            summary_wc = count_words(st.session_state.summary)
            compression_ratio = (1 - summary_wc / original_wc) * 100 if original_wc > 0 else 0
            processing_time = (st.session_state.end_time - st.session_state.start_time) if (st.session_state.start_time and st.session_state.end_time) else 0
            
            # Statistics dashboard
            stat_cols = st.columns(4)
            stat_cols[0].metric("📄 Original", f"{original_wc} words")
            stat_cols[1].metric("📝 Summary", f"{summary_wc} words")
            stat_cols[2].metric("📉 Compression", f"{compression_ratio:.1f}%")
            stat_cols[3].metric("⏱️ Time", f"{processing_time:.2f}s")
            st.markdown("---")

    if st.session_state.results:
        res: Dict[str, Dict[str, Any]] = st.session_state.results
        tabs = st.tabs(list(res.keys()))

        for tab, label in zip(tabs, res.keys()):
            with tab:
                summary_text = res[label].get("summary", "")
                qa_report = res[label].get("qa_report")
                metrics = res[label].get("metrics")

                render_summary_card(summary_text, summary_mode)

                ov = qa_overview(qa_report)
                if ov:
                    flagged_s, total_s, flagged_e = ov
                    st.caption(f"QA flags: {flagged_s}/{total_s} sentences | {flagged_e} entities")

                colA, colB = st.columns([1, 1])
                with colA:
                    with st.expander("QA details"):
                        st.json(qa_report or {})
                with colB:
                    with st.expander("Metrics"):
                        if metrics:
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("ROUGE-1", f"{metrics.get('rouge1', 0):.3f}")
                            c2.metric("ROUGE-2", f"{metrics.get('rouge2', 0):.3f}")
                            c3.metric("ROUGE-L", f"{metrics.get('rougeL', 0):.3f}")
                            c4.metric("BLEU", f"{metrics.get('bleu', 0):.3f}")
                        else:
                            st.info("No metrics returned.")
    else:
        summary_text = st.session_state.summary or ""
        if summary_text:
            render_summary_card(summary_text, summary_mode)
            
            # Add download button
            st.download_button(
                label="💾 Download Summary",
                data=summary_text,
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("ℹ️ No summary yet. Click **✨ Summarize** to generate output.")

# =========================
# QA & EVALUATION (single run)
# =========================
st.divider()
st.subheader("QA & Evaluation (single-run)")

if st.session_state.qa_report:
    ov = qa_overview(st.session_state.qa_report)
    if ov:
        flagged_s, total_s, flagged_e = ov
        st.caption(f"QA flags: {flagged_s}/{total_s} sentences | {flagged_e} entities")
    st.json(st.session_state.qa_report)

if st.session_state.metrics:
    m = st.session_state.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROUGE-1", f"{m.get('rouge1', 0):.3f}")
    c2.metric("ROUGE-2", f"{m.get('rouge2', 0):.3f}")
    c3.metric("ROUGE-L", f"{m.get('rougeL', 0):.3f}")
    c4.metric("BLEU", f"{m.get('bleu', 0):.3f}")
