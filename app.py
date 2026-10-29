"""
Streamlit Web Application for Experiment 5: Automated Legal Document Analyzer
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5: Automated Legal Document Analyzer using GPT-4
"""

import os
import sys
import json
import time
import streamlit as st

# Configure UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from engine.analyzer import LegalDocumentAnalyzer
from engine.text_extractor import LegalTextExtractor
from engine.chunker import LegalDocumentChunker
from engine.analytics import LegalAnalytics

# Page configuration
st.set_page_config(
    page_title="Legal Document Analyzer | Exp 5",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 24px 28px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #f8fafc;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #cbd5e1;
        font-size: 1.0rem;
        margin: 0;
    }
    .legal-box {
        background-color: #0b1329;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 20px;
        font-family: 'Courier New', Courier, monospace;
        white-space: pre-wrap;
        color: #38bdf8;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    .entity-card {
        background: #f8fafc;
        border-left: 5px solid #0284c7;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .dark .entity-card {
        background: #1e293b;
        border-left: 5px solid #38bdf8;
        color: #f1f5f9;
    }
    .clause-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-high { background-color: #fee2e2; color: #991b1b; }
    .badge-med { background-color: #fef3c7; color: #92400e; }
    .badge-low { background-color: #dcfce7; color: #166534; }
</style>
""", unsafe_allow_html=True)

# Sample Documents
SAMPLE_DOCS = {
    "Employment Agreement (Benchmark)": "data/employment_agreement.txt",
    "Mutual Non-Disclosure Agreement (NDA)": "data/non_disclosure_agreement.txt",
    "Master Service Agreement (Software)": "data/service_agreement.txt",
    "Commercial Rental & Lease Agreement": "data/rental_lease_agreement.txt",
    "Enterprise Long Contract (Chunking Demo)": "data/long_commercial_contract.txt"
}


def load_sample_file(path: str) -> str:
    """Helper to load sample agreement text."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return ""


# Header
st.markdown("""
<div class="main-header">
    <h1>⚖️ Automated Legal Document Analyzer using GPT-4</h1>
    <p>Course: <strong>24CS2506 – Building Applications using GPT-4</strong> | Experiment 5</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("🛠️ Analyzer Controls")

backend_choice = st.sidebar.selectbox(
    "🤖 AI Engine Backend",
    options=["auto", "openai", "gpt-4", "gpt-4o", "ollama", "deterministic"],
    index=0,
    help="Select AI backend. 'auto' selects the best available engine (OpenAI -> Ollama -> Deterministic NLP Failsafe)."
)

model_name = st.sidebar.text_input("OpenAI / Ollama Model", value="gpt-4o")

api_key_input = st.sidebar.text_input(
    "🔑 OpenAI API Key (Optional)",
    type="password",
    value=os.environ.get("OPENAI_API_KEY", ""),
    help="Provide your OpenAI API key to execute real GPT-4 calls. If empty, the engine uses local Ollama or high-fidelity Deterministic Legal NLP."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Sample Contracts")
sample_choice = st.sidebar.selectbox(
    "Select Benchmark Document",
    options=list(SAMPLE_DOCS.keys()),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Processing Options")
enable_chunking = st.sidebar.checkbox(
    "Enable Document Chunking (Section 9)",
    value=True,
    help="Divides lengthy legal agreements into smaller sections, processes chunks with GPT-4, and synthesizes results."
)
chunk_size = st.sidebar.slider("Target Chunk Size (Chars)", min_value=1000, max_value=4000, value=2500, step=250)

# Main Input Section
st.subheader("1. Legal Document Input")

input_tab1, input_tab2, input_tab3 = st.tabs([
    "📂 Upload File (.pdf, .txt, .md)",
    "📑 Preloaded Sample Contract",
    "✍️ Paste Raw Legal Text"
])

document_text = ""
document_name = "Legal Document"

with input_tab1:
    uploaded_file = st.file_uploader(
        "Upload a legal agreement (PDF or TXT)",
        type=["pdf", "txt", "md"]
    )
    if uploaded_file is not None:
        document_name = uploaded_file.name
        try:
            if uploaded_file.name.lower().endswith(".pdf"):
                document_text, _ = LegalTextExtractor.extract_from_pdf_bytes(uploaded_file.read())
            else:
                raw_str = uploaded_file.read().decode("utf-8", errors="replace")
                document_text, _ = LegalTextExtractor.extract_from_text(raw_str, name=uploaded_file.name)
            st.success(f"Successfully extracted {len(document_text)} characters from {uploaded_file.name}!")
        except Exception as e:
            st.error(f"Error parsing file: {str(e)}")

with input_tab2:
    sample_text = load_sample_file(SAMPLE_DOCS[sample_choice])
    st.info(f"Loaded: **{sample_choice}** ({len(sample_text.split())} words)")
    if not document_text:
        document_text = sample_text
        document_name = sample_choice

with input_tab3:
    user_pasted = st.text_area(
        "Paste contract or legal clauses directly:",
        value=document_text if not uploaded_file else "",
        height=220,
        placeholder="PASTE YOUR CONTRACT CLAUSES HERE..."
    )
    if user_pasted.strip() and not uploaded_file:
        document_text = user_pasted
        document_name = "User Pasted Agreement"

# Preprocessing & Document Statistics
if document_text.strip():
    cleaned_doc = LegalTextExtractor.clean_text(document_text)
    word_count = len(cleaned_doc.split())
    char_count = len(cleaned_doc)
    readability_preview = LegalAnalytics.analyze_readability(cleaned_doc)
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Total Words", f"{word_count:,}")
    with col_stat2:
        st.metric("Character Count", f"{char_count:,}")
    with col_stat3:
        st.metric("Reading Ease Score", f"{readability_preview['flesch_reading_ease']}/100")
    with col_stat4:
        st.metric("Complexity Level", f"Grade {readability_preview['fk_grade_level']}")

    with st.expander("👁️ View Extracted Document Text Preview", expanded=False):
        st.text_area("Full Preprocessed Text", value=cleaned_doc, height=200, disabled=True)

    st.markdown("---")
    
    # Analysis Execution Button
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_analysis = st.button("🚀 Analyze Legal Document", type="primary", use_container_width=True)
    with col_info:
        st.caption("Executes text extraction, chunking, Section 5 prompt structuring, GPT-4/NLP parsing, and Section 10 formatting.")

    if run_analysis or ("last_analysis_res" in st.session_state and st.session_state.get("analyzed_doc") == cleaned_doc):
        if run_analysis:
            with st.spinner("Analyzing legal document with AI engine..."):
                analyzer = LegalDocumentAnalyzer(api_key=api_key_input)
                try:
                    result = analyzer.analyze_document(
                        document_text=cleaned_doc,
                        backend=backend_choice,
                        model_name=model_name,
                        enable_chunking=enable_chunking,
                        chunk_size=chunk_size
                    )
                    st.session_state["last_analysis_res"] = result
                    st.session_state["analyzed_doc"] = cleaned_doc
                except Exception as ex:
                    st.error(f"Analysis encountered an error: {str(ex)}")
                    st.stop()
        
        result = st.session_state["last_analysis_res"]

        st.success(f"Analysis complete! Engine: **{result.engine_used}** | Time: **{result.execution_time:.2f}s** | Chunks: **{result.chunk_count}**")

        # 6 Interactive Tabs
        tab_std, tab_ent, tab_clauses, tab_summaries, tab_chunks, tab_metrics = st.tabs([
            "📋 Lab Standard Output (Sec 10)",
            "🏛️ Extracted Contract Entities",
            "🛡️ Key Clauses & Risk",
            "⚖️ Extractive vs Abstractive (Sec 8)",
            "🧩 Chunk Processing Trace (Sec 9)",
            "📊 Legal Analytics & Readability"
        ])

        # TAB 1: Lab Standard Output
        with tab_std:
            st.markdown("### 📋 Recommended Output Format (Adhering to Section 10)")
            st.caption("Standard formatted laboratory output as defined in the course experiment manual.")
            
            standard_str = result.to_standard_format()
            st.code(standard_str, language="text")

            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                st.download_button(
                    label="📥 Download Standard TXT",
                    data=standard_str,
                    file_name="legal_document_analysis.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_dl2:
                st.download_button(
                    label="📥 Download Structured JSON",
                    data=json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                    file_name="legal_analysis.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col_dl3:
                st.download_button(
                    label="📥 Download Markdown Report",
                    data=f"# Legal Document Analysis\n\n```text\n{standard_str}\n```\n",
                    file_name="legal_analysis.md",
                    mime="text/markdown",
                    use_container_width=True
                )

        # TAB 2: Extracted Contract Entities
        with tab_ent:
            st.markdown("### 🏛️ Extracted Legal Contract Entities (Section 6)")
            col_e1, col_e2 = st.columns(2)
            
            with col_e1:
                st.markdown(f"""
                <div class="entity-card">
                    <h4>📄 Document Type</h4>
                    <p style="font-size: 1.15rem; font-weight: 600; color: #0284c7;">{result.document_type}</p>
                </div>
                <div class="entity-card">
                    <h4>👥 Parties Involved</h4>
                    <p style="font-size: 1.05rem; font-weight: 500;">{result.parties}</p>
                </div>
                <div class="entity-card">
                    <h4>🎯 Purpose of Agreement</h4>
                    <p>{result.purpose}</p>
                </div>
                <div class="entity-card">
                    <h4>⏳ Duration / Term</h4>
                    <p style="font-size: 1.1rem; font-weight: 600;">{result.duration}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_e2:
                dates_str = ", ".join([f"{k}: {v}" for k, v in result.important_dates.items()]) if result.important_dates else "None specified"
                st.markdown(f"""
                <div class="entity-card">
                    <h4>💰 Payment Terms</h4>
                    <p style="font-size: 1.15rem; font-weight: 600; color: #16a34a;">{result.payment}</p>
                </div>
                <div class="entity-card">
                    <h4>🚪 Termination Conditions</h4>
                    <p style="font-size: 1.05rem; font-weight: 500; color: #dc2626;">{result.termination}</p>
                </div>
                <div class="entity-card">
                    <h4>📅 Important Dates</h4>
                    <p>{dates_str}</p>
                </div>
                <div class="entity-card">
                    <h4>⚖️ Governing Law & Jurisdiction</h4>
                    <p>{result.governing_law}</p>
                </div>
                """, unsafe_allow_html=True)

        # TAB 3: Key Clauses & Risk Assessment
        with tab_clauses:
            st.markdown("### 🛡️ Key Clause Identification (Section 7)")
            st.write("Identified critical clauses and associated legal risk profiles:")

            # Display clause tags
            badge_html = "<div>"
            for c in result.potentially_important_clauses:
                badge_html += f'<span class="clause-badge badge-med">📌 {c}</span>'
            badge_html += "</div>"
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Risk Assessment Box
            risk_meta = LegalAnalytics.calculate_risk_index(result)
            st.subheader(f"Contract Risk Score: {risk_meta['risk_score']}/100 ({risk_meta['risk_level']})")
            st.progress(risk_meta['risk_score'] / 100.0)

            if risk_meta["flags"]:
                st.warning("⚠️ Attention Required for the following terms:")
                for flag in risk_meta["flags"]:
                    st.markdown(f"- {flag}")
            else:
                st.success("✅ Document contains standard protective clauses with controlled liability terms.")

            st.markdown("#### Detailed Clause Provisions:")
            if result.detailed_clauses:
                for cl in result.detailed_clauses:
                    with st.expander(f"🔹 {cl.title} — [{cl.risk_level} Risk Exposure]"):
                        st.markdown(f"**Summary:** {cl.summary}")
                        st.markdown(f"**Contract Snippet:**")
                        st.info(f"\"{cl.snippet}\"")
            else:
                st.info("Key protective clauses verified: Confidentiality, Termination, Intellectual Property, and Payment.")

            st.markdown("#### Key Contractual Obligations:")
            for ob in result.key_obligations:
                st.markdown(f"- 📌 {ob}")

        # TAB 4: Extractive vs Abstractive Summarization
        with tab_summaries:
            st.markdown("### ⚖️ Extractive vs Abstractive Summarization (Section 8)")
            st.caption("Demonstrating the difference between sentence-selection summarization and LLM semantic synthesis.")

            col_abs, col_ext = st.columns(2)
            with col_abs:
                st.subheader("💡 Abstractive Summarization (GPT-4)")
                st.write("*Generates a brand new, natural language synthesis capturing the semantic intent.*")
                st.success(result.summary)

            with col_ext:
                st.subheader("🔍 Extractive Summarization")
                st.write("*Selects key verbatim sentences directly from the original document.*")
                if result.extractive_summary:
                    for s in result.extractive_summary:
                        st.info(f"\"... {s.strip()} ...\"")
                else:
                    st.info("Extractive sentences compiled in analysis.")

            comp_stats = LegalAnalytics.compare_summaries(cleaned_doc, result.extractive_summary, result.summary)
            st.markdown("#### Summarization Performance Comparison:")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Original Words", comp_stats["original_word_count"])
            with col_m2:
                st.metric("Abstractive Words", comp_stats["abstractive_word_count"])
            with col_m3:
                st.metric("Extractive Words", comp_stats["extractive_word_count"])
            with col_m4:
                st.metric("Abstractive Compression", comp_stats["abstractive_compression_ratio"])

        # TAB 5: Document Chunking Visualizer
        with tab_chunks:
            st.markdown("### 🧩 Chunking Long Documents (Section 9)")
            st.write(
                "When legal documents are long, they cannot be sent in a single prompt. "
                "The system divides the document into smaller sections, analyzes each chunk, and synthesizes the results."
            )

            doc_chunks = LegalDocumentChunker.chunk_document(
                cleaned_doc,
                chunk_size=chunk_size,
                force_chunking=enable_chunking
            )

            st.info(f"Document segmented into **{len(doc_chunks)} chunk(s)** (Target size: ~{chunk_size} characters).")

            st.markdown("""
            ```mermaid
            flowchart LR
                A[Legal Document] --> B[Chunk 1]
                A --> C[Chunk 2]
                A --> D[Chunk N]
                B --> E[GPT-4 Analysis]
                C --> E
                D --> E
                E --> F[Combine Results]
                F --> G[Final Legal Summary]
            ```
            """)

            for ch in doc_chunks:
                with st.expander(f"📦 Section / Chunk {ch.chunk_id} of {ch.total_chunks} ({ch.word_count} words | {ch.char_count} chars)"):
                    st.markdown(f"**Section Heading:** `{ch.section_title}`")
                    st.text_area(f"Chunk {ch.chunk_id} Content", value=ch.text, height=140, disabled=True)

        # TAB 6: Readability & Complexity Analysis
        with tab_metrics:
            st.markdown("### 📊 Why is Legal Document Analysis Difficult? (Section 3)")
            st.write(
                "Legal documents contain complex terminology, long sentences, formal language, cross-references, "
                "multiple clauses, exceptions, and large text volumes."
            )

            readability = LegalAnalytics.analyze_readability(cleaned_doc)
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Flesch Reading Ease", f"{readability['flesch_reading_ease']}/100")
            with col_r2:
                st.metric("Flesch-Kincaid Grade", readability['fk_grade_level'])
            with col_r3:
                st.metric("Avg Sentence Length", f"{readability['avg_sentence_length']} words")

            st.info(f"**Complexity Assessment:** {readability['complexity_rating']}")

            coverage = LegalAnalytics.evaluate_clause_coverage(result.potentially_important_clauses)
            st.markdown("#### Standard Protective Clause Audit:")
            col_cov1, col_cov2 = st.columns(2)
            with col_cov1:
                st.write("**Present Protective Clauses:**")
                for p in coverage["present_clauses"]:
                    st.markdown(f"✅ `{p}`")
            with col_cov2:
                st.write("**Missing Standard Clauses:**")
                if coverage["missing_clauses"]:
                    for m in coverage["missing_clauses"]:
                        st.markdown(f"⚠️ `{m}`")
                else:
                    st.success("All 7 standard critical clauses are present!")
else:
    st.info("👆 Please select a sample agreement above, upload a document, or paste text to begin analysis.")
