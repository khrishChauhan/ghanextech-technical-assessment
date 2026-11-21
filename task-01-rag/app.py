"""Ghanex Tech AI Document Assistant - Streamlit Application.

Production-ready RAG application integrating PDFProcessor, RAGEngine,
and an interactive streaming conversational interface powered by Groq and Google Gemini.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Add task directory to python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pdf_processor import (
    PDFProcessor,
    EncryptedPDFError,
    EmptyPDFError,
    ScannedPDFError,
    CorruptPDFError,
)
from rag_engine import RAGEngine

# Page configuration
st.set_page_config(
    page_title="Ghanex DocAssist RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #2563EB 100%);
        padding: 24px 28px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .main-header p {
        margin-top: 6px;
        margin-bottom: 0;
        color: #E2E8F0;
        font-size: 1.05rem;
    }
    .status-badge-active {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-badge-inactive {
        background-color: #FEECDC;
        color: #9C4221;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .citation-box {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .citation-title {
        font-weight: 700;
        color: #1E3A8A;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    .citation-text {
        font-size: 0.9rem;
        color: #334155;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_default_api_key() -> str:
    """Retrieve API key from environment or Streamlit secrets (Groq or Gemini)."""
    key = os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        try:
            if hasattr(st, "secrets"):
                key = (
                    st.secrets.get("GROQ_API_KEY")
                    or st.secrets.get("GEMINI_API_KEY")
                    or st.secrets.get("GOOGLE_API_KEY")
                )
        except Exception:
            pass
    return key or ""


# Initialize session states
if "api_key" not in st.session_state:
    st.session_state.api_key = get_default_api_key()

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine(api_key=st.session_state.api_key)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ Assistant Configuration")

    # API Key Input
    has_api_key = bool(st.session_state.api_key)
    provider_name = "Groq" if st.session_state.rag_engine.provider == "groq" else "Gemini"

    if has_api_key:
        st.markdown(
            f'<span class="status-badge-active">● {provider_name} API Key Connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge-inactive">○ API Key Missing</span>',
            unsafe_allow_html=True,
        )

    user_key = st.text_input(
        "API Key (Groq `gsk_...` or Gemini)",
        value=st.session_state.api_key,
        type="password",
        placeholder="Enter gsk_... or AIzaSy...",
        help="Reads automatically from .env or Streamlit secrets.",
    )
    if user_key != st.session_state.api_key:
        st.session_state.api_key = user_key.strip()
        st.session_state.rag_engine.set_api_key(user_key.strip())
        st.rerun()

    st.markdown("---")

    # Model Selection
    st.markdown("### 🧠 LLM Model")
    if st.session_state.rag_engine.provider == "groq":
        model_options = [
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b",
        ]
    else:
        model_options = ["gemini-2.0-flash", "gemini-1.5-flash"]

    selected_model = st.selectbox(
        "Active Model",
        options=model_options,
        index=0,
        help="Select inference model.",
    )
    if selected_model != st.session_state.rag_engine.llm_model:
        st.session_state.rag_engine.llm_model = selected_model

    st.markdown("---")

    # Multi-PDF File Uploader
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDF Document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF files.",
    )

    col1, col2 = st.columns(2)
    with col1:
        index_clicked = st.button("Index Documents", type="primary", use_container_width=True)
    with col2:
        clear_clicked = st.button("Clear / Reset", use_container_width=True)

    # Built-in Sample Loader
    sample_pdf_path = BASE_DIR / "sample_docs" / "sample_policy.pdf"
    if sample_pdf_path.exists():
        if st.button("🚀 Load Sample Handbook PDF", use_container_width=True):
            if not st.session_state.api_key:
                st.error("Please enter an API Key first.")
            else:
                with st.spinner("Indexing sample policy PDF..."):
                    try:
                        processor = PDFProcessor()
                        sample_docs = processor.extract_text_from_pdf(sample_pdf_path)
                        st.session_state.rag_engine.clear_index()
                        num_chunks = st.session_state.rag_engine.index_documents(sample_docs)
                        st.session_state.indexed_files = [
                            {"name": "sample_policy.pdf", "pages": len(sample_docs), "chunks": num_chunks}
                        ]
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": (
                                    f"Loaded and indexed **sample_policy.pdf** ({len(sample_docs)} pages, {num_chunks} chunks). "
                                    f"Using **{selected_model}** on {provider_name} with FAISS retrieval. "
                                    "Ask me anything about engineering architecture, encryption standards, or incident SLAs!"
                                ),
                                "sources": [],
                            }
                        )
                        st.success(f"Indexed {num_chunks} chunks from sample policy.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load sample PDF: {e}")

    # Handle indexing
    if index_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        elif not st.session_state.api_key:
            st.error("API Key is required. Please provide it above.")
        else:
            processor = PDFProcessor()
            all_documents = []
            successful_file_names = []

            for f in uploaded_files:
                file_name = f.name
                try:
                    docs = processor.extract_text_from_pdf(f, source_name=file_name)
                    all_documents.extend(docs)
                    successful_file_names.append(
                        {"name": file_name, "pages": len(docs), "chunks": 0}
                    )
                except EncryptedPDFError as e:
                    st.error(f"🔒 **Password Protected:** {e}")
                except EmptyPDFError as e:
                    st.warning(f"⚠️ **Empty File:** {e}")
                except ScannedPDFError as e:
                    st.warning(f"🖼️ **Scanned / Image-Only PDF:** {e}")
                except CorruptPDFError as e:
                    st.error(f"❌ **Corrupted File:** {e}")
                except Exception as e:
                    st.error(f"Error processing '{file_name}': {e}")

            if all_documents:
                with st.spinner(f"Indexing {len(all_documents)} pages into FAISS vector store..."):
                    try:
                        # Clear old index to prevent stale vector leakage
                        st.session_state.rag_engine.clear_index()
                        total_chunks = st.session_state.rag_engine.index_documents(all_documents)

                        st.session_state.indexed_files = [
                            {"name": f["name"], "pages": f["pages"], "chunks": total_chunks}
                            for f in successful_file_names
                        ]
                        st.success(f"Successfully indexed {len(successful_file_names)} file(s) into {total_chunks} chunks!")
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": f"Indexed **{len(successful_file_names)}** file(s) with **{total_chunks}** passages. You can now ask questions about your documents!",
                                "sources": [],
                            }
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Embedding / Indexing Error: {e}")

    # Handle clear / reset
    if clear_clicked:
        st.session_state.rag_engine.clear_index()
        st.session_state.chat_history = []
        st.session_state.indexed_files = []
        st.success("Index and conversation history cleared.")
        st.rerun()

    # Index Status
    st.markdown("---")
    st.markdown("### 📊 Document Index Status")
    total_indexed_chunks = st.session_state.rag_engine.total_chunks
    embed_desc = "Local Dense (384-dim)" if st.session_state.rag_engine.provider == "groq" else "Gemini (768-dim)"
    st.write(f"- **Provider:** `{provider_name}`")
    st.write(f"- **Indexed Chunks:** `{total_indexed_chunks}`")
    st.write(f"- **Embeddings:** `{embed_desc}`")
    st.write(f"- **Vector Index:** `FAISS (IndexFlatIP)`")

    if st.session_state.indexed_files:
        st.markdown("**Active Files:**")
        for f in st.session_state.indexed_files:
            st.markdown(f"• `{f['name']}` ({f['pages']} pages)")


# ---------------- MAIN CHAT AREA ----------------
st.markdown(
    """
    <div class="main-header">
        <h1>📄 Ghanex AI Document Assistant</h1>
        <p>Production Retrieval-Augmented Generation (RAG) powered by Groq, Gemini & FAISS</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Suggested queries if index is populated
if st.session_state.rag_engine.total_chunks > 0 and len(st.session_state.chat_history) <= 1:
    st.markdown("##### 💡 Suggested Questions:")
    sample_queries = [
        "What cryptographic standard is required for data at rest?",
        "What are the resolution SLA targets for Sev-1 incidents?",
        "When are production deployments allowed to take place?",
        "What is Ghanex Tech's policy on interstellar space travel?",  # Anti-hallucination test
    ]
    cols = st.columns(len(sample_queries))
    for i, q in enumerate(sample_queries):
        with cols[i]:
            if st.button(q, key=f"sample_q_{i}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()

# Display conversation history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"View Source Citations ({len(msg['sources'])} chunks)"):
                for idx, doc in enumerate(msg["sources"], start=1):
                    src_name = doc.metadata.get("source", "Document")
                    page_num = doc.metadata.get("page", "Unknown")
                    score = doc.metadata.get("score", 0.0)
                    st.markdown(
                        f"""
                        <div class="citation-box">
                            <div class="citation-title">Passage {idx} • {src_name} (Page {page_num}) • Similarity Score: {score:.4f}</div>
                            <div class="citation-text">{doc.page_content}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Determine user input (from pending sample query or chat input)
user_query = None
if "pending_query" in st.session_state and st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    chat_prompt = st.chat_input("Ask a question about your uploaded documents...")
    if chat_prompt:
        user_query = chat_prompt

# Process query
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not st.session_state.api_key:
            err_text = "Please enter your API Key in the sidebar to generate answers."
            st.error(err_text)
            st.session_state.chat_history.append({"role": "assistant", "content": err_text, "sources": []})
        elif st.session_state.rag_engine.total_chunks == 0:
            warn_text = "The vector store is empty. Please upload and index PDF documents or click 'Load Sample Handbook PDF' in the sidebar."
            st.warning(warn_text)
            st.session_state.chat_history.append({"role": "assistant", "content": warn_text, "sources": []})
        else:
            with st.spinner("Searching document context and generating answer..."):
                stream_gen, retrieved_sources = st.session_state.rag_engine.generate_answer_stream(
                    query=user_query, top_k=4
                )

            # Stream response in real time
            full_response = st.write_stream(stream_gen)

            # Render expandable citations
            if retrieved_sources:
                with st.expander(f"View Source Citations ({len(retrieved_sources)} chunks)"):
                    for idx, doc in enumerate(retrieved_sources, start=1):
                        src_name = doc.metadata.get("source", "Document")
                        page_num = doc.metadata.get("page", "Unknown")
                        score = doc.metadata.get("score", 0.0)
                        st.markdown(
                            f"""
                            <div class="citation-box">
                                <div class="citation-title">Passage {idx} • {src_name} (Page {page_num}) • Similarity Score: {score:.4f}</div>
                                <div class="citation-text">{doc.page_content}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "sources": retrieved_sources,
                }
            )
