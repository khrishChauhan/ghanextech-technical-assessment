"""Streamlit Web Interface for Ghanex Tech AI Document Assistant (RAG)."""

import os
import sys
from pathlib import Path
from typing import List

# Ensure task-01-rag directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st

from config.settings import get_settings
from src.embeddings.gemini_embeddings import GeminiEmbedder, GeminiAPIKeyError, GeminiEmbeddingError
from src.generation.rag_engine import RAGEngine
from src.ingestion.chunker import RecursiveTextChunker
from src.ingestion.pdf_loader import (
    PDFLoader,
    PDFProcessingError,
    EncryptedPDFError,
    EmptyPDFError,
    ScannedPDFError,
    CorruptPDFError,
)
from src.utils.logger import get_logger
from src.vectorstore.faiss_store import FAISSVectorStore

logger = get_logger("app")
settings = get_settings()

# Streamlit Page Setup
st.set_page_config(
    page_title="DocAssist RAG | Ghanex Tech",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished enterprise look
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
    }
    .main-header p {
        margin-top: 6px;
        margin-bottom: 0;
        opacity: 0.9;
        font-size: 1.05rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-green { background-color: #D1FAE5; color: #065F46; }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 14px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
    .citation-meta {
        font-weight: 600;
        color: #1E40AF;
        margin-bottom: 4px;
    }
    .sample-pill-btn {
        margin-right: 8px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize persistent Streamlit session states."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = FAISSVectorStore(dimension=settings.embedding_dimension)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "loaded_files" not in st.session_state:
        st.session_state.loaded_files = []  # List of dicts with file stats
    if "api_key" not in st.session_state:
        st.session_state.api_key = settings.get_api_key() or ""


init_session_state()


# ---------------- SIDEBAR CONFIGURATION ----------------
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")

    # API Key Input
    api_key_status = bool(st.session_state.api_key)
    if api_key_status:
        st.markdown(
            '<span class="badge-pill badge-green">✓ Gemini API Key Active</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="badge-pill badge-amber">⚠ API Key Required</span>',
            unsafe_allow_html=True,
        )

    user_api_key = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="Enter AIzaSy...",
        help="Reads automatically from .env or Streamlit secrets if configured.",
    )
    if user_api_key != st.session_state.api_key:
        st.session_state.api_key = user_api_key.strip()
        st.rerun()

    st.markdown("---")

    # Model & Generation Parameters
    st.markdown("### 🧠 Model Parameters")
    selected_model = st.selectbox(
        "LLM Model",
        options=settings.available_llm_models,
        index=0,
        help="Select Gemini model family.",
    )

    with st.expander("🛠️ Advanced RAG Hyperparameters", expanded=False):
        top_k = st.slider(
            "Top-K Retrieval",
            min_value=1,
            max_value=settings.max_top_k,
            value=settings.default_top_k,
            help="Number of context passages to inject into the LLM prompt.",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=settings.default_temperature,
            step=0.05,
            help="Lower values enforce deterministic, strictly grounded responses.",
        )
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=settings.min_chunk_size,
            max_value=settings.max_chunk_size,
            value=settings.default_chunk_size,
            step=100,
            help="Optimal size for semantic paragraph segmentation.",
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=50,
            max_value=min(400, chunk_size - 50),
            value=min(settings.default_chunk_overlap, chunk_size - 50),
            step=25,
            help="Continuity buffer between adjacent text chunks.",
        )

    st.markdown("---")

    # Quick Demo Sample PDF Loader
    st.markdown("### 🚀 Quick Demo")
    sample_pdf_path = BASE_DIR / "sample_docs" / "sample_policy.pdf"
    if sample_pdf_path.exists():
        if st.button("📁 Load Built-in Sample PDF", use_container_width=True):
            if not st.session_state.api_key:
                st.error("Please enter a Google Gemini API Key first.")
            else:
                with st.spinner("Indexing Ghanex Tech Engineering Handbook..."):
                    try:
                        loader = PDFLoader()
                        doc = loader.load_pdf(sample_pdf_path)
                        chunker = RecursiveTextChunker(
                            chunk_size=chunk_size, chunk_overlap=chunk_overlap
                        )
                        chunks = chunker.chunk_document(doc)

                        embedder = GeminiEmbedder(
                            api_key=st.session_state.api_key,
                            model_name=settings.default_embedding_model,
                        )
                        texts = [c.text for c in chunks]
                        embeddings = embedder.embed_texts(texts)

                        # Reset store and populate
                        st.session_state.vector_store.clear()
                        st.session_state.vector_store.add_chunks(chunks, embeddings)
                        st.session_state.loaded_files = [
                            {
                                "name": doc.source_name,
                                "pages": doc.total_pages,
                                "chars": doc.total_chars,
                                "chunks": len(chunks),
                            }
                        ]
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": (
                                    f"Loaded and indexed **{doc.source_name}** ({doc.total_pages} pages, "
                                    f"{len(chunks)} chunks). Ask me anything about engineering architecture, "
                                    "data governance, encryption standards, or incident SLAs!"
                                ),
                                "sources": [],
                            }
                        )
                        st.success("Sample policy document indexed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error indexing sample document: {e}")

    # Index Diagnostics
    stats = st.session_state.vector_store.get_stats()
    st.markdown("### 📊 Index Statistics")
    st.markdown(
        f"""
        - **Total Documents:** `{stats['total_documents']}`
        - **Indexed Chunks:** `{stats['total_chunks']}`
        - **Total Pages:** `{stats['total_pages']}`
        - **Embedding Dim:** `{stats['dimension']}`
        """
    )

    if st.button("🗑️ Clear Index & Reset Chat", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.chat_history = []
        st.session_state.loaded_files = []
        st.success("Vector store and chat history cleared.")
        st.rerun()


# ---------------- MAIN APPLICATION HEADER ----------------
st.markdown(
    """
    <div class="main-header">
        <h1>📄 Ghanex AI Document Assistant</h1>
        <p>Production-grade Retrieval-Augmented Generation (RAG) powered by Google Gemini & FAISS</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- DOCUMENT INGESTION TABS ----------------
tab_chat, tab_upload, tab_inspect = st.tabs(["💬 Assistant Chat", "📤 Upload Documents", "🔍 Vector Store Inspection"])

with tab_upload:
    st.subheader("Document Ingestion & Indexing")
    st.caption("Upload one or more PDF files. The system automatically extracts text, performs recursive character chunking, generates embeddings with text-embedding-004, and indexes them in FAISS.")

    uploaded_files = st.file_uploader(
        "Select PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload standard text-based PDF documents.",
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        index_btn = st.button("⚡ Index Uploaded PDFs", type="primary", use_container_width=True)

    if index_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        elif not st.session_state.api_key:
            st.error("Google Gemini API Key is missing. Please provide it in the sidebar.")
        else:
            loader = PDFLoader()
            chunker = RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            embedder = GeminiEmbedder(
                api_key=st.session_state.api_key,
                model_name=settings.default_embedding_model,
            )

            progress_bar = st.progress(0, text="Preparing ingestion...")
            total_files = len(uploaded_files)
            successfully_processed = 0

            for idx, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                progress_bar.progress(
                    (idx) / total_files,
                    text=f"Extracting text from '{file_name}' ({idx + 1}/{total_files})...",
                )

                try:
                    # Ingest PDF
                    doc = loader.load_pdf(uploaded_file, source_name=file_name)

                    # Chunk document
                    chunks = chunker.chunk_document(doc)

                    # Generate embeddings
                    progress_bar.progress(
                        (idx + 0.5) / total_files,
                        text=f"Generating embeddings for {len(chunks)} chunks in '{file_name}'...",
                    )
                    texts = [c.text for c in chunks]
                    embeddings = embedder.embed_texts(texts)

                    # Store in FAISS
                    st.session_state.vector_store.add_chunks(chunks, embeddings)

                    st.session_state.loaded_files.append(
                        {
                            "name": doc.source_name,
                            "pages": doc.total_pages,
                            "chars": doc.total_chars,
                            "chunks": len(chunks),
                        }
                    )
                    successfully_processed += 1

                except EncryptedPDFError as e:
                    st.error(f"🔒 **{file_name}**: {e}")
                except EmptyPDFError as e:
                    st.error(f"⚠️ **{file_name}**: {e}")
                except ScannedPDFError as e:
                    st.error(f"🖼️ **{file_name}**: {e}")
                except CorruptPDFError as e:
                    st.error(f"❌ **{file_name}**: {e}")
                except GeminiAPIKeyError as e:
                    st.error(f"🔑 API Key Error: {e}")
                    break
                except GeminiEmbeddingError as e:
                    st.error(f"🌐 Embedding API Error: {e}")
                    break
                except Exception as e:
                    st.error(f"Unexpected error processing '{file_name}': {e}")

            progress_bar.progress(1.0, text="Ingestion complete!")
            if successfully_processed > 0:
                st.success(
                    f"Successfully indexed {successfully_processed} document(s). "
                    f"Vector store now holds {st.session_state.vector_store.count} total chunks."
                )
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": f"Successfully indexed **{successfully_processed}** new document(s). You can now ask questions about them in the chat tab!",
                        "sources": [],
                    }
                )

    if st.session_state.loaded_files:
        st.markdown("#### Currently Indexed Files:")
        for f in st.session_state.loaded_files:
            st.markdown(
                f"- **{f['name']}** — `{f['pages']} pages`, `{f['chars']} characters`, `{f['chunks']} chunks`"
            )


with tab_inspect:
    st.subheader("Vector Store Inspection & Metadata Browser")
    st.caption("Inspect indexed passages, chunk IDs, source document bindings, and internal representations.")

    if st.session_state.vector_store.count == 0:
        st.info("The vector store is currently empty. Ingest documents or load the sample PDF to view chunks.")
    else:
        st.write(f"Total Chunks: **{len(st.session_state.vector_store.chunks)}**")
        for idx, chunk in enumerate(st.session_state.vector_store.chunks):
            with st.expander(f"Chunk #{idx} | {chunk.source_name} (Page {chunk.page_number}) | ID: {chunk.chunk_id}"):
                st.markdown(f"**Length:** {chunk.char_count} chars")
                st.code(chunk.text, language="text")


with tab_chat:
    # Sample Questions Banner
    if st.session_state.vector_store.count > 0:
        st.markdown("##### 💡 Suggested Questions:")
        sample_questions = [
            "What cryptographic standard is required for data at rest?",
            "What are the resolution SLA targets for Sev-1 incidents?",
            "What days and times are production deployments allowed?",
            "What is Ghanex Tech's policy on time travel?",  # Hallucination test
        ]
        cols = st.columns(len(sample_questions))
        for i, q in enumerate(sample_questions):
            with cols[i]:
                if st.button(q, key=f"sq_{i}", use_container_width=True):
                    st.session_state.pending_query = q
                    st.rerun()

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"📚 Retrieved Context & Sources ({len(message['sources'])} passages)"):
                    for s_idx, src in enumerate(message["sources"], start=1):
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <div class="citation-meta">Passage {s_idx} • {src.chunk.source_name} (Page {src.chunk.page_number}) • Cosine Similarity: {src.score:.4f}</div>
                                <div>{src.chunk.text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Check for pending query or chat input
    user_query = None
    if "pending_query" in st.session_state and st.session_state.pending_query:
        user_query = st.session_state.pending_query
        st.session_state.pending_query = None
    else:
        chat_prompt = st.chat_input("Ask a question grounded in your uploaded documents...")
        if chat_prompt:
            user_query = chat_prompt

    if user_query:
        # Display user question
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant generation
        with st.chat_message("assistant"):
            if not st.session_state.api_key:
                err_msg = "Please configure your Google Gemini API Key in the sidebar to generate answers."
                st.error(err_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": err_msg, "sources": []})
            elif st.session_state.vector_store.count == 0:
                err_msg = "No documents have been loaded into the vector index yet. Please upload a PDF or click 'Load Built-in Sample PDF' in the sidebar."
                st.warning(err_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": err_msg, "sources": []})
            else:
                embedder = GeminiEmbedder(
                    api_key=st.session_state.api_key,
                    model_name=settings.default_embedding_model,
                )
                engine = RAGEngine(
                    vector_store=st.session_state.vector_store,
                    embedder=embedder,
                    settings=settings,
                )

                with st.spinner("Retrieving relevant passages and synthesizing answer..."):
                    stream_generator, retrieved_sources = engine.generate_answer_stream(
                        query=user_query,
                        top_k=top_k,
                        temperature=temperature,
                        model_name=selected_model,
                    )

                # Stream response
                full_response = st.write_stream(stream_generator)

                # Show citations
                if retrieved_sources:
                    with st.expander(f"📚 Retrieved Context & Sources ({len(retrieved_sources)} passages)"):
                        for s_idx, src in enumerate(retrieved_sources, start=1):
                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    <div class="citation-meta">Passage {s_idx} • {src.chunk.source_name} (Page {src.chunk.page_number}) • Cosine Similarity: {src.score:.4f}</div>
                                    <div>{src.chunk.text}</div>
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
