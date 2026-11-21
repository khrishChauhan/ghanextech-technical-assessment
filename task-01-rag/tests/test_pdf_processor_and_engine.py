"""Unit tests for PDFProcessor and RAGEngine modules."""

import io
from pathlib import Path
import pytest
import pypdf

from pdf_processor import (
    PDFProcessor,
    EncryptedPDFError,
    EmptyPDFError,
    ScannedPDFError,
    CorruptPDFError,
)
from rag_engine import RAGEngine, STRICT_SYSTEM_PROMPT


@pytest.fixture
def sample_pdf_path():
    path = Path(__file__).parent.parent / "sample_docs" / "sample_policy.pdf"
    assert path.exists(), "sample_policy.pdf must exist for tests."
    return str(path)


def test_pdf_processor_extract_text(sample_pdf_path):
    processor = PDFProcessor()
    docs = processor.extract_text_from_pdf(sample_pdf_path)

    assert len(docs) == 3
    for doc in docs:
        assert doc.metadata["source"] == "sample_policy.pdf"
        assert doc.metadata["page"] in [1, 2, 3]
        assert len(doc.page_content) > 100


def test_pdf_processor_encrypted():
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("topsecret")
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)

    processor = PDFProcessor()
    with pytest.raises(EncryptedPDFError):
        processor.extract_text_from_pdf(buf, source_name="locked.pdf")


def test_pdf_processor_scanned():
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)

    processor = PDFProcessor()
    with pytest.raises(ScannedPDFError):
        processor.extract_text_from_pdf(buf, source_name="blank.pdf")


def test_pdf_processor_corrupt():
    processor = PDFProcessor()
    with pytest.raises(CorruptPDFError):
        processor.extract_text_from_pdf(b"invalid_bytes_xyz", source_name="bad.pdf")


def test_rag_engine_groq_provider_detection():
    engine = RAGEngine(api_key="gsk_test123456789")
    assert engine.provider == "groq"
    assert engine.dimension == 384
    assert "gpt-oss" in engine.llm_model or "qwen" in engine.llm_model


def test_rag_engine_index_search_and_clear():
    engine = RAGEngine(api_key="gsk_test_key")

    from langchain_core.documents import Document
    mock_docs = [
        Document(
            page_content="Data encryption must use AES-256-GCM standard for all at-rest storage.",
            metadata={"source": "sec.pdf", "page": 1},
        ),
        Document(
            page_content="Production deployment pipeline runs on Tuesday mornings.",
            metadata={"source": "sec.pdf", "page": 2},
        ),
    ]

    total_chunks = engine.index_documents(mock_docs)
    assert total_chunks > 0
    assert engine.total_chunks == total_chunks

    # Search
    results = engine.search("What is the encryption standard?", top_k=2)
    assert len(results) == 2
    assert "score" in results[0].metadata

    # Clear index
    engine.clear_index()
    assert engine.total_chunks == 0
    assert len(engine.chunks) == 0

    # Search on cleared index
    empty_results = engine.search("any query", top_k=2)
    assert empty_results == []


def test_strict_system_prompt_refusal_phrase():
    assert (
        "The provided document(s) do not contain information regarding this question."
        in STRICT_SYSTEM_PROMPT
    )
