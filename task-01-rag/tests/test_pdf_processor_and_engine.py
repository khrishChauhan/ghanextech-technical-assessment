"""Unit tests for PDFProcessor and RAGEngine top-level modules."""

import io
from pathlib import Path
import numpy as np
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


def test_rag_engine_index_and_clear():
    engine = RAGEngine(api_key="mock_key")

    # Mock embed_texts to return random unit vectors
    def mock_embed(texts, batch_size=20):
        vecs = np.random.randn(len(texts), 768).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1e-12, norms)

    engine._embed_texts = mock_embed

    from langchain_core.documents import Document
    mock_docs = [
        Document(
            page_content="Data encryption must use AES-256-GCM. " * 30,
            metadata={"source": "sec.pdf", "page": 1},
        ),
        Document(
            page_content="Deployment occurs on Tuesday mornings. " * 30,
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
