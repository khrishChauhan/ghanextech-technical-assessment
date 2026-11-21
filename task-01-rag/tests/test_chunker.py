"""Unit tests for recursive text chunking and metadata preservation."""

import pytest
from src.ingestion.chunker import RecursiveTextChunker, TextChunk
from src.ingestion.pdf_loader import LoadedDocument, PageContent


def test_chunker_parameter_validation():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        RecursiveTextChunker(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        RecursiveTextChunker(chunk_size=500, chunk_overlap=-10)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        RecursiveTextChunker(chunk_size=500, chunk_overlap=500)


def test_chunk_standalone_text():
    text = (
        "Paragraph 1 discusses system reliability and uptime targets. " * 10
        + "\n\n"
        + "Paragraph 2 discusses encryption standards with AES-256-GCM. " * 10
    )
    chunker = RecursiveTextChunker(chunk_size=400, chunk_overlap=80)
    chunks = chunker.chunk_text(text, source_name="test_doc.pdf", page_number=2)

    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, TextChunk)
        assert chunk.source_name == "test_doc.pdf"
        assert chunk.page_number == 2
        assert len(chunk.text) <= 450  # Allows slight margin for word boundary preservation
        assert chunk.char_count == len(chunk.text)


def test_chunk_loaded_document():
    doc = LoadedDocument(
        source_name="annual_report.pdf",
        total_pages=2,
        total_chars=1200,
        pages=[
            PageContent(
                page_number=1,
                text="Page 1 header.\n\nFirst quarter financial results exceeded expectations.",
                char_count=65,
                source_name="annual_report.pdf",
            ),
            PageContent(
                page_number=2,
                text="Page 2 header.\n\nSecond quarter expansion into AI operations.",
                char_count=61,
                source_name="annual_report.pdf",
            ),
        ],
    )

    chunker = RecursiveTextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert "Page 1" in chunks[0].text
    assert chunks[1].page_number == 2
    assert "Page 2" in chunks[1].text


def test_empty_text_chunking():
    chunker = RecursiveTextChunker()
    chunks = chunker.chunk_text("   \n\n   ")
    assert chunks == []
