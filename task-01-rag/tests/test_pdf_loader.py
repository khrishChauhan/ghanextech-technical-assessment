"""Unit tests for PDF ingestion and error handling."""

import io
from pathlib import Path
import pytest
import pypdf

from src.ingestion.pdf_loader import (
    PDFLoader,
    EncryptedPDFError,
    EmptyPDFError,
    ScannedPDFError,
    CorruptPDFError,
)


@pytest.fixture
def sample_pdf_path():
    path = Path(__file__).parent.parent / "sample_docs" / "sample_policy.pdf"
    assert path.exists(), "sample_policy.pdf must exist for tests."
    return str(path)


def test_load_valid_multipage_pdf(sample_pdf_path):
    loader = PDFLoader()
    doc = loader.load_pdf(sample_pdf_path)

    assert doc.source_name == "sample_policy.pdf"
    assert doc.total_pages == 3
    assert doc.total_chars > 3000
    assert len(doc.pages) == 3

    # Check page 1 contents
    assert doc.pages[0].page_number == 1
    assert "Ghanex Tech" in doc.pages[0].text
    assert doc.pages[0].char_count > 500

    # Check page 2 contents
    assert doc.pages[1].page_number == 2
    assert "AES-256-GCM" in doc.pages[1].text

    # Check page 3 contents
    assert doc.pages[2].page_number == 3
    assert "text-embedding-004" in doc.pages[2].text


def test_corrupt_pdf_handling():
    loader = PDFLoader()
    garbage_bytes = b"NOT_A_REAL_PDF_HEADER_1234567890"

    with pytest.raises(CorruptPDFError):
        loader.load_pdf(garbage_bytes, source_name="corrupt.pdf")


def test_empty_or_scanned_pdf_handling():
    # Create a 1-page PDF with no text (blank page)
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=300, height=300)
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)

    loader = PDFLoader(min_text_threshold=25)
    with pytest.raises(ScannedPDFError) as exc_info:
        loader.load_pdf(buffer, source_name="blank.pdf")

    assert "scanned or contains only images" in str(exc_info.value)


def test_encrypted_pdf_handling():
    # Create password-protected PDF
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    writer.encrypt("supersecretpassword")
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)

    loader = PDFLoader()
    with pytest.raises(EncryptedPDFError) as exc_info:
        loader.load_pdf(buffer, source_name="locked.pdf")

    assert "password-protected" in str(exc_info.value)
