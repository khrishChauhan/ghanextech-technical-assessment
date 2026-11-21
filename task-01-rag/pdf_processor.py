"""PDF Processor module for RAG Document Assistant.

Extracts text from PDF documents and attaches page-level metadata
while gracefully handling password protection, scanned/empty files, and corruption.
"""

import io
from pathlib import Path
from typing import BinaryIO, List, Optional, Union
import pypdf

try:
    from langchain_core.documents import Document
except ImportError:
    from dataclasses import dataclass, field
    from typing import Dict, Any

    @dataclass
    class Document:
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)


class PDFProcessingError(Exception):
    """Base exception for PDF processing errors."""
    pass


class EncryptedPDFError(PDFProcessingError):
    """Raised when a PDF is password-protected or encrypted."""
    pass


class EmptyPDFError(PDFProcessingError):
    """Raised when a PDF contains no pages."""
    pass


class ScannedPDFError(PDFProcessingError):
    """Raised when a PDF contains only raster images and no extractable text."""
    pass


class CorruptPDFError(PDFProcessingError):
    """Raised when a PDF file stream is corrupted or unreadable."""
    pass


class PDFProcessor:
    """Handles PDF ingestion, text extraction, and metadata tagging."""

    def __init__(self, min_text_threshold: int = 25):
        """
        Args:
            min_text_threshold: Minimum total characters expected across document
                                to consider it machine-readable text rather than scanned/empty.
        """
        self.min_text_threshold = min_text_threshold

    def extract_text_from_pdf(
        self,
        uploaded_file: Union[str, Path, bytes, BinaryIO],
        source_name: Optional[str] = None,
    ) -> List[Document]:
        """Extract text page-by-page from an uploaded PDF file and return LangChain Documents.

        Args:
            uploaded_file: File path, byte stream, or Streamlit UploadedFile object.
            source_name: Optional explicit name of the source file.

        Returns:
            List[Document]: Each document contains page_content and metadata
                            with 'source' (file name) and 'page' (1-based index).

        Raises:
            EncryptedPDFError: If the document is password-protected.
            EmptyPDFError: If the document contains zero pages.
            ScannedPDFError: If no extractable text is detected.
            CorruptPDFError: If file is corrupted or not a valid PDF.
        """
        # Resolve source name
        if source_name is None:
            if isinstance(uploaded_file, (str, Path)):
                source_name = Path(uploaded_file).name
            elif hasattr(uploaded_file, "name"):
                source_name = getattr(uploaded_file, "name")
            else:
                source_name = "uploaded_document.pdf"

        # Obtain seekable byte stream
        stream = self._to_bytes_io(uploaded_file)

        # Parse with pypdf
        try:
            reader = pypdf.PdfReader(stream)
        except pypdf.errors.PdfReadError as e:
            raise CorruptPDFError(
                f"File '{source_name}' is corrupted or not a valid PDF: {e}"
            )
        except Exception as e:
            raise CorruptPDFError(f"Could not parse '{source_name}': {e}")

        # Check for encryption / password lock
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
                if decrypted == 0:
                    raise EncryptedPDFError(
                        f"PDF '{source_name}' is password-protected. Please remove the password before uploading."
                    )
            except Exception:
                raise EncryptedPDFError(
                    f"PDF '{source_name}' is password-protected. Please remove the password before uploading."
                )

        num_pages = len(reader.pages)
        if num_pages == 0:
            raise EmptyPDFError(f"PDF '{source_name}' contains zero pages.")

        documents: List[Document] = []
        total_extracted_chars = 0

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""

            cleaned_text = self._clean_text(raw_text)
            total_extracted_chars += len(cleaned_text)

            if cleaned_text:
                documents.append(
                    Document(
                        page_content=cleaned_text,
                        metadata={
                            "source": source_name,
                            "page": page_idx,  # 1-based page number
                        },
                    )
                )

        # Scanned / raster image detection
        if total_extracted_chars < self.min_text_threshold:
            # Attempt secondary fallback via pdfplumber if installed
            fallback_docs = self._fallback_pdfplumber(stream, source_name)
            if fallback_docs:
                return fallback_docs

            raise ScannedPDFError(
                f"Document '{source_name}' contains no extractable text ({total_extracted_chars} characters found). "
                "The PDF appears to be a scanned image or empty. Please upload a document with selectable text."
            )

        return documents

    def _to_bytes_io(self, uploaded_file: Union[str, Path, bytes, BinaryIO]) -> io.BytesIO:
        """Ensure input is converted into a seekable BytesIO stream."""
        if isinstance(uploaded_file, (str, Path)):
            with open(uploaded_file, "rb") as f:
                return io.BytesIO(f.read())
        elif isinstance(uploaded_file, bytes):
            return io.BytesIO(uploaded_file)
        elif hasattr(uploaded_file, "read"):
            content = uploaded_file.read()
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            return io.BytesIO(content)
        else:
            raise ValueError(f"Unsupported file type: {type(uploaded_file)}")

    def _clean_text(self, text: str) -> str:
        """Strip null bytes, normalize line breaks, and trim extraneous whitespace."""
        if not text:
            return ""
        text = text.replace("\x00", "")
        text = text.replace("\xa0", " ")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text.strip()

    def _fallback_pdfplumber(self, stream: io.BytesIO, source_name: str) -> Optional[List[Document]]:
        """Fallback text extractor using pdfplumber for complex layouts."""
        try:
            import pdfplumber
            stream.seek(0)
            fallback_docs = []
            total_chars = 0
            with pdfplumber.open(stream) as pdf:
                for idx, page in enumerate(pdf.pages, start=1):
                    extracted = page.extract_text() or ""
                    cleaned = self._clean_text(extracted)
                    total_chars += len(cleaned)
                    if cleaned:
                        fallback_docs.append(
                            Document(
                                page_content=cleaned,
                                metadata={"source": source_name, "page": idx},
                            )
                        )
            if total_chars >= self.min_text_threshold:
                return fallback_docs
        except Exception:
            pass
        return None
