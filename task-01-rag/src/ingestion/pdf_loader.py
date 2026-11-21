"""PDF ingestion and text extraction module with robust error handling.

Handles multi-page layouts, password-protected files, empty files,
and scanned (image-only) PDFs.
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, List, Optional, Union
import pypdf

from src.utils.logger import get_logger

logger = get_logger("pdf_loader")


class PDFProcessingError(Exception):
    """Base exception for PDF ingestion errors."""
    pass


class EncryptedPDFError(PDFProcessingError):
    """Raised when a PDF is password-protected or encrypted."""
    pass


class EmptyPDFError(PDFProcessingError):
    """Raised when a PDF contains no pages or zero extractable content."""
    pass


class ScannedPDFError(PDFProcessingError):
    """Raised when a PDF contains only raster images and no machine-readable text."""
    pass


class CorruptPDFError(PDFProcessingError):
    """Raised when the PDF file stream is malformed or corrupted."""
    pass


@dataclass
class PageContent:
    """Represents text and metadata extracted from a single PDF page."""
    page_number: int  # 1-indexed
    text: str
    char_count: int
    source_name: str


@dataclass
class LoadedDocument:
    """Represents a fully ingested document."""
    source_name: str
    total_pages: int
    total_chars: int
    pages: List[PageContent] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenated text of all pages separated by newlines."""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


class PDFLoader:
    """Extracts clean text and page-level metadata from PDF files.
    
    Supports file paths, byte streams, and Streamlit UploadedFile objects.
    """

    def __init__(self, min_text_threshold: int = 25):
        """
        Args:
            min_text_threshold: Minimum total characters expected across document
                                to consider it text-based rather than scanned/empty.
        """
        self.min_text_threshold = min_text_threshold

    def load_pdf(
        self,
        file_input: Union[str, Path, bytes, BinaryIO],
        source_name: Optional[str] = None,
    ) -> LoadedDocument:
        """Extract text page-by-page from a PDF source.

        Args:
            file_input: Path to PDF file, raw bytes, or file-like buffer.
            source_name: Optional human-readable name of the file.

        Returns:
            LoadedDocument containing page-level extracted text and statistics.

        Raises:
            EncryptedPDFError: If the document is password-protected.
            EmptyPDFError: If the document contains zero pages.
            ScannedPDFError: If no extractable text is detected.
            CorruptPDFError: If parsing fails due to stream corruption.
        """
        # Determine source name
        if source_name is None:
            if isinstance(file_input, (str, Path)):
                source_name = Path(file_input).name
            elif hasattr(file_input, "name"):
                source_name = getattr(file_input, "name")
            else:
                source_name = "uploaded_document.pdf"

        # Normalize stream
        stream = self._get_stream(file_input)

        try:
            reader = pypdf.PdfReader(stream)
        except pypdf.errors.PdfReadError as e:
            logger.error(f"Failed to read PDF '{source_name}': {e}")
            raise CorruptPDFError(
                f"Document '{source_name}' is corrupt or not a valid PDF file. ({e})"
            )
        except Exception as e:
            logger.error(f"Unexpected error loading '{source_name}': {e}")
            raise CorruptPDFError(f"Could not parse '{source_name}': {e}")

        # Check encryption
        if reader.is_encrypted:
            # Check if it opens with empty password (some PDFs have dummy encryption)
            try:
                decrypted = reader.decrypt("")
                if decrypted == 0:  # Decryption failed
                    raise EncryptedPDFError(
                        f"Document '{source_name}' is password-protected. "
                        "Please remove the password before uploading."
                    )
            except Exception:
                raise EncryptedPDFError(
                    f"Document '{source_name}' is password-protected. "
                    "Please remove the password before uploading."
                )

        num_pages = len(reader.pages)
        if num_pages == 0:
            raise EmptyPDFError(f"Document '{source_name}' contains zero pages.")

        pages: List[PageContent] = []
        total_chars = 0

        for idx, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(
                    f"Page {idx} in '{source_name}' extraction warning: {e}"
                )
                raw_text = ""

            cleaned_text = self._clean_text(raw_text)
            char_count = len(cleaned_text)
            total_chars += char_count

            pages.append(
                PageContent(
                    page_number=idx,
                    text=cleaned_text,
                    char_count=char_count,
                    source_name=source_name,
                )
            )

        # Scanned document detection: if total characters are below threshold
        if total_chars < self.min_text_threshold:
            # Attempt secondary fallback with pdfplumber if available
            pdfplumber_text = self._fallback_pdfplumber(stream, source_name)
            if pdfplumber_text and len(pdfplumber_text.strip()) >= self.min_text_threshold:
                # Fallback succeeded
                return self._build_document_from_text(
                    source_name=source_name,
                    raw_text=pdfplumber_text,
                    num_pages=num_pages,
                )

            raise ScannedPDFError(
                f"Document '{source_name}' appears to be scanned or contains only images. "
                f"Extracted only {total_chars} characters. Machine-readable text is required for RAG."
            )

        logger.info(
            f"Successfully parsed '{source_name}': {num_pages} pages, {total_chars} characters."
        )

        return LoadedDocument(
            source_name=source_name,
            total_pages=num_pages,
            total_chars=total_chars,
            pages=pages,
        )

    def _get_stream(self, file_input: Union[str, Path, bytes, BinaryIO]) -> io.BytesIO:
        """Convert input to a seekable BytesIO stream."""
        if isinstance(file_input, (str, Path)):
            with open(file_input, "rb") as f:
                return io.BytesIO(f.read())
        elif isinstance(file_input, bytes):
            return io.BytesIO(file_input)
        elif hasattr(file_input, "read"):
            content = file_input.read()
            if hasattr(file_input, "seek"):
                file_input.seek(0)
            return io.BytesIO(content)
        else:
            raise ValueError(f"Unsupported file input type: {type(file_input)}")

    def _clean_text(self, text: str) -> str:
        """Normalize extracted text (strip null bytes, excessive spaces, keep paragraph structure)."""
        if not text:
            return ""
        # Remove null characters
        text = text.replace("\x00", "")
        # Normalize non-breaking spaces
        text = text.replace("\xa0", " ")
        # Strip trailing carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text.strip()

    def _fallback_pdfplumber(self, stream: io.BytesIO, source_name: str) -> Optional[str]:
        """Secondary extraction fallback using pdfplumber."""
        try:
            import pdfplumber
            stream.seek(0)
            with pdfplumber.open(stream) as pdf:
                all_text = []
                for p in pdf.pages:
                    extracted = p.extract_text()
                    if extracted:
                        all_text.append(extracted)
                return "\n\n".join(all_text)
        except Exception as e:
            logger.debug(f"pdfplumber fallback skipped or failed for '{source_name}': {e}")
            return None

    def _build_document_from_text(
        self, source_name: str, raw_text: str, num_pages: int
    ) -> LoadedDocument:
        """Create a LoadedDocument from single concatenated fallback text."""
        cleaned = self._clean_text(raw_text)
        page = PageContent(
            page_number=1,
            text=cleaned,
            char_count=len(cleaned),
            source_name=source_name,
        )
        return LoadedDocument(
            source_name=source_name,
            total_pages=num_pages,
            total_chars=len(cleaned),
            pages=[page],
        )
