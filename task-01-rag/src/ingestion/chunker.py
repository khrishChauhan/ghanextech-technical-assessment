"""Recursive text chunking module with document and page metadata preservation.

Splits extracted document text into semantically cohesive passages while
preserving context continuity and source metadata (document name, page number, chunk index).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re

from src.ingestion.pdf_loader import LoadedDocument, PageContent
from src.utils.logger import get_logger

logger = get_logger("chunker")


@dataclass
class TextChunk:
    """Represents a chunk of text with comprehensive lineage metadata."""
    chunk_id: str
    text: str
    source_name: str
    page_number: int
    chunk_index: int
    char_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_citation_header(self) -> str:
        """Formatted citation header for context injection."""
        return f"[Document: {self.source_name} | Page: {self.page_number} | Chunk #{self.chunk_index}]"


class RecursiveTextChunker:
    """Splits document text into overlapping chunks using hierarchical separators.
    
    Default parameters:
        - chunk_size = 1000 characters (~200-250 tokens): Optimally matches paragraph
          length in reports, manuals, and technical documents, capturing full semantic units.
        - chunk_overlap = 200 characters (~40-50 tokens): Retains boundary context across
          adjacent chunks without redundant repetition or embedding bloat.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk_document(self, document: LoadedDocument) -> List[TextChunk]:
        """Chunk an entire LoadedDocument page by page, tracking page numbers and indices.

        Args:
            document: LoadedDocument instance containing extracted pages.

        Returns:
            List of TextChunk instances with populated metadata.
        """
        all_chunks: List[TextChunk] = []
        global_chunk_idx = 0

        for page in document.pages:
            if not page.text.strip():
                continue

            page_raw_chunks = self._split_text(page.text, self.separators)
            page_merged = self._merge_splits(page_raw_chunks)

            for p_idx, text_segment in enumerate(page_merged):
                cleaned_segment = text_segment.strip()
                if not cleaned_segment:
                    continue

                chunk_id = f"{self._sanitize_name(document.source_name)}_p{page.page_number}_c{p_idx}"
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    text=cleaned_segment,
                    source_name=document.source_name,
                    page_number=page.page_number,
                    chunk_index=global_chunk_idx,
                    char_count=len(cleaned_segment),
                    metadata={
                        "total_pages": document.total_pages,
                        "page_chunk_index": p_idx,
                    },
                )
                all_chunks.append(chunk)
                global_chunk_idx += 1

        logger.info(
            f"Chunked '{document.source_name}' into {len(all_chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})."
        )
        return all_chunks

    def chunk_text(
        self,
        text: str,
        source_name: str = "direct_text",
        page_number: int = 1,
    ) -> List[TextChunk]:
        """Chunk a standalone string for testing or ad-hoc input."""
        if not text.strip():
            return []

        splits = self._split_text(text, self.separators)
        merged = self._merge_splits(splits)

        chunks: List[TextChunk] = []
        for idx, segment in enumerate(merged):
            clean = segment.strip()
            if not clean:
                continue
            chunk_id = f"{self._sanitize_name(source_name)}_p{page_number}_c{idx}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=clean,
                    source_name=source_name,
                    page_number=page_number,
                    chunk_index=idx,
                    char_count=len(clean),
                )
            )
        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Hierarchically split text using the first matching separator."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Perform split on current separator
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        # Retain separator suffix except for empty string
        good_splits: List[str] = []
        for s in splits:
            if s:
                good_splits.append(s + separator if separator else s)

        # Recursively break down pieces that are still oversized
        for piece in good_splits:
            if len(piece) <= self.chunk_size:
                final_chunks.append(piece)
            elif new_separators:
                # Recurse with lower-level separators
                sub_chunks = self._split_text(piece, new_separators)
                final_chunks.extend(sub_chunks)
            else:
                # Force hard chop if no further separators exist
                for i in range(0, len(piece), self.chunk_size - self.chunk_overlap):
                    part = piece[i:i + self.chunk_size]
                    if part:
                        final_chunks.append(part)

        return final_chunks

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Combine raw splits into chunks up to chunk_size, respecting chunk_overlap."""
        merged_docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for split in splits:
            split_len = len(split)
            if total + split_len > self.chunk_size:
                if current_doc:
                    doc_text = "".join(current_doc)
                    merged_docs.append(doc_text)

                    # Calculate overlap: backtrack elements from current_doc
                    overlap_doc: List[str] = []
                    overlap_len = 0
                    for token in reversed(current_doc):
                        if overlap_len + len(token) <= self.chunk_overlap:
                            overlap_doc.insert(0, token)
                            overlap_len += len(token)
                        else:
                            break

                    current_doc = overlap_doc
                    total = overlap_len

            current_doc.append(split)
            total += split_len

        if current_doc:
            merged_docs.append("".join(current_doc))

        return merged_docs

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize file name for use in chunk IDs."""
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)[:30]
