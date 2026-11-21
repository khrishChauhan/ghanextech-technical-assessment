"""FAISS vector store implementation with cosine similarity search and metadata management."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import faiss

from src.ingestion.chunker import TextChunk
from src.utils.logger import get_logger

logger = get_logger("vectorstore")


@dataclass
class SearchResult:
    """Retrieved text chunk paired with similarity metric."""
    chunk: TextChunk
    score: float  # Cosine similarity score [-1.0, 1.0]

    @property
    def formatted_citation(self) -> str:
        """User-friendly citation string."""
        return f"{self.chunk.source_name} (Page {self.chunk.page_number})"


class FAISSVectorStore:
    """In-memory vector store powered by FAISS IndexFlatIP (Inner Product).
    
    When queried with L2-normalized vectors, Inner Product is mathematically
    identical to Cosine Similarity.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.chunks: List[TextChunk] = []

    @property
    def count(self) -> int:
        """Total number of indexed vectors."""
        return self.index.ntotal

    def add_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray):
        """Add chunks and corresponding embeddings into the FAISS index.

        Args:
            chunks: List of TextChunk objects.
            embeddings: 2D numpy array of float32 embeddings (N, dimension).
        """
        if len(chunks) == 0:
            logger.warning("No chunks provided to add_chunks.")
            return

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding shape (N, {self.dimension}), got {embeddings.shape}"
            )

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
            )

        # Ensure float32 format
        vectors = embeddings.astype(np.float32)
        self.index.add(vectors)
        self.chunks.extend(chunks)

        logger.info(
            f"Added {len(chunks)} chunks to FAISS index. Total indexed: {self.count}"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        min_score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Perform semantic search for most relevant chunks.

        Args:
            query_embedding: 1D or 2D numpy array representing query embedding.
            top_k: Number of most similar chunks to return.
            min_score_threshold: Optional minimum cosine similarity cutoff.

        Returns:
            List of SearchResult objects sorted by descending similarity score.
        """
        if self.count == 0:
            logger.warning("Search attempted on empty vector store.")
            return []

        # Reshape to (1, dimension)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_vector = query_embedding.astype(np.float32)
        effective_k = min(top_k, self.count)

        distances, indices = self.index.search(query_vector, effective_k)

        results: List[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            score = float(dist)
            if min_score_threshold is not None and score < min_score_threshold:
                continue

            chunk = self.chunks[idx]
            results.append(SearchResult(chunk=chunk, score=score))

        return results

    def clear(self):
        """Reset the vector index and remove all stored chunks."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        logger.info("Cleared FAISS vector store and reset index.")

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics of current index state."""
        doc_names = set(c.source_name for c in self.chunks)
        total_chars = sum(c.char_count for c in self.chunks)
        pages_by_doc: Dict[str, set] = {}
        for c in self.chunks:
            pages_by_doc.setdefault(c.source_name, set()).add(c.page_number)

        total_pages = sum(len(p) for p in pages_by_doc.values())

        return {
            "total_chunks": self.count,
            "total_documents": len(doc_names),
            "document_names": list(doc_names),
            "total_pages": total_pages,
            "total_characters": total_chars,
            "dimension": self.dimension,
        }

    def save_local(self, directory_path: str):
        """Persist FAISS index and chunk metadata to disk."""
        target_dir = Path(directory_path)
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "index.faiss"
        metadata_file = target_dir / "metadata.json"

        faiss.write_index(self.index, str(index_file))

        metadata = [asdict(c) for c in self.chunks]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Persisted FAISS index and {len(metadata)} chunks to {target_dir}")

    def load_local(self, directory_path: str):
        """Load FAISS index and chunk metadata from disk."""
        target_dir = Path(directory_path)
        index_file = target_dir / "index.faiss"
        metadata_file = target_dir / "metadata.json"

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Missing index or metadata in {directory_path}")

        self.index = faiss.read_index(str(index_file))

        with open(metadata_file, "r", encoding="utf-8") as f:
            raw_metadata = json.load(f)

        self.chunks = [TextChunk(**item) for item in raw_metadata]
        logger.info(f"Loaded FAISS index with {self.count} chunks from {target_dir}")
