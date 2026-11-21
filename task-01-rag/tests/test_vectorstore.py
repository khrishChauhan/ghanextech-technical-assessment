"""Unit tests for FAISS vector store and similarity search."""

import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.ingestion.chunker import TextChunk
from src.vectorstore.faiss_store import FAISSVectorStore


@pytest.fixture
def sample_chunks():
    return [
        TextChunk(
            chunk_id="doc1_p1_c0",
            text="Ghanex Tech engineers observe zero-trust architecture.",
            source_name="handbook.pdf",
            page_number=1,
            chunk_index=0,
            char_count=52,
        ),
        TextChunk(
            chunk_id="doc1_p2_c0",
            text="AES-256-GCM encryption is mandatory for all data at rest.",
            source_name="handbook.pdf",
            page_number=2,
            chunk_index=1,
            char_count=57,
        ),
        TextChunk(
            chunk_id="doc1_p3_c0",
            text="Production deployments are executed on Tuesday mornings.",
            source_name="handbook.pdf",
            page_number=3,
            chunk_index=2,
            char_count=57,
        ),
    ]


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1e-12, norms)).astype(np.float32)


def test_add_and_search(sample_chunks):
    store = FAISSVectorStore(dimension=8)
    np.random.seed(42)
    raw_vecs = np.random.randn(3, 8).astype(np.float32)
    norm_vecs = normalize(raw_vecs)

    store.add_chunks(sample_chunks, norm_vecs)
    assert store.count == 3

    # Query with exact second vector (should match chunk 1 with score 1.0)
    query_vec = norm_vecs[1:2]
    results = store.search(query_vec, top_k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "doc1_p2_c0"
    assert abs(results[0].score - 1.0) < 1e-4


def test_shape_mismatch_error(sample_chunks):
    store = FAISSVectorStore(dimension=8)
    wrong_dim_vecs = np.random.randn(3, 16).astype(np.float32)

    with pytest.raises(ValueError, match="Expected embedding shape"):
        store.add_chunks(sample_chunks, wrong_dim_vecs)


def test_clear_vector_store(sample_chunks):
    store = FAISSVectorStore(dimension=8)
    raw_vecs = np.random.randn(3, 8).astype(np.float32)
    store.add_chunks(sample_chunks, normalize(raw_vecs))

    assert store.count == 3
    store.clear()
    assert store.count == 0
    assert len(store.chunks) == 0

    # Search on empty store should return empty list gracefully
    results = store.search(np.random.randn(1, 8).astype(np.float32))
    assert results == []


def test_save_and_load_local(sample_chunks):
    temp_dir = tempfile.mkdtemp()
    try:
        store = FAISSVectorStore(dimension=8)
        raw_vecs = np.random.randn(3, 8).astype(np.float32)
        norm_vecs = normalize(raw_vecs)
        store.add_chunks(sample_chunks, norm_vecs)

        store.save_local(temp_dir)

        # Load into new store instance
        new_store = FAISSVectorStore(dimension=8)
        new_store.load_local(temp_dir)

        assert new_store.count == 3
        assert len(new_store.chunks) == 3
        assert new_store.chunks[1].chunk_id == "doc1_p2_c0"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
