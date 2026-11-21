"""Unit tests for prompt formatting, grounding, and RAG coordination."""

import numpy as np
import pytest

from src.ingestion.chunker import TextChunk
from src.vectorstore.faiss_store import FAISSVectorStore, SearchResult
from src.generation.prompts import (
    SYSTEM_PROMPT,
    format_context_block,
    build_user_prompt,
)
from src.generation.rag_engine import RAGEngine


class MockEmbedder:
    """Mock embedder providing deterministic vectors for testing without network."""

    def __init__(self, dimension: int = 8):
        self.dimension = dimension
        self.api_key = "test_api_key_mock"

    def embed_texts(self, texts):
        np.random.seed(len(texts))
        vecs = np.random.randn(len(texts), self.dimension).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1e-12, norms)

    def embed_query(self, query):
        np.random.seed(len(query))
        vec = np.random.randn(1, self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        return (vec / (norm if norm > 0 else 1.0)).astype(np.float32)


def test_system_prompt_anti_hallucination_directives():
    assert "STRICT GROUNDING" in SYSTEM_PROMPT
    assert "ANTI-HALLUCINATION" in SYSTEM_PROMPT
    assert "CITATIONS" in SYSTEM_PROMPT
    assert "Based on the provided documents, I could not find information regarding" in SYSTEM_PROMPT


def test_context_block_formatting():
    chunk = TextChunk(
        chunk_id="test_c1",
        text="The SLA target is 99.95% uptime.",
        source_name="sla_spec.pdf",
        page_number=1,
        chunk_index=0,
        char_count=32,
    )
    result = SearchResult(chunk=chunk, score=0.9234)

    block = format_context_block([result])
    assert "PASSAGE 1" in block
    assert "Source Document: sla_spec.pdf" in block
    assert "Page Number: 1" in block
    assert "0.9234" in block
    assert "The SLA target is 99.95% uptime." in block


def test_build_user_prompt():
    prompt = build_user_prompt(
        query="What is the uptime SLA?",
        context_block="Passage 1 content here.",
    )
    assert "CONTEXT PASSAGES:" in prompt
    assert "Passage 1 content here." in prompt
    assert "USER QUESTION:" in prompt
    assert "What is the uptime SLA?" in prompt


def test_rag_engine_retrieval():
    embedder = MockEmbedder(dimension=8)
    store = FAISSVectorStore(dimension=8)

    chunks = [
        TextChunk(
            chunk_id="c1",
            text="AES-256 encryption is required.",
            source_name="sec.pdf",
            page_number=1,
            chunk_index=0,
            char_count=30,
        ),
        TextChunk(
            chunk_id="c2",
            text="CI/CD runs on GitHub Actions.",
            source_name="devops.pdf",
            page_number=1,
            chunk_index=1,
            char_count=29,
        ),
    ]
    vectors = embedder.embed_texts([c.text for c in chunks])
    store.add_chunks(chunks, vectors)

    engine = RAGEngine(vector_store=store, embedder=embedder)
    results = engine.retrieve("What encryption is used?", top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
