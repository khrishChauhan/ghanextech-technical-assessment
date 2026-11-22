"""RAG Engine module coordinating text chunking, dense embeddings,
FAISS vector storage, semantic search, and grounded generation with Groq and Gemini support.
"""

import json
import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple
import faiss
import numpy as np
import requests

try:
    from langchain_core.documents import Document
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class Document:
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

STRICT_SYSTEM_PROMPT = """You are a strict, factual AI Document Assistant.
Your mission is to answer user questions based strictly and exclusively on the provided context passages.

CRITICAL GROUNDING INSTRUCTIONS:
1. Grounding: Rely ONLY on the provided context passages. Do NOT extrapolate, speculate, or utilize external knowledge.
2. Anti-Hallucination: If the context does not contain enough information to answer the question, explicitly state:
   "The provided document(s) do not contain information regarding this question."
3. Page Citations: Reference the exact page numbers where facts were found (e.g., [Doc: filename, Page: X]).
4. Formatting: Present information in clear, clean markdown with bullet points where appropriate.
"""


class RAGEngine:
    """Retrieval-Augmented Generation engine managing the end-to-end RAG lifecycle.
    
    Supports Groq (e.g. openai/gpt-oss-120b, qwen/qwen3.8-27b) and Google Gemini (gemini-2.0-flash).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Args:
            api_key: Groq API key (starts with 'gsk_') or Gemini API key.
            llm_model: Target LLM model name.
            chunk_size: Size of text chunks (characters).
                Chunk size of 1000 characters (~200-250 tokens) corresponds to the
                average length of a cohesive informational paragraph. This preserves
                complete semantic units while staying well within model attention spans.
            chunk_overlap: Overlap between adjacent chunks (characters).
                Chunk overlap of 200 characters (~40-50 tokens) prevents cutting
                sentences, entities, or definitions in half across chunk boundaries,
                maintaining contextual continuity without causing redundant token bloat.
        """
        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        # Detect provider
        self.provider = self._detect_provider(self.api_key)

        # Default model selection
        if llm_model:
            self.llm_model = llm_model
        else:
            self.llm_model = "openai/gpt-oss-120b" if self.provider == "groq" else "gemini-2.0-flash"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

        # Vector dimension: 384 for fast local dense embedder, 768 for gemini
        self.dimension = 768 if self.provider == "gemini" else 384

        # In-memory FAISS Index with Inner Product (Cosine similarity on normalized vectors)
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[Document] = []
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._genai_client = None

    def __getattr__(self, name: str) -> Any:
        """Safeguard for legacy cached session instances."""
        if name == "provider":
            key = getattr(self, "api_key", None)
            val = self._detect_provider(key)
            self.provider = val
            return val
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _detect_provider(self, key: Optional[str]) -> str:
        """Auto-detect provider based on API key prefix."""
        if not key:
            if os.getenv("GROQ_API_KEY"):
                return "groq"
            return "gemini"
        if key.startswith("gsk_"):
            return "groq"
        return "gemini"

    def set_api_key(self, api_key: str):
        """Update active API key and configure provider."""
        self.api_key = api_key.strip()
        self.provider = self._detect_provider(self.api_key)
        if self.provider == "groq" and "gemini" in self.llm_model:
            self.llm_model = "openai/gpt-oss-120b"
        elif self.provider == "gemini" and ("gpt-oss" in self.llm_model or "qwen" in self.llm_model):
            self.llm_model = "gemini-2.0-flash"

        new_dim = 768 if self.provider == "gemini" else 384
        if new_dim != self.dimension:
            self.dimension = new_dim
            self.clear_index()

        self._genai_client = None

    def clear_index(self):
        """Completely reset the FAISS vector index and stored chunks.
        
        Guarantees that subsequent uploads start from a completely clean slate
        with zero stale vector leakage.
        """
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks.clear()
        self._tfidf_vectorizer = None

    @property
    def total_chunks(self) -> int:
        """Total number of indexed vectors."""
        return self.index.ntotal

    def index_documents(self, documents: List[Document]) -> int:
        """Split documents into chunks, compute embeddings, and index into FAISS.

        Args:
            documents: List of Document objects extracted from uploaded PDFs.

        Returns:
            int: Number of chunks added to the index.
        """
        if not documents:
            return 0

        # Chunk documents while preserving existing metadata (source, page)
        raw_chunks = self.text_splitter.split_documents(documents)
        if not raw_chunks:
            return 0

        texts = [c.page_content for c in raw_chunks]

        # Fit vectorizer for local dense embeddings if in Groq mode
        if self.provider != "gemini":
            self._tfidf_vectorizer = TfidfVectorizer(max_features=self.dimension, sublinear_tf=True)
            self._tfidf_vectorizer.fit(texts)

        embeddings = self._embed_texts(texts)

        # Add vectors to FAISS
        self.index.add(embeddings)
        self.chunks.extend(raw_chunks)

        return len(raw_chunks)

    def search(self, query: str, top_k: int = 4) -> List[Document]:
        """Perform semantic similarity search for a query string.

        Args:
            query: Natural language question.
            top_k: Number of relevant passages to retrieve (default: 4).

        Returns:
            List[Document]: Retrieved top-k documents sorted by similarity.
        """
        if self.total_chunks == 0 or not query.strip():
            return []

        query_vector = self._embed_texts([query.strip()])  # shape (1, dimension)
        k = min(top_k, self.total_chunks)
        distances, indices = self.index.search(query_vector, k)

        results: List[Document] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            original_chunk = self.chunks[idx]
            doc_copy = Document(
                page_content=original_chunk.page_content,
                metadata={**original_chunk.metadata, "score": float(score)},
            )
            results.append(doc_copy)

        return results

    def generate_answer(self, query: str, top_k: int = 4) -> Tuple[str, List[Document]]:
        """Generate a grounded, cited answer using retrieved context passages."""
        stream_gen, sources = self.generate_answer_stream(query, top_k=top_k)
        tokens = [token for token in stream_gen]
        return "".join(tokens), sources

    def generate_answer_stream(
        self, query: str, top_k: int = 4
    ) -> Tuple[Generator[str, None, None], List[Document]]:
        """Stream the grounded answer token-by-token for interactive UI responses."""
        sources = self.search(query, top_k=top_k)

        if not sources:
            def fallback():
                yield "The provided document(s) do not contain information regarding this question."
            return fallback(), []

        # Construct context block
        context_parts = []
        for i, doc in enumerate(sources, start=1):
            source = doc.metadata.get("source", "Document")
            page = doc.metadata.get("page", "Unknown")
            score = doc.metadata.get("score", 0.0)
            context_parts.append(
                f"[Passage {i}] Source: {source} | Page: {page} | Similarity: {score:.4f}\n{doc.page_content}\n"
            )

        context_str = "\n".join(context_parts)
        prompt_text = (
            f"CONTEXT PASSAGES:\n{context_str}\n\n"
            f"USER QUESTION:\n{query}\n\n"
            f"Please answer based strictly on the context above. If not found, output: "
            f"\"The provided document(s) do not contain information regarding this question.\""
        )

        if self.provider == "groq":
            stream = self._call_groq_stream(prompt_text)
        else:
            stream = self._call_gemini_stream(prompt_text)

        return stream, sources

    def _call_groq_stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream response from Groq Cloud API."""
        if not self.api_key:
            yield "Groq API Key is missing. Please set GROQ_API_KEY in .env or provide it in the sidebar."
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        }
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "stream": True,
        }

        try:
            with requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=45,
            ) as response:
                if response.status_code != 200:
                    yield f"Error from Groq API (HTTP {response.status_code}): {response.text}"
                    return

                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except Exception:
                            pass
        except Exception as e:
            yield f"\n\n[Error communicating with Groq: {e}]"

    def _call_gemini_stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream response from Google Gemini API."""
        if not self.api_key:
            yield "Gemini API Key is missing. Please set GEMINI_API_KEY in .env or provide it in the sidebar."
            return

        clean_model = self.llm_model.replace("models/", "")
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content_stream(
                model=clean_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=STRICT_SYSTEM_PROMPT,
                    temperature=0.2,
                ),
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n\n[Error generating Gemini response: {e}]"

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Compute L2-normalized float32 embeddings."""
        if self.provider == "gemini":
            return self._embed_gemini(texts)
        else:
            return self._embed_local(texts)

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Fast local dense text embeddings (384 dimensions) with L2 normalization."""
        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer(max_features=self.dimension, sublinear_tf=True)
            self._tfidf_vectorizer.fit(texts)

        raw = self._tfidf_vectorizer.transform(texts).toarray().astype(np.float32)
        # Pad to self.dimension if vocabulary is smaller
        if raw.shape[1] < self.dimension:
            raw = np.pad(raw, ((0, 0), (0, self.dimension - raw.shape[1])))

        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        return (raw / norms).astype(np.float32)

    def _embed_gemini(self, texts: List[str], batch_size: int = 20) -> np.ndarray:
        """Generate Gemini embeddings using models/text-embedding-004."""
        from google import genai
        client = genai.Client(api_key=self.api_key)
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + batch_size]]
            for attempt in range(1, 4):
                try:
                    resp = client.models.embed_content(
                        model="text-embedding-004",
                        contents=batch,
                    )
                    all_embeddings.extend([e.values for e in resp.embeddings])
                    break
                except Exception as e:
                    if attempt == 3:
                        raise e
                    time.sleep(2 ** attempt)

        vectors = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        return (vectors / norms).astype(np.float32)
