"""RAG Engine module coordinating text chunking, Gemini embeddings,
FAISS vector storage, semantic search, and grounded generation.
"""

import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple
import faiss
import numpy as np

try:
    from langchain_core.documents import Document
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class Document:
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)

from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    """Retrieval-Augmented Generation engine managing the end-to-end RAG lifecycle."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: str = "gemini-2.0-flash",
        embedding_model: str = "models/text-embedding-004",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        dimension: int = 768,
    ):
        """
        Args:
            api_key: Google Gemini API key.
            llm_model: Target Gemini model for grounded generation.
            embedding_model: Embedding model identifier.
            chunk_size: Size of text chunks (characters).
                Chunk size of 1000 characters (~200-250 tokens) corresponds to the
                average length of a cohesive informational paragraph. This preserves
                complete semantic units while staying well within model attention spans.
            chunk_overlap: Overlap between adjacent chunks (characters).
                Chunk overlap of 200 characters (~40-50 tokens) prevents cutting
                sentences, entities, or definitions in half across chunk boundaries,
                maintaining contextual continuity without causing redundant token bloat.
            dimension: Embedding vector dimension (768 for text-embedding-004).
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dimension = dimension

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

        # In-memory FAISS Index with Inner Product (Cosine similarity on normalized vectors)
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[Document] = []
        self._genai_client = None

    def set_api_key(self, api_key: str):
        """Update active API key."""
        self.api_key = api_key.strip()
        self._genai_client = None

    def _get_client(self):
        """Lazy initialization of Google GenAI client."""
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Please set GEMINI_API_KEY in .env or provide it in the UI."
            )
        if self._genai_client is None:
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.api_key)
            except ImportError:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self._genai_client = legacy_genai
        return self._genai_client

    def clear_index(self):
        """Completely reset the FAISS vector index and stored chunks.
        
        Guarantees that subsequent uploads start from a completely clean slate
        with zero stale vector leakage.
        """
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks.clear()

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

        query_vector = self._embed_texts([query.strip()])  # shape (1, 768)
        k = min(top_k, self.total_chunks)
        distances, indices = self.index.search(query_vector, k)

        results: List[Document] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            original_chunk = self.chunks[idx]
            # Create a copy with score in metadata
            doc_copy = Document(
                page_content=original_chunk.page_content,
                metadata={**original_chunk.metadata, "score": float(score)},
            )
            results.append(doc_copy)

        return results

    def generate_answer(self, query: str, top_k: int = 4) -> Tuple[str, List[Document]]:
        """Generate a grounded, cited answer using retrieved context passages.

        Args:
            query: User's question.
            top_k: Number of chunks to retrieve.

        Returns:
            Tuple[str, List[Document]]: (Generated answer, Retrieved source chunks)
        """
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

        stream = self._call_gemini_stream(prompt_text)
        return stream, sources

    def _call_gemini_stream(self, prompt: str) -> Generator[str, None, None]:
        """Call Gemini API with streaming."""
        client = self._get_client()
        clean_model = self.llm_model.replace("models/", "")

        try:
            if hasattr(client, "models"):
                # google-genai SDK
                from google.genai import types
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
            else:
                # google-generativeai SDK
                model = client.GenerativeModel(
                    model_name=clean_model,
                    system_instruction=STRICT_SYSTEM_PROMPT,
                )
                response = model.generate_content(
                    prompt,
                    stream=True,
                    generation_config={"temperature": 0.2},
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
        except Exception as e:
            yield f"\n\n[Error generating response: {e}]"

    def _embed_texts(self, texts: List[str], batch_size: int = 20) -> np.ndarray:
        """Generate L2-normalized float32 embeddings for a list of texts."""
        client = self._get_client()
        clean_model = self.embedding_model.replace("models/", "")

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + batch_size]]
            for attempt in range(1, 4):
                try:
                    if hasattr(client, "models"):
                        resp = client.models.embed_content(
                            model=clean_model,
                            contents=batch,
                        )
                        embeddings = [e.values for e in resp.embeddings]
                        all_embeddings.extend(embeddings)
                        break
                    else:
                        resp = client.embed_content(
                            model=f"models/{clean_model}",
                            content=batch,
                        )
                        all_embeddings.extend(resp["embedding"])
                        break
                except Exception as e:
                    if attempt == 3:
                        raise e
                    time.sleep(2 ** attempt)

        vectors = np.array(all_embeddings, dtype=np.float32)
        # L2 normalize so Inner Product = Cosine Similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        return (vectors / norms).astype(np.float32)
